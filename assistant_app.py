from datetime import datetime

import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
# Wikipedia
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities import WikipediaAPIWrapper
from openai import OpenAI
from openai.types.beta.assistant_stream_event import ThreadRunRequiresAction, ThreadMessageDelta
import json


# load_dotenv()

def search_wikipedia(parameters):
    api_wrapper = WikipediaAPIWrapper(top_k_results=2)
    tool = WikipediaQueryRun(api_wrapper=api_wrapper)
    return tool._run(parameters["query"])


def search_duckduckgo(parameters):
    duckduck_api_wrapper = DuckDuckGoSearchAPIWrapper(time="m")
    docs = duckduck_api_wrapper.results(parameters["query"], max_results=1)
    links = []

    for doc in docs:
        links.append(doc["link"])

    loader = WebBaseLoader(
        web_path=links,
        bs_get_text_kwargs={
            "strip": True,
        }
    )

    load = loader.load()
    return "\n".join([content.page_content for content in load])


def save_file(parameters):
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    filename = f"research_result_{now}.txt"
    with open(filename, 'w', encoding="utf-8") as f:
        f.write(parameters["content"] + "\n")
    return parameters["content"]


functions = [
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "query에 대한 답을 Wikipedia를 활용해 검색합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query you will search",
                    },
                },
                "required": ["query"],
            },

        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_duckduckgo",
            "description": "query에 대한 답을 DuckDuckGo를 활용해 검색합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query you will search",
                    },
                },
                "required": ["query"]
            },

        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_file",
            "description": """ 리서치 작업이 완료되면 반드시 사용해야 하는 도구입니다.
                            Wikipedia와 DuckDuckGo에서 수집한 모든 정보를 정리하여 content 파라미터에 전달하면 자동으로 파일로 저장됩니다.
                            주의: content에는 파일명이 아닌, 실제로 조사한 내용 요약/정리해서 넣어야 합니다.
                            """,
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "검색 도구들로부터 수집한 전체 리서치 내용과 결과를 여기에 전달하세요. 파일명이 아닌 실제 조사한 내용 전체를 포함해야 합니다.",
                    },
                },
                "required": ["content"]
            },

        }
    },
]

functions_map = {
    "search_wikipedia": search_wikipedia,
    "search_duckduckgo": search_duckduckgo,
    "save_file": save_file,
}


@st.cache_data(show_spinner="Generating... Assistant....")
def create_assistant(openai_api_key):
    assistant = client.beta.assistants.create(
        name="search assistant",
        instructions="""
              당신은 주제에 대한 광범위한 조사를 수행하는 AI 어시스턴트입니다.

              **필수 워크플로우:**
              1. 사용자 질문을 받으면 Wikipedia와 DuckDuckGo로 조사
              2. 조사 결과를 종합하고 정리
              3. save_file 도구를 사용하여 결과를 반드시 파일에 저장
              4. "✓ 파일에 저장되었습니다:
              research_result_YYYY-MM-DDTHH:MM:SS.txt" 형식으로 알림
              5. 저장된 내용의 요약을 사용자에게 제공

              **주의:**
              - 모든 조사 결과는 반드시 파일로 저장되어야 함
              - 파일 저장 없이 조사만 하면 안 됨
              """,
        model="gpt-4-turbo",
        tools=functions,
    )

    return assistant


st.set_page_config(
    page_title="OpenAI Assistants (Graduation Project)",
    page_icon="👏",
)

with st.sidebar:
    openai_api_key = st.text_input(
        "Enter your OpenAI API Key",
        placeholder="sk-...",
        key="openai_api_key",
    )
    st.markdown("[Github](https://github.com/sseregit/FULLSTACK-GPT-CHALLENGE)")

st.session_state["messages"] = []

if openai_api_key:
    try:
        client = OpenAI(api_key=openai_api_key)
        assistant = create_assistant(openai_api_key)

        if assistant:
            st.chat_message("assistant").write("Please search for the keyword.")

            if user_input := st.chat_input():
                st.chat_message("user").write(user_input)

                with client.beta.threads.create_and_run_stream(
                    assistant_id=assistant.id,
                    thread={
                        "messages": [{
                            "role": "user",
                            "content": user_input,
                        }]
                    },
                ) as stream:
                    for event in stream:
                        if isinstance(event, ThreadRunRequiresAction):
                            run = event.data
                            outputs = []
                            for action in run.required_action.submit_tool_outputs.tool_calls:
                                action_id = action.id
                                function = action.function
                                outputs.append(
                                    {
                                        "output": functions_map[function.name](json.loads(function.arguments)),
                                        "tool_call_id": action_id,
                                    }
                                )
                            with client.beta.threads.runs.submit_tool_outputs_stream(tool_outputs=outputs,
                                                                                     run_id=run.id,
                                                                                     thread_id=run.thread_id) as next_stream:
                                for followup_event in next_stream:
                                    if isinstance(followup_event, ThreadRunRequiresAction):
                                        run = followup_event.data
                                        outputs = []
                                        for action in run.required_action.submit_tool_outputs.tool_calls:
                                            action_id = action.id
                                            function = action.function
                                            outputs.append(
                                                {
                                                    "output": functions_map[action.function.name](json.loads(action.function.arguments)),
                                                    "tool_call_id": action.id,
                                                }
                                            )
                                        with client.beta.threads.runs.submit_tool_outputs_stream(tool_outputs=outputs,
                                                                                                 run_id=run.id,
                                                                                                 thread_id=run.thread_id) as next_stream:
                                            full_text = ""
                                            with st.chat_message("ai"):
                                                ai_placeholder = st.empty()
                                                for followup_event in next_stream:
                                                    if isinstance(followup_event, ThreadMessageDelta):
                                                        delta_text = followup_event.data.delta.content[0].text.value
                                                        full_text += delta_text
                                                        ai_placeholder.markdown(full_text)

    except Exception as e:
        st.error(f"please change openai_api_key {e}")
        del st.session_state["openai_api_key"]
