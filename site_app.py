import os

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import SitemapLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🛜",
)

with st.sidebar:
    openai_api_key = st.text_input(
        "Enter your OpenAI API Key",
        placeholder="sk-...",
        key="openai_api_key",
    )
    st.markdown("[Github](https://github.com/sseregit/FULLSTACK-GPT-CHALLENGE)")


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        append_message_to_session(message, role)


def append_message_to_session(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def get_ai_response(response):
    gathered = None
    for chunk in response:
        yield chunk

        if gathered is None:
            gathered = chunk
        else:
            gathered += chunk


def clean_text(text: str) -> str:
    return (
        text.replace('\n', ' ')
        .replace('\r', ' ')
        .replace('\t', ' ')
        .replace('\xa0', ' ')
        .replace('  ', ' ')
        .strip()
    )


def parse_page(soup):
    # main-pane
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()

    main_pane = soup.find("div", {"class": "main-pane"})
    if main_pane is None:
        return clean_text(str(soup))
    return clean_text(main_pane.get_text())


def load_website(url):
    base_path = './faiss_data'

    vector_store = None
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    if os.path.exists(base_path):
        vector_store = FAISS.load_local(
            base_path,
            embeddings,
        )
    else:
        loader = SitemapLoader(
            url,
            filter_urls=[
                r"^(.*\/(ai-gateway|vectorize|workers-ai)\/).*",
            ],
            parsing_function=parse_page,

        )
        loader.requests_per_second = 2

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000,
            chunk_overlap=200,
        )
        docs = loader.load_and_split(text_splitter=splitter)
        batch_size = 100

        for i in range(0, len(docs), batch_size):
            batch = docs[i: i + batch_size]
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embeddings)
            else:
                vector_store.add_documents(batch)

        os.makedirs(base_path, exist_ok=True)
        vector_store.save_local(base_path)

    if vector_store is None:
        raise Exception("No vector store")

    return vector_store.as_retriever()


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        당신은 웹 문서 내용을 이해하고 답변하는 전문 AI 어시스턴트입니다.
        주어진 문맥(context)을 참고하여 가장 정확한 답변을 제공합니다.
        
        규칙:
        1. {context} 안의 내용을 바탕으로 답변하세요.  
        2. 문맥에 답이 없다면, "제공된 문서 안에서는 관련 정보를 찾을 수 없습니다."라고 답하세요.
        
        항상 명확하고 간결하게 답변하세요.
        """
    ),
    (
        "human",
        """            
        {question}
        """
    ),
])

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if openai_api_key:
    llm = ChatOpenAI(openai_api_key=openai_api_key)

    retriever = load_website("https://developers.cloudflare.com/sitemap-0.xml")

    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = (
                {
                    "context": retriever,
                    "question": RunnablePassthrough(),
                }
                | prompt
                | llm
        )
        response = chain.stream(message)
        with st.spinner("Generating..."):
            with st.chat_message("ai"):
                ai_placeholder = st.empty()
                full_response = ""

                for chunk in response:
                    full_response += chunk.content
                    ai_placeholder.markdown(full_response)

            append_message_to_session(full_response, "ai")
