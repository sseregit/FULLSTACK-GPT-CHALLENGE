import json

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import WikipediaRetriever

st.set_page_config(
    page_title="QuizGPT Turbo",
    page_icon="🧐",
)

function_schema = {
  "name": "generate_quiz",
  "description": "Generate a list of quiz questions with 4 options (one correct).",
  "parameters": {
    "type": "object",
    "properties": {
      "difficulty": {
        "type": "string",
        "enum": ["Easy", "Medium", "Hard"]
      },
      "questions": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "question": {"type": "string"},
            "answers": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "answer": {"type": "string"},
                  "correct": {"type": "boolean"}
                },
                "required": ["answer", "correct"]
              }
            }
          },
          "required": ["question", "answers"]
        }
      }
    },
    "required": ["difficulty", "questions"]
  }
}

with st.sidebar:
  openai_api_key = st.text_input(
      "Enter your OpenAI API Key",
      placeholder="sk-...",
      key="openai_api_key",
  )
  difficulty = st.sidebar.selectbox("Select Difficulty", ["Easy", "Hard"])
  st.markdown("[Github](https://github.com/sseregit/FULLSTACK-GPT-CHALLENGE)")

if openai_api_key:
  llm = ChatOpenAI(
      openai_api_key=openai_api_key
  ).bind(
      functions=[function_schema],
      function_call={"name": "generate_quiz"}
  )


@st.cache_data(show_spinner="Making quiz...")
def generate_quiz_with_function(context, difficulty):
  prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are QuizGPT, a teacher that creates multiple-choice questions."),
    ("user",
     "Generate 5 {difficulty} quiz questions based ONLY on this text:\n\n{context}")
  ])

  chain = prompt | llm

  return chain.invoke({"context": context, "difficulty": difficulty})


def format_docs(docs):
  return "\n\n".join(document.page_content for document in docs)


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
  retriever = WikipediaRetriever(top_k_results=5)
  docs = retriever.get_relevant_documents(term)
  return docs


if openai_api_key:
  topic = st.text_input("Search Wikipedia...", placeholder="topic")
  if topic:
    docs = wiki_search(topic)

    if docs:
      context = format_docs(docs)
      response = generate_quiz_with_function(context, difficulty)

      parsed = json.loads(
          response.additional_kwargs['function_call']['arguments'])

      difficulty, questions = parsed["difficulty"], parsed["questions"]

      with st.form("quiz_form"):
        st.subheader(f"{difficulty} Level Quiz")

        for idx, question in enumerate(questions):
          st.write(question["question"])
          selected = st.radio(
              f"Select an option for Q{idx + 1}:",
              [ans["answer"] for ans in question["answers"]],
              index=None,
              key=f"radio_{idx}"
          )
          st.write(f"Currently selected for Q{idx + 1}: {selected}")

        submit_button = st.form_submit_button("Submit Answers")

      if submit_button:
        correct_count = 0
        total = len(questions)

        for idx, question in enumerate(questions):
          selected = st.session_state.get(f"radio_{idx}")
          if selected:
            for answer in question["answers"]:
              if answer["answer"] == selected and answer["correct"]:
                correct_count += 1
                break

        st.success(f"{correct_count}/{total} correct!")
        if correct_count == total:
          st.balloons()
        else:
          if st.button("🔁 Retake Quiz"):
            for idx in range(len(questions)):
              st.session_state.pop(f"radio_{idx}", None)
            st.experimental_rerun()
