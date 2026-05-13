
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from rag_query import run_rag


st.session_state.setdefault("chat_history", [])

st.title("Personal RAG Assistant")



def get_chat_history():
    history_text = ""
    for msg in st.session_state.chat_history:
        history_text += f"{msg['role']}: {msg['content']}\n"
    return history_text


for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


question = st.chat_input("Ask a question about your documents...")

if question:

    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)


    history = get_chat_history()

    response = run_rag(question, history)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response
    })

    with st.chat_message("assistant"):
        st.write(response)


# #streamlit run streamlit_app.py