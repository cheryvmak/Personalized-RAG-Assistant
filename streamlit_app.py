

import streamlit as st
# import requests


# # Page config
# st.set_page_config(page_title="Personal RAG Assistant", layout="centered")

# # Title & description
# st.title("My Personal RAG Assistant")
# st.write("Ask questions about my CV and personal documents.")

# st.markdown(
#     """
#     **Assistant Persona:**  
#     I answer questions strictly based on Okemakinde Sherif's personal documents  
#     (CV, experience, education, projects).
#     """
# )


# # Input box
# question = st.text_input("Ask a question")

# # Button
# if st.button("Ask"):
#     if question.strip() == "":
#         st.warning("Please enter a question.")
#     else:
#         with st.spinner("Thinking..."):
#             response = requests.post(
#                 "http://127.0.0.1:8000/query",
#                 json={"question": question}
#             )

#         if response.status_code == 200:
#             data = response.json()

#             st.subheader("Answer")
#             st.write(data["answer"])
#         else:
#             st.error("Something went wrong. Please try again.")



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