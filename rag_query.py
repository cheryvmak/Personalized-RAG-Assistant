

# import os
# from dotenv import load_dotenv

# from langchain_chroma import Chroma
# #from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings

# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import (
#     RunnableParallel,
#     RunnablePassthrough,
#     RunnableLambda
# )
# from langchain_core.output_parsers import StrOutputParser

# from langchain_groq import ChatGroq


# # ENV
# load_dotenv()


# # EMBEDDINGS 
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )


# # VECTOR STORE
# vectorstore = Chroma(
#     persist_directory="chroma_db1",
#     embedding_function=embeddings
# )

# retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# # GROQ LLM (PRIMARY)
# llm = ChatGroq(
#     model="llama-3.3-70b-versatile",
#     temperature=0
# )


# # PROMPT
# prompt = ChatPromptTemplate.from_messages([
#     (
#         "system",
#         """
# You are a helpful AI assistant answering questions using retrieved documents.

# Rules:
# - Use the provided context as the primary source.
# - If the answer exists in the context, answer clearly and confidently.
# - If partial information exists, infer carefully from the context.
# - Only say information is missing if it truly does not appear in the retrieved text.
# """
#     ),
#     (
#         "human",
#         "Context:\n{context}\n\nQuestion:\n{question}"
#     )
# ])


# # FORMAT DOCS
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)


# # RAG PIPELINE
# rag_chain = (
#     RunnableParallel(
#         context=retriever | RunnableLambda(format_docs),
#         question=RunnablePassthrough()
#     )
#     | prompt
#     | llm
#     | StrOutputParser()
# )


# # ENTRY FUNCTION
# def run_rag(question: str):
#     return rag_chain.invoke(question)








# import os
# import streamlit as st
# from dotenv import load_dotenv

# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings

# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import (
#     RunnableParallel,
#     RunnablePassthrough,
#     RunnableLambda
# )
# from langchain_core.output_parsers import StrOutputParser

# from langchain_groq import ChatGroq
# import streamlit as st


# # STREAMLIT MEMORY
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # ENV
# load_dotenv()

# # EMBEDDINGS

# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )


# # VECTOR DATABASE

# vectorstore = Chroma(
#     persist_directory="chroma_db1",
#     embedding_function=embeddings
# )

# retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# # LLM (GROQ)

# llm = ChatGroq(
#     model="llama-3.3-70b-versatile",
#     temperature=0
# )


# # FORMAT DOCS

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)


# # CHAT HISTORY FORMATTER

# def get_chat_history():
#     history_text = ""
#     for msg in st.session_state.chat_history:
#         history_text += f"{msg['role']}: {msg['content']}\n"
#     return history_text


# # PROMPT 

# prompt = ChatPromptTemplate.from_messages([
#     (
#         "system",
#         """
# You are a helpful AI assistant.

# You MUST use:
# 1. Conversation history
# 2. Retrieved document context

# to answer naturally and accurately.

# If the answer is in the context, use it.
# If not, say you cannot find it in the document.
# """
#     ),
#     (
#         "human",
#         """
# Conversation History:
# {history}

# Context:
# {context}

# Question:
# {question}
# """
#     )
# ])


# # RAG PIPELINE

# rag_chain = (
#     RunnableParallel(
#         context=retriever | RunnableLambda(format_docs),
#         question=RunnablePassthrough(),
#         history=RunnableLambda(lambda x: get_chat_history())
#     )
#     | prompt
#     | llm
#     | StrOutputParser()
# )


# # UI

# #st.title(" Conversational RAG Assistant")

# # display chat history
# for msg in st.session_state.chat_history:
#     with st.chat_message(msg["role"]):
#         st.write(msg["content"])

# # user input
# user_question = st.chat_input("Ask a question about your documents...")

# if user_question:

#     # store user message
#     st.session_state.chat_history.append({
#         "role": "user",
#         "content": user_question
#     })

#     # show user message
#     with st.chat_message("user"):
#         st.write(user_question)

#     # get response
#     response = rag_chain.invoke(user_question)

#     # store assistant message
#     st.session_state.chat_history.append({
#         "role": "assistant",
#         "content": response
#     })

#     # show response
#     with st.chat_message("assistant"):
#         st.write(response)

# # ENTRY FUNCTION
# def run_rag(question: str):
#     return rag_chain.invoke(question)








from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough
)
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser

from langchain_groq import ChatGroq



# EMBEDDINGS
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ----------------------------
# VECTOR DB
# ----------------------------
vectorstore = Chroma(
    persist_directory="chroma_db1",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ----------------------------
# LLM
# ----------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# ----------------------------
# PROMPT
# ----------------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a helpful AI assistant.

Use:
- Conversation history
- Retrieved context

to answer accurately.

If answer is not in context, say you cannot find it.
"""
    ),
    (
        "human",
        """
Conversation History:
{history}

Context:
{context}

Question:
{question}
"""
    )
])

# ----------------------------
# FORMAT DOCS
# ----------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ----------------------------
# RAG CHAIN
# ----------------------------
rag_chain = (
    RunnableParallel(
        question=RunnablePassthrough(),
        history=RunnableLambda(lambda x: x.get("history", ""))
    )
    | RunnableLambda(lambda x: {
        "question": x["question"],
        "context": retriever.get_relevant_documents(x["question"]),
        "history": x["history"]
    })
    | RunnableLambda(lambda x: {
        "question": x["question"],
        "context": "\n\n".join(doc.page_content for doc in x["context"]),
        "history": x["history"]
    })
    | prompt
    | llm
    | StrOutputParser()
)

# ----------------------------
# MAIN FUNCTION (NO STREAMLIT HERE)
# ----------------------------
def run_rag(question: str, history: str = ""):

    docs = retriever.invoke(question)
    context = "\n\n".join(docs.page_content for docs in docs)

    response = llm.invoke(
        prompt.format(
            question=question,
            context=context,
            history=history
        )
    )

    return response.content