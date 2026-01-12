import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Load API key (local + cloud safe)
OPENAI_API_KEY = (
    st.secrets.get("OPENAI_API_KEY")
    or os.getenv("OPENAI_API_KEY")
)

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found")

# REQUIRED for new OpenAI SDK
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Vector store
vectorstore = Chroma(
    persist_directory="chroma_db1",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant answering questions using personal documents. Use only the provided context."),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    RunnableParallel(
        context=retriever | format_docs,
        question=RunnablePassthrough()
    )
    | prompt
    | llm
    | StrOutputParser()
)

def run_rag(question: str) -> str:
    return rag_chain.invoke(question)
