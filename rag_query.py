
import warnings
warnings.filterwarnings("ignore")

from langchain_chroma import Chroma
#from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

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


# VECTOR DB

vectorstore = Chroma(
    persist_directory="chroma_db1",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)



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



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


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