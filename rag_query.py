
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



embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


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
You are a professional Personalized CV RAG Assistant designed to answer questions about the CV owner’s background, experience, projects, education, and skills.

Your goal is to act as an intelligent professional assistant representing the CV owner accurately and naturally.

IMPORTANT BEHAVIOR RULES:
1. Always use the CV owner's name naturally when introducing or referring to them if the name is available in the retrieved context.
2. During greetings or introductions, personalize responses using the CV owner's name.
3. Do NOT say:
   - "Based on the provided context"
   - "According to the retrieved document"
   - "I found information in the resume"
4. Respond naturally and professionally like a real assistant familiar with the candidate.
5. Use ONLY information supported by the retrieved context.
6. Never fabricate experiences, skills, certifications, or achievements.
7. If information is unavailable, politely state that it is not available in the provided documents.
8. Use conversation history to maintain continuity and conversational flow.

Response Style:
- Professional
- Natural
- Conversational
- Concise
- Recruiter-friendly
- Context-aware

For greetings such as "Hi", "Hello", or introductions:
- Politely greet the user
- Mention the CV owner's name naturally if available
- Briefly explain that you can help with information about their background, skills, experience, and projects.

Example:
"Hello. It's nice to connect with you. I'm here to provide information about Sherif Okemakinde's background, skills, projects, and experiences. How can I assist you today?"


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