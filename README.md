# 🧠 Personalized CV RAG Assistant  

> **AI Engineering Portfolio Project**

A **Personalized Retrieval-Augmented Generation (RAG)** system that answers questions using **my CV and curated academic documents** as the primary knowledge source.  
This project demonstrates how to build **grounded, explainable AI systems** using modern LLM tooling and vector databases.

---

## 🚀 Project Overview

This system allows users to ask questions about **my professional background, skills, and study materials**, with answers generated **only when supported by retrieved documents**.

Unlike generic chatbots, this assistant:
- Uses **my CV as its core knowledge base**
- Retrieves **relevant document chunks** when necessary
- Avoids retrieval for simple or general questions
- Produces **context-aware, verifiable answers**

This repository serves as a **portfolio-grade demonstration** of real-world RAG engineering.

---

## 🎯 Why This Project?

I built this project to demonstrate:
- Practical **Retrieval-Augmented Generation**
- Intelligent **agent decision-making**
- Clean separation of **indexing, querying, API, and UI**
- Production-minded AI system design

It reflects how LLMs should be used responsibly: **grounded in data, not hallucinations**.

---

## 🏗️ System Architecture

```
                    ┌────────────────────┐
                    │        User        │
                    │ (Streamlit / API)  │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │   Query Agent      │
                    │ (Decision Logic)   │
                    └─────────┬──────────┘
               ┌──────────────┴──────────────┐
               │                             │
               ▼                             ▼
     ┌──────────────────┐                    ▼
       Direct LLM       │          ┌────────────────────┐
     │ Response         │          │  Chroma Vector DB  │
     └──────────────────┘          │  (Embeddings)      │                  
                                   └─────────┬──────────┘
                                             ▼
                                   ┌────────────────────┐
                                   │ CV in PDF          │
                                   │ (Chunked Doc)      │
                                   └────────────────────┘
                                           
                                   
```

---

## 📁 Project Structure

```
Week_20/Day_5/
│
├── app.py                  # FastAPI backend
├── rag_index.py            # Document ingestion & embedding
├── rag_query.py            # Retrieval + agent logic
├── streamlit_app.py        # Portfolio UI
├── requirements.txt        # Dependencies
│
├── chroma_db1/             # Vector database directory
│   ├── chroma.sqlite3
│   └── ... (ChromaDB files)
│
└── README.md
```

---

## 🧩 Key Components

### 🔹 rag_index.py
- Loads CV and PDF documents
- Cleans and chunks text
- Generates embeddings
- Stores vectors in **ChromaDB**

### 🔹 rag_query.py
- Performs retrieval
- Returns diverse, relevant context
- Preserves metadata for transparency

### 🔹 app.py
- FastAPI backend
- `/query` endpoint
- Designed for extension and deployment

### 🔹 streamlit_app.py
- Interactive chat interface
- Ideal for demos and portfolio review

---

## 🛠️ Tech Stack

| Tool | Purpose |
|----|----|
| Python | Core language |
| LangChain | Agent orchestration |
| ChromaDB | Vector storage |
| OpenAI / LLM API | Language reasoning |
| FastAPI | Backend API |
| Streamlit | Frontend UI |
| PyPDFLoader | Document parsing |

---

## ▶️ Running the Project

```bash
pip install -r requirements.txt
python rag_index.py
uvicorn app:app --reload
streamlit run streamlit_app.py
```

---

## 🧪 Example Queries

- “Summarize my background in data science.”



The agent decides **when retrieval is required**.

---

## 📌 Design Highlights

- ✅ Intelligent retrieval decisions
- 📏 Optimized chunk sizing
- 🧠 Clean modular design

---

## 🔮 Future Enhancements

- Source citation highlighting
- Multi-document synthesis
- CV auto-refresh pipeline
- Authentication & user profiles
- Cloud deployment (AWS / Azure)

---

## 👤 About This Project

This project demonstrates how **personal data and LLMs** can be combined to build **trustworthy, explainable AI systems**.  
It reflects my growth as an **AI Engineer focused on real-world applications**.

---

⭐ *If you find this project useful or insightful, feel free to star the repository.*
