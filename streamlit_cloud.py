import streamlit as st
from rag_query_cloud import run_rag

# Page config
st.set_page_config(page_title="Personal RAG Assistant", layout="centered")

# Title & description
st.title("My Personal RAG Assistant")
st.write("Ask questions about my CV and personal documents.")

st.markdown(
    """
    **Assistant Persona:**  
    I answer questions strictly based on Okemakinde Sherif's personal documents  
    (CV, experience, education, projects).
    """
)

# Input box
question = st.text_input("Ask a question")

# Button
if st.button("Ask"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                answer = run_rag(question)
                st.subheader("Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")
