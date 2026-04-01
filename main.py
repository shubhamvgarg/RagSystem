import streamlit as st
import os
from vector_store import build_or_load_vectordb
from rag_service import ask_question

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.set_page_config(page_title="Local RAG App", layout="wide")

st.title("📄 Local RAG Application")

st.header("Upload PDFs")

uploaded_files = st.file_uploader(
    "Upload up to 5 PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) > 5:
        st.error("Maximum 5 PDFs allowed")
    else:
        saved_files = []

        if st.button("Process PDFs"):
            try:
                for file in uploaded_files:
                    safe_name = file.name.replace(" ", "_")
                    path = os.path.join(UPLOAD_FOLDER, safe_name)

                    with open(path, "wb") as buffer:
                        buffer.write(file.getbuffer())

                    saved_files.append(path)

                for pdf in saved_files:
                    build_or_load_vectordb(pdf)

                st.success("PDFs processed successfully")
                st.write(saved_files)

            except Exception as e:
                st.error(f"Error: {str(e)}")



st.header("💬 Chat with your PDFs")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
prompt = st.chat_input("Ask something about your PDFs...")

if prompt:

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        print(prompt)
        result = ask_question(prompt)
        answer = result["answer"]

        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.markdown(answer)

        with st.expander("📊 RAGAS Scores"):
            st.json(result["ragas_scores"])

    except Exception as e:
        st.error(f"Error: {str(e)}")