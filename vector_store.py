import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

INDEX_DIR = "faiss_indexes"

embeddings = OllamaEmbeddings(model="nomic-embed-text")


def index_path_for(pdf_path: str):

    base = os.path.splitext(os.path.basename(pdf_path))[0]
    return os.path.join(INDEX_DIR, base)


def convert_pdf_to_chunks(pdf_path: str):

    loader = PyMuPDFLoader(pdf_path)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents)

    for doc in docs:
        doc.metadata["source"] = pdf_path

    return docs


def build_or_load_vectordb(pdf_path: str):

    os.makedirs(INDEX_DIR, exist_ok=True)

    index_dir = index_path_for(pdf_path)

    if os.path.exists(index_dir):

        return FAISS.load_local(
            index_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )

    docs = convert_pdf_to_chunks(pdf_path)

    vectordb = FAISS.from_documents(docs, embeddings)

    vectordb.save_local(index_dir)

    return vectordb