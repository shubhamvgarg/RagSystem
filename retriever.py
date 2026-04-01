import os
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaEmbeddings
from sentence_transformers import CrossEncoder

INDEX_DIR = "faiss_indexes"

embeddings = OllamaEmbeddings(model="nomic-embed-text")

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def load_all_vector_dbs():

    folders = [
        os.path.join(INDEX_DIR, f)
        for f in os.listdir(INDEX_DIR)
        if os.path.isdir(os.path.join(INDEX_DIR, f))
    ]

    merged = None
    docs = []

    for folder in folders:

        db = FAISS.load_local(
            folder,
            embeddings,
            allow_dangerous_deserialization=True
        )

        docs.extend(db.docstore._dict.values())

        if merged is None:
            merged = db
        else:
            merged.merge_from(db)

    return merged, docs


def hybrid_retrieval(query):

    vectordb, docs = load_all_vector_dbs()

    vector_docs = vectordb.similarity_search(query, k=20)

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 20

    keyword_docs = bm25.invoke(query)

    seen = set()
    unique_docs = []

    for doc in vector_docs + keyword_docs:
        content = doc.page_content.strip()
        if content not in seen:
            seen.add(content)
            unique_docs.append(doc)
    return unique_docs



def rerank(query, docs, top_k=12):

    pairs = [[query, doc.page_content] for doc in docs]

    scores = reranker.predict(pairs)

    scored_docs = list(zip(scores, docs))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]