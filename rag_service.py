import json

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from rag_evaluator import evaluate_rag
from retriever import hybrid_retrieval, rerank
import math
import json


llm = ChatOllama(
    model="llama3.2",
    temperature=0.2
)


# memory settings
CHAT_MEMORY_SIZE = 5
chat_history: list[tuple[str, str]] = []

prompt_template = """
You are a factual QA assistant.

STRICT RULES:
1. Answer ONLY from the provided context and provided reference
2. Do NOT infer answers beyond the reference or context
3. If multiple years or authors appear, choose ONLY the one explicitly stated as the main claim
4. If unsure, say: "Answer not found in context"
5. Do NOT guess

Conversation history:
{history}

Context:
{context}

Question:
{question}

Answer (with source citation):
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["history", "context", "question"]
)


def format_docs(docs):

    formatted = []

    for doc in docs:

        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")

        formatted.append(
            f"(Source: {source} Page: {page})\n{doc.page_content}"
        )
    return "\n\n".join(formatted)


def format_history(history_list: list[tuple[str, str]]) -> str:
    """Convert a list of (question, answer) tuples into a string."""
    lines = []
    for q, a in history_list:
        lines.append(f"User: {q}")
        lines.append(f"Assistant: {a}")
    return "\n".join(lines)


def ask_question(query):

    history = format_history(chat_history)
    print(f"[memory] Current chat history:\n{history}")

    retrieved = hybrid_retrieval(query)

    reranked = rerank(query, retrieved)

    context = format_docs(reranked)

    prompt = PROMPT.format(
        history=history,
        context=context,
        question=query
    )

    response = llm.invoke(prompt)
    answer = response.content

    # Prepare context list for RAGAS
    context_list = [doc.page_content for doc in reranked]
    with open("data.json", "r") as f:
        data = json.load(f)

    def get_ground_truth(query):
        for item in data:
            if item["question"].strip().lower() == query.strip().lower():
                print(f"[ground_truth] Found ground truth for query: {item['ground_truth']}")
                return item["ground_truth"]
        return ""
    truth = get_ground_truth(query)

    scores = evaluate_rag(
        question=query,
        answer=answer,
        contexts=context_list,
        ground_truth=truth

    )
   
    clean_scores = scores
    chat_history.append((query, answer))
    if len(chat_history) > CHAT_MEMORY_SIZE:
        del chat_history[: len(chat_history) - CHAT_MEMORY_SIZE]
    print(f"[memory] Updated chat history: {chat_history}")
    # returning answer and scores for evaluation
    return {
        "answer": answer,
        "ragas_scores": clean_scores
    }