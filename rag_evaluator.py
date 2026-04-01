from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)

from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

import numpy as np



llm = ChatOllama(model="llama3.2", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

ragas_llm = LangchainLLMWrapper(llm)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)


def safe_similarity(a, b):
    try:
        vecs = embeddings.embed_documents([a, b])
        v1, v2 = np.array(vecs[0]), np.array(vecs[1])

        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom == 0:
            return 0.0

        return float(np.dot(v1, v2) / denom)
    except:
        return 0.0
    
def fallback_scores(question, answer, contexts, ground_truth):
    """
    Ensures NO metric is ever None/NaN
    """

    context_text = " ".join(contexts) if contexts else ""

    scores = {}

    scores["answer_relevancy"] = safe_similarity(question, answer)

    scores["faithfulness"] = safe_similarity(answer, context_text)

    scores["context_precision"] = safe_similarity(context_text, answer)

    if ground_truth:
        scores["context_recall"] = safe_similarity(context_text, ground_truth)
    else:
        scores["context_recall"] = 0.0

    if ground_truth:
        scores["answer_correctness"] = safe_similarity(answer, ground_truth)
    else:
        scores["answer_correctness"] = 0.0

    return scores



def evaluate_rag(question, answer, contexts, ground_truth=None):

    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts if contexts else [""]],
    }

    if ground_truth:
        data["reference"] = [ground_truth]

    dataset = Dataset.from_dict(data)

    try:
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                answer_correctness
            ],
            llm=ragas_llm,
            embeddings=ragas_embeddings
        )

        scores = result.to_pandas().to_dict(orient="records")[0]

    except Exception as e:
        print(f"[RAGAS ERROR] {e}")
        scores = {}


    final_scores = {}
    fallback = fallback_scores(question, answer, contexts, ground_truth)

    for key in [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_correctness"
    ]:
        val = scores.get(key)

        if val is None or not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
            final_scores[key] = fallback[key]
        else:
            final_scores[key] = float(val)

    return final_scores