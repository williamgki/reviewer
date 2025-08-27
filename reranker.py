from typing import List, Tuple

try:
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover - fallback when not available
    CrossEncoder = None  # type: ignore


class CrossEncoderReranker:
    """Simple cross-encoder based reranker."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name) if CrossEncoder else None

    def rerank(self, query: str, docs: List[Tuple[str, str]]) -> List[Tuple[str, float]]:
        if not self.model:
            # Fallback: keep original order with zero scores
            return [(doc_id, 0.0) for doc_id, _ in docs]
        pairs = [[query, text] for _, text in docs]
        scores = self.model.predict(pairs)
        return [(doc_id, float(score)) for (doc_id, _), score in zip(docs, scores)]
