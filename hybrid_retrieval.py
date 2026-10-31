"""Local semantic and lexical retrieval for the curated knowledge entries.

The encoder is injected so ranking can be tested without a model download.
FastEmbed's TextEmbedding implements the passage_embed/query_embed interface.
"""

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
import math
import operator
from typing import Any, Protocol


class RetrievalError(RuntimeError):
    """Semantic retrieval could not be completed as configured."""


class EmbeddingModel(Protocol):
    def passage_embed(self, texts: Sequence[str]) -> Iterable[Sequence[float]]: ...

    def query_embed(self, text: str) -> Iterable[Sequence[float]]: ...


def _normalise(vector: Sequence[float]) -> tuple[float, ...]:
    """Copy and normalise a vector, rejecting malformed encoder output."""
    if isinstance(vector, (str, bytes)) or getattr(vector, "ndim", 1) != 1:
        raise RetrievalError("The embedding model returned a non-numeric vector.")
    try:
        values = tuple(float(value) for value in vector)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RetrievalError("The embedding model returned a malformed vector.") from exc
    if not values or not all(math.isfinite(value) for value in values):
        raise RetrievalError("Embeddings must contain finite numeric values.")
    scale = max(abs(value) for value in values)
    if scale == 0:
        raise RetrievalError("The embedding model returned a zero vector.")
    # Scaling first also handles large finite values without overflowing the norm.
    scaled = tuple(value / scale for value in values)
    norm = math.sqrt(math.fsum(value * value for value in scaled))
    return tuple(value / norm for value in scaled)


def _passage_text(entry: Mapping[str, Any]) -> str:
    return "\n".join(
        (
            f"Title: {entry.get('title', '')}",
            f"Topics: {'; '.join(entry.get('topics', []))}",
            f"Keywords: {'; '.join(entry.get('keywords', []))}",
            f"Summary: {entry.get('summary', '')}",
        )
    )


class HybridKnowledgeRetriever:
    """Fuse positive lexical matches and cosine-ranked semantic matches.

    Each ranking contributes ``1 / (rrf_k + rank)``, with one-based ranks.
    Entries below min_similarity receive no semantic contribution. Positive
    lexical matches remain eligible even when their semantic score is lower.
    """

    def __init__(
        self,
        entries: Sequence[Mapping[str, Any]],
        embedder: EmbeddingModel,
        *,
        min_similarity: float = 0.25,
        rrf_k: float = 60,
    ) -> None:
        if not math.isfinite(min_similarity) or not -1 <= min_similarity <= 1:
            raise ValueError("min_similarity must be finite and between -1 and 1.")
        if not math.isfinite(rrf_k) or rrf_k < 0:
            raise ValueError("rrf_k must be finite and non-negative.")
        self._embedder = embedder
        self._min_similarity = min_similarity
        self._rrf_k = rrf_k
        self._entries: dict[str, dict[str, Any]] = {}
        for entry in entries:
            entry_id = entry.get("id")
            if not isinstance(entry_id, str) or not entry_id:
                raise ValueError("Every knowledge entry requires a non-empty string ID.")
            if entry_id in self._entries:
                raise ValueError(f"Duplicate knowledge entry ID: {entry_id}")
            self._entries[entry_id] = deepcopy(dict(entry))
        self._vectors: dict[str, tuple[float, ...]] = {}
        self._dimension = 0
        if not self._entries:
            return
        try:
            texts = [_passage_text(entry) for entry in self._entries.values()]
            vectors = list(embedder.passage_embed(texts))
        except Exception as exc:
            raise RetrievalError("Could not encode the curated knowledge entries.") from exc
        if len(vectors) != len(self._entries):
            raise RetrievalError("The embedding model returned an incorrect passage count.")
        for entry_id, vector in zip(self._entries, vectors):
            normalised = _normalise(vector)
            if not self._dimension:
                self._dimension = len(normalised)
            if len(normalised) != self._dimension:
                raise RetrievalError("Passage embeddings have inconsistent dimensions.")
            self._vectors[entry_id] = normalised

    def rank(
        self,
        query: str,
        lexical_scores: Mapping[str, float],
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Return up to limit entries; semantic failures always raise RetrievalError."""
        limit = operator.index(limit)
        if limit <= 0 or not query.strip() or not self._entries:
            return []
        try:
            outputs = list(self._embedder.query_embed(query))
        except Exception as exc:
            raise RetrievalError("Could not encode the retrieval query.") from exc
        if len(outputs) != 1:
            raise RetrievalError("The embedding model must return one query vector.")
        query_vector = _normalise(outputs[0])
        if len(query_vector) != self._dimension:
            raise RetrievalError("Query and passage embedding dimensions differ.")

        semantic: dict[str, float] = {}
        lexical: dict[str, float] = {}
        for entry_id, vector in self._vectors.items():
            cosine = math.fsum(a * b for a, b in zip(query_vector, vector))
            cosine = max(-1.0, min(1.0, cosine))
            if cosine >= self._min_similarity:
                semantic[entry_id] = cosine
            try:
                score = float(lexical_scores.get(entry_id, 0))
            except (TypeError, ValueError, OverflowError) as exc:
                raise RetrievalError("Lexical retrieval returned a non-numeric score.") from exc
            if not math.isfinite(score):
                raise RetrievalError("Lexical scores must be finite.")
            if score > 0:
                lexical[entry_id] = score

        fused: dict[str, float] = {}
        for scores in (lexical, semantic):
            ranking = sorted(scores, key=lambda entry_id: (-scores[entry_id], entry_id))
            for rank, entry_id in enumerate(ranking, start=1):
                fused[entry_id] = fused.get(entry_id, 0.0) + 1 / (self._rrf_k + rank)
        ranking = sorted(fused, key=lambda entry_id: (-fused[entry_id], entry_id))
        return [deepcopy(self._entries[entry_id]) for entry_id in ranking[:limit]]
