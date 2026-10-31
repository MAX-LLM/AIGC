import json
import os
import re
from functools import lru_cache
from pathlib import Path
from threading import RLock

import requests

from hybrid_retrieval import HybridKnowledgeRetriever, RetrievalError


KNOWLEDGE_BASE_PATH = (
    Path(__file__).resolve().parent / "knowledge_base" / "mnd_knowledge.json"
)

_STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "be",
    "can",
    "do",
    "for",
    "from",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "this",
    "to",
    "what",
    "with",
    "would",
}


MND_GENE_FACTS = {
    "C9ORF72": {
        "summary": (
            "C9orf72 repeat expansions are one of the commonest known genetic causes "
            "of familial ALS/MND and can also be associated with frontotemporal dementia. "
            "Repeat-expansion testing is different from ordinary sequencing."
        ),
        "ncbi_gene_id": "203228",
    },
    "SOD1": {
        "summary": (
            "SOD1 variants are an established genetic cause of ALS/MND. Many disease-causing "
            "SOD1 variants are inherited in an autosomal dominant pattern, but exact risk "
            "and implications depend on the specific variant and family context."
        ),
        "ncbi_gene_id": "6647",
    },
    "TARDBP": {
        "summary": (
            "TARDBP variants are a recognised but less common genetic cause of ALS/MND. "
            "Interpretation depends on the exact variant and its clinical classification."
        ),
        "ncbi_gene_id": "23435",
    },
    "FUS": {
        "summary": (
            "FUS variants are a recognised genetic cause of ALS/MND and can sometimes be "
            "associated with younger onset. Interpretation requires the exact variant."
        ),
        "ncbi_gene_id": "2521",
    },
    "TBK1": {
        "summary": (
            "TBK1 variants can be associated with ALS/MND and frontotemporal dementia. "
            "The meaning of a result depends on the exact variant and family history."
        ),
        "ncbi_gene_id": "29110",
    },
    "OPTN": {
        "summary": (
            "OPTN variants have been linked with ALS/MND in some families. Interpretation "
            "depends on the specific variant and inheritance pattern."
        ),
        "ncbi_gene_id": "10133",
    },
    "VCP": {
        "summary": (
            "VCP variants can be associated with a spectrum that may include ALS/MND, "
            "frontotemporal dementia, muscle disease, or bone disease in some families."
        ),
        "ncbi_gene_id": "7415",
    },
}


def normalize_gene_symbol(symbol: str) -> str:
    return symbol.strip().replace("-", "").upper()


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in _STOP_WORDS
    }


@lru_cache(maxsize=1)
def load_mnd_knowledge() -> list[dict]:
    with KNOWLEDGE_BASE_PATH.open(encoding="utf-8") as source:
        payload = json.load(source)
    return list(payload.get("entries", []))


def _lexical_scores(query: str, entries: list[dict]) -> dict[str, float]:
    """Preserve the weighted lexical ranking used in the reported evaluations."""
    query_tokens = _tokens(query)
    scores = {}
    for entry in entries:
        topic_text = " ".join(entry.get("topics", []))
        keyword_text = " ".join(entry.get("keywords", []))
        title_tokens = _tokens(entry.get("title", ""))
        topic_tokens = _tokens(topic_text)
        keyword_tokens = _tokens(keyword_text)
        summary_tokens = _tokens(entry.get("summary", ""))
        score = (
            4 * len(query_tokens & title_tokens)
            + 3 * len(query_tokens & topic_tokens)
            + 2 * len(query_tokens & keyword_tokens)
            + len(query_tokens & summary_tokens)
        )
        if score:
            scores[entry["id"]] = float(score)
    return scores


def _retrieval_settings() -> dict:
    mode = os.environ.get("MND_RETRIEVAL_MODE", "hybrid").strip().lower()
    if mode not in {"hybrid", "semantic", "lexical"}:
        raise RetrievalError("MND_RETRIEVAL_MODE must be hybrid, semantic or lexical.")
    local_only = os.environ.get("MND_EMBEDDING_LOCAL_ONLY", "1").strip().lower()
    if local_only not in {"1", "true", "yes", "0", "false", "no"}:
        raise RetrievalError("MND_EMBEDDING_LOCAL_ONLY must be 1 or 0.")
    try:
        minimum = float(os.environ.get("MND_RETRIEVAL_MIN_SIMILARITY", "0.25"))
        rank_constant = int(os.environ.get("MND_RETRIEVAL_RRF_K", "60"))
        threads = int(os.environ.get("MND_EMBEDDING_THREADS", "2"))
    except ValueError as exc:
        raise RetrievalError("Invalid numeric retrieval setting.") from exc
    if not 0 <= minimum <= 1 or rank_constant <= 0 or threads <= 0:
        raise RetrievalError("Invalid retrieval threshold, rank constant or thread count.")
    return {
        "mode": mode,
        "model": os.environ.get("MND_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        "cache": str(Path(os.environ.get(
            "MND_EMBEDDING_CACHE", str(Path(__file__).parent / ".runtime" / "embeddings")
        )).expanduser().resolve()),
        "local_only": local_only in {"1", "true", "yes"},
        "minimum": minimum,
        "rank_constant": rank_constant,
        "threads": threads,
    }


_RETRIEVER_LOCK = RLock()


@lru_cache(maxsize=4)
def _make_hybrid_retriever(
    manifest: str, model: str, cache: str, local_only: bool,
    minimum: float, rank_constant: int, threads: int,
) -> HybridKnowledgeRetriever:
    try:
        from fastembed import TextEmbedding

        embedder = TextEmbedding(
            model_name=model, cache_dir=cache, threads=threads,
            providers=["CPUExecutionProvider"], cuda=False,
            local_files_only=local_only,
        )
        return HybridKnowledgeRetriever(
            json.loads(manifest), embedder,
            min_similarity=minimum, rrf_k=rank_constant,
        )
    except RetrievalError:
        raise
    except Exception as exc:
        raise RetrievalError(
            "Semantic retrieval could not start. Run 'python -m pip install fastembed==0.9.1' "
            "and run 'python prepare_retrieval.py --download' before starting the agent. "
            "Hybrid mode requires the local embedding model."
        ) from exc


def _get_hybrid_retriever(entries: list[dict], settings: dict) -> HybridKnowledgeRetriever:
    manifest = json.dumps(entries, sort_keys=True, ensure_ascii=False)
    with _RETRIEVER_LOCK:
        return _make_hybrid_retriever(
            manifest, settings["model"], settings["cache"], settings["local_only"],
            settings["minimum"], settings["rank_constant"], settings["threads"],
        )


def warm_knowledge_retrieval() -> dict:
    """Prepare the local corpus before accepting conversations."""
    settings = _retrieval_settings()
    entries = load_mnd_knowledge()
    if settings["mode"] != "lexical":
        _get_hybrid_retriever(entries, settings)
    return {
        "mode": settings["mode"], "entries": len(entries),
        "embedding_model": settings["model"] if settings["mode"] != "lexical" else None,
    }


def retrieve_mnd_knowledge(query: str, limit: int = 3) -> list[dict]:
    """Rank local references with hybrid retrieval, or an explicit comparison mode."""
    if limit <= 0 or not query.strip():
        return []
    settings = _retrieval_settings()
    entries = load_mnd_knowledge()
    scores = _lexical_scores(query, entries)
    if settings["mode"] == "lexical":
        ranked = sorted(
            (entry for entry in entries if entry["id"] in scores),
            key=lambda entry: (-scores[entry["id"]], entry["id"]),
        )
        return ranked[:limit]

    retriever = _get_hybrid_retriever(entries, settings)
    return retriever.rank(query, scores if settings["mode"] == "hybrid" else {}, limit)


def format_mnd_knowledge(query: str, limit: int = 3) -> tuple[str, list[dict]]:
    references = retrieve_mnd_knowledge(query, limit=limit)
    blocks = []
    for entry in references:
        blocks.append(
            f"Source: {entry['title']} ({entry['publisher']}, {entry['year']})\n"
            f"URL: {entry['url']}\n"
            f"Reviewed summary: {entry['summary']}"
        )
    return "\n\n".join(blocks), references


def lookup_gene_info(symbol: str) -> str:
    """
    Look up basic information about a human gene symbol.

    The tool first uses a local MND/ALS-relevant knowledge base so it still works
    when the deployment environment has no internet. It then adds a short MyGene
    reference summary when outbound network access is available.
    """
    normalized = normalize_gene_symbol(symbol)
    local_fact = MND_GENE_FACTS.get(normalized)

    parts = []
    if local_fact:
        parts.append(
            f"Curated MND genetics note for {normalized}: {local_fact['summary']}\n"
            "Sources:\n"
            "- ALS genetic testing consensus guideline: "
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC10646996/\n"
            f"- NCBI Gene: https://www.ncbi.nlm.nih.gov/gene/{local_fact['ncbi_gene_id']}"
        )

    try:
        url = "https://mygene.info/v3/query"
        params = {
            "q": f"symbol:{normalized}",
            "species": "human",
            "fields": "symbol,name,summary",
        }
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        hits = data.get("hits", [])
        if hits:
            hit = hits[0]
            sym = hit.get("symbol", normalized)
            name = hit.get("name", "Unknown")
            summary = hit.get("summary", "No summary available.")
            parts.append(
                "Reference lookup:\n"
                f"Gene: {sym}\n"
                f"Name: {name}\n"
                f"Summary: {summary}"
            )
    except Exception as exc:
        if not parts:
            return f"No local information found for '{symbol}', and online gene lookup failed: {exc}"

    if not parts:
        return f"No detailed reference found for gene symbol '{symbol}' in this lookup."

    return "\n\n".join(parts)
