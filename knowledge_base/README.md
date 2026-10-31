# Curated MND knowledge base

`mnd_knowledge.json` stores reviewed summaries and source links for MND/ALS,
genetics, gene testing, and pre-test and post-test counselling. Source links
and review dates are stored with the entries so the collection can be audited.

## Retrieval modes

The runtime reads this file locally. Select retrieval with
`MND_RETRIEVAL_MODE`:

- `lexical`: weighted word matching; no embedding model or download required.
- `hybrid`: lexical matching plus local semantic embeddings, fused with
  reciprocal-rank scoring.
- `semantic`: semantic retrieval without lexical contributions.

The Python retrieval code defaults to `hybrid`. The README quick start exports
`MND_RETRIEVAL_MODE=lexical` so hosted inference can run without an embedding
download. Hybrid and semantic retrieval do not call a separate language model;
embeddings run locally on the CPU.

## Prepare hybrid retrieval

```bash
python -m pip install fastembed==0.9.1
MND_RETRIEVAL_MODE=hybrid python prepare_retrieval.py --download
```

Set `MND_RETRIEVAL_MODE=hybrid` in `.env` and restart the app. The default
embedding model is `BAAI/bge-small-en-v1.5`; files are cached under
`.runtime/embeddings/` and are excluded from Git. `MND_EMBEDDING_MODEL` and
`MND_EMBEDDING_CACHE` select a different supported model or cache directory.
Subsequent runs require local embedding files by default. Hybrid mode reports a
setup error when those files are unavailable; it does not silently fall back to
lexical retrieval.

The summaries are research reference material, not clinical protocols. They do
not replace the linked documents or the person's clinical genetics or neurology
service.
