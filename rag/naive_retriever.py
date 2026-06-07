"""
Naive Baseline Retriever — dense-only cosine similarity (no BM25, no reranking).
Used only for evaluation comparison against HybridRetriever.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import settings
from rag.embeddings import LocalEmbeddings
from rag.indexer import COLLECTION_ALL

logger = logging.getLogger(__name__)

_instance = None


def get_naive_retriever() -> "NaiveRetriever":
    global _instance
    if _instance is None:
        _instance = NaiveRetriever()
    return _instance


@dataclass
class NaiveDoc:
    doc_id: str
    content: str
    source: str
    score: float  # cosine similarity (0–1); higher = more similar


class NaiveRetriever:
    """
    Baseline retriever: embed query → ChromaDB cosine search → return top-k.
    No BM25, no RRF fusion, no cross-encoder reranking.
    """

    def __init__(self):
        self.embedder = LocalEmbeddings(model_name=settings.embedding_model)
        client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        try:
            self.collection = client.get_collection(COLLECTION_ALL)
            logger.info("NaiveRetriever: %d docs loaded.", self.collection.count())
        except Exception as e:
            logger.error("NaiveRetriever: collection not found — %s", e)
            self.collection = None

    def retrieve(self, query: str, top_k: int = 3) -> list[NaiveDoc]:
        if not self.collection or not query.strip():
            return []

        vector = self.embedder.embed_query(query)
        results = self.collection.query(
            query_embeddings=[vector],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        docs = []
        for text, meta, distance, doc_id in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ):
            docs.append(NaiveDoc(
                doc_id=doc_id,
                content=text,
                source=(meta or {}).get("source", "unknown"),
                score=max(0.0, 1.0 - distance),
            ))
        return docs
