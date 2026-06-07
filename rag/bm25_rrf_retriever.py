"""
BM25 + dense + RRF retriever — no cross-encoder reranking.
Sits between NaiveRetriever (dense-only) and HybridRetriever (adds reranking).
Used only in evaluation to measure how much the cross-encoder actually helps.
"""
from rag.retriever import get_retriever, _RRF_K


class BM25RRFRetriever:
    """BM25 + dense fused with RRF. No cross-encoder step."""

    def retrieve(self, query: str, top_k: int = 3):
        r = get_retriever()

        bm25_docs  = r._bm25_search(query, top_k * 3)
        dense_docs = r._dense_search(query, top_k * 3)

        merged: dict = {}
        for rank, doc in enumerate(bm25_docs, 1):
            doc.hybrid_score = 1.0 / (_RRF_K + rank)
            merged[doc.doc_id] = doc
        for rank, doc in enumerate(dense_docs, 1):
            rrf = 1.0 / (_RRF_K + rank)
            if doc.doc_id in merged:
                merged[doc.doc_id].hybrid_score += rrf
            else:
                doc.hybrid_score = rrf
                merged[doc.doc_id] = doc

        return sorted(merged.values(), key=lambda d: d.hybrid_score, reverse=True)[:top_k]
