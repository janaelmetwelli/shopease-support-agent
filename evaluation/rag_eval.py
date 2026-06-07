"""
RAG Evaluation — Context Precision and Recall
Compares three retrievers to isolate each pipeline stage:
  1. Naive      — dense cosine similarity only
  2. BM25 + RRF — BM25 + dense + RRF fusion, no cross-encoder
  3. Hybrid     — BM25 + dense + RRF + cross-encoder reranking

Metrics (all @k, where k = top_k):
  Precision = (docs from a relevant source) / k
  Recall    = (expected source types found) / (total expected source types)
  F1        = harmonic mean of precision and recall

Run standalone:
    .venv\\Scripts\\python.exe evaluation/rag_eval.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.agentic_retriever import get_agentic_retriever
from rag.bm25_rrf_retriever import BM25RRFRetriever
from rag.stepback_retriever import get_stepback_retriever


# ── Ground truth ──────────────────────────────────────────────────────────────
GROUND_TRUTH = [
    # ── Returns Policy ────────────────────────────────────────────────────────
    {
        "query": "how do I return a damaged product",
        "relevant_sources": {"returns_policy", "faq"}
    },
    {
        "query": "how long does a refund take to process",
        "relevant_sources": {"returns_policy"}
    },
    {
        "query": "what items cannot be returned",
        "relevant_sources": {"returns_policy"}
    },
    {
        "query": "return window how many days",
        "relevant_sources": {"returns_policy", "faq"}
    },

    # ── Shipping Policy ───────────────────────────────────────────────────────
    {
        "query": "what shipping options are available",
        "relevant_sources": {"shipping_policy", "faq"}
    },
    {
        "query": "how long does standard delivery take",
        "relevant_sources": {"shipping_policy"}
    },
    {
        "query": "do you ship internationally",
        "relevant_sources": {"shipping_policy", "faq"}
    },
    {
        "query": "package delivered but I never received it",
        "relevant_sources": {"faq", "shipping_policy"}
    },

    # ── Product Catalog ───────────────────────────────────────────────────────
    {
        "query": "ProBook Laptop 15 price and specifications",
        "relevant_sources": {"product_catalog"}
    },
    {
        "query": "SoundWave Headphones features and price",
        "relevant_sources": {"product_catalog"}
    },
    {
        "query": "EcoBrew Coffee Maker warranty",
        "relevant_sources": {"product_catalog", "returns_policy"}  # FIX 2
    },

    # ── Product Manuals ───────────────────────────────────────────────────────
    {
        "query": "how to use EcoBrew Coffee Maker step by step",
        "relevant_sources": {"product_manuals"}
    },
    {
        "query": "SoundWave Headphones Bluetooth not connecting",
        "relevant_sources": {"product_manuals"}
    },
    {
        "query": "InstantPot burn warning fix",
        "relevant_sources": {"product_manuals"}
    },
    {
        "query": "EcoBrew coffee tastes weak what to do",
        "relevant_sources": {"product_manuals"}
    },
    {
        "query": "InstantPot sealing ring how to clean",
        "relevant_sources": {"product_manuals"}
    },
    {
        "query": "FitTrack watch how to set up",
        "relevant_sources": {"product_manuals"}
    },

    # ── Cosmetics ─────────────────────────────────────────────────────────────
    {
        "query": "HydraGlow Vitamin C Serum ingredients and skin benefits",
        "relevant_sources": {"cosmetics_catalog"}
    },
    {
        "query": "skincare routine for oily skin",
        "relevant_sources": {"cosmetics_catalog", "recommendations"}
    },
    {
        "query": "best moisturizer for dry skin",
        "relevant_sources": {"cosmetics_catalog"}
    },

    # ── Store Info ────────────────────────────────────────────────────────────
    {
        "query": "ShopEase store locations in Cairo",
        "relevant_sources": {"store_info"}
    },
    {
        "query": "payment methods accepted at ShopEase",
        "relevant_sources": {"store_info", "faq"}
    },
    {
        "query": "customer support phone number and hours",
        "relevant_sources": {"store_info"}
    },

    # ── Recommendations ───────────────────────────────────────────────────────
    {
        "query": "current promotions and discount codes",
        "relevant_sources": {"store_info", "recommendations"}
    },
    {
        "query": "trending products and best sellers",
        "relevant_sources": {"recommendations"}
    },

    # ── FAQs ──────────────────────────────────────────────────────────────────
    {
        "query": "how do I track my order",
        "relevant_sources": {"faq"}
    },
    {
        "query": "can I cancel my order after placing it",
        "relevant_sources": {"faq"}
    },

    # ── Edge Cases ────────────────────────────────────────────────────────────
    {
        "query": "it keeps burning",  # FIX 1 — removed duplicate
        "relevant_sources": {"product_manuals"}
    },
    {
        "query": "not working properly",
        "relevant_sources": {"product_manuals", "faq"}
    },

    # ── Vague / Incomplete ────────────────────────────────────────────────────
    {
        "query": "not working",
        "relevant_sources": {"product_manuals", "faq"}
    },
    {
        "query": "broken",
        "relevant_sources": {"returns_policy", "faq"}
    },
    {
        "query": "how much",
        "relevant_sources": {"product_catalog", "cosmetics_catalog"}
    },
    {
        "query": "where",
        "relevant_sources": {"store_info"}
    },

    # ── Typos ─────────────────────────────────────────────────────────────────
    {
        "query": "retrn polcy",
        "relevant_sources": {"returns_policy", "faq"}
    },
    {
        "query": "shiping coost",
        "relevant_sources": {"shipping_policy"}
    },
    {
        "query": "cancell order",
        "relevant_sources": {"faq"}
    },
    {
        "query": "probook laptoop price",
        "relevant_sources": {"product_catalog"}
    },

    # ── Multi-Source ──────────────────────────────────────────────────────────
    {
        "query": "I want to return my laptop and buy a new one",
        "relevant_sources": {"returns_policy", "faq"}  # FIX 5
    },
    {
        "query": "shipping time and return policy",
        "relevant_sources": {"shipping_policy", "returns_policy"}
    },
    {
        "query": "store location and opening hours",
        "relevant_sources": {"store_info"}
    },
    {
        "query": "best skincare products and how to use them",
        "relevant_sources": {"cosmetics_catalog", "recommendations"}  # FIX 5
    },

    # ── Indirect / Implied ────────────────────────────────────────────────────
    {
        "query": "my package never arrived",
        "relevant_sources": {"faq", "shipping_policy"}
    },
    {
        "query": "I got the wrong item",
        "relevant_sources": {"returns_policy", "faq"}
    },
    {
        "query": "product stopped working after one week",
        "relevant_sources": {"returns_policy", "product_manuals"}
    },
    {
        "query": "I want my money back",
        "relevant_sources": {"returns_policy", "faq"}
    },
    {
        "query": "something smells bad in the box",
        "relevant_sources": {"returns_policy"}  # FIX 2
    },

    # ── Very Short ────────────────────────────────────────────────────────────
    {
        "query": "refund",
        "relevant_sources": {"returns_policy", "faq"}
    },
    {
        "query": "warranty",
        "relevant_sources": {"product_catalog", "returns_policy"}
    },
    {
        "query": "delivery",
        "relevant_sources": {"shipping_policy", "faq"}
    },
    {
        "query": "discount",
        "relevant_sources": {"store_info", "recommendations"}
    },

    # ── Arabic ────────────────────────────────────────────────────────────────
    {
        "query": "سياسة الإرجاع",
        "relevant_sources": {"returns_policy", "faq"}
    },
    {
        "query": "كيف أتتبع طلبي",
        "relevant_sources": {"faq"}
    },
    {
        "query": "فروع المتجر في القاهرة",
        "relevant_sources": {"store_info"}
    },
    {
        "query": "طرق الدفع",
        "relevant_sources": {"store_info", "faq"}
    },

    # ── Long Conversational ───────────────────────────────────────────────────
    {
        "query": "I bought a coffee maker last week and it stopped working after 3 days, what can I do",
        "relevant_sources": {"returns_policy", "faq"}  # FIX 5
    },
    {
        "query": "I ordered the wrong size headphones and want to exchange them for a smaller size",
        "relevant_sources": {"returns_policy", "faq"}
    },
    {
        "query": "what is the difference between standard and express shipping and which one is faster",
        "relevant_sources": {"shipping_policy", "faq"}
    },

    # ── Ambiguous ─────────────────────────────────────────────────────────────
    {
        "query": "can I get a replacement",
        "relevant_sources": {"returns_policy", "faq"}
    },
    {
        "query": "is the ProBook Laptop 15 in stock",  # FIX 2
        "relevant_sources": {"product_catalog"}
    },
    {
        "query": "what are my options",
        "relevant_sources": {"returns_policy", "faq", "shipping_policy"}
    },
]


# ── Metric computation ────────────────────────────────────────────────────────

def evaluate(retrieve_fn, top_k: int = 5) -> dict:
    queries = []
    latencies = []

    for item in GROUND_TRUTH:
        query    = item["query"]
        expected = item["relevant_sources"]

        # Always retrieve top-5 so we can compute @1, @3, @5
        t_start = time.perf_counter()
        docs = retrieve_fn(query, 5)  # always 5
        latency_ms = (time.perf_counter() - t_start) * 1000
        latencies.append(latency_ms)

        retrieved = [doc.source for doc in docs]

        # Precision@k = relevant in top-k / k
        def precision_at(k):
            top = retrieved[:k]
            if not top:
                return 0.0
            return sum(1 for s in top if s in expected) / k

        # Recall@k = relevant sources found in top-k / total expected
        def recall_at(k):
            top = retrieved[:k]
            if not expected:
                return 0.0
            return len(set(top) & expected) / len(expected)

        # MRR = 1 / rank of first relevant doc
        rr = 0.0
        for rank, source in enumerate(retrieved, 1):
            if source in expected:
                rr = 1.0 / rank
                break

        # F1@3 for backward compatibility
        p3 = precision_at(3)
        r3 = recall_at(3)
        f1 = 2 * p3 * r3 / (p3 + r3) if (p3 + r3) else 0.0

        queries.append({
            "query":             query,
            "expected_sources":  sorted(expected),
            "retrieved_sources": retrieved,
            "precision@1":       precision_at(1),
            "precision@3":       precision_at(3),
            "precision@5":       precision_at(5),
            "recall@1":          recall_at(1),
            "recall@3":          recall_at(3),
            "recall@5":          recall_at(5),
            "f1":                f1,
            "rr":                rr,
            "latency_ms":        round(latency_ms, 2),
        })

    n = len(queries)
    avg = lambda key: sum(q[key] for q in queries) / n
    sorted_lat = sorted(latencies)
    p50 = sorted_lat[min(int(n * 0.50), n - 1)]
    p95 = sorted_lat[min(int(n * 0.95), n - 1)]

    return {
        "avg_precision@1": avg("precision@1"),
        "avg_precision@3": avg("precision@3"),
        "avg_precision@5": avg("precision@5"),
        "avg_recall@1":    avg("recall@1"),
        "avg_recall@3":    avg("recall@3"),
        "avg_recall@5":    avg("recall@5"),
        "avg_f1":          avg("f1"),
        "mrr":             avg("rr"),
        "avg_latency_ms":  sum(latencies) / n,
        "p50_latency_ms":  p50,
        "p95_latency_ms":  p95,
        "max_latency_ms":  max(latencies),
        "queries":         queries,
    }


def evaluate_agentic(agentic, top_k: int = 5) -> dict:
    """Like evaluate(), but skipped queries are excluded from metrics rather than scored as 0."""
    queries, latencies = [], []

    for item in GROUND_TRUTH:
        query    = item["query"]
        expected = item["relevant_sources"]

        t_start = time.perf_counter()
        docs, was_skipped = agentic.retrieve_with_skip(query, 5)  # always 5
        latency_ms = (time.perf_counter() - t_start) * 1000
        latencies.append(latency_ms)

        if was_skipped:
            queries.append({"query": query, "skipped": True, "latency_ms": round(latency_ms, 2)})
            continue

        retrieved = [doc.source for doc in docs]

        def precision_at(k):
            top = retrieved[:k]
            if not top:
                return 0.0
            return sum(1 for s in top if s in expected) / k

        def recall_at(k):
            top = retrieved[:k]
            if not expected:
                return 0.0
            return len(set(top) & expected) / len(expected)

        rr = 0.0
        for rank, source in enumerate(retrieved, 1):
            if source in expected:
                rr = 1.0 / rank
                break

        p3 = precision_at(3)
        r3 = recall_at(3)
        f1 = 2 * p3 * r3 / (p3 + r3) if (p3 + r3) else 0.0

        queries.append({
            "query":             query,
            "skipped":           False,
            "expected_sources":  sorted(expected),
            "retrieved_sources": retrieved,
            "precision@1":       precision_at(1),
            "precision@3":       precision_at(3),
            "precision@5":       precision_at(5),
            "recall@1":          recall_at(1),
            "recall@3":          recall_at(3),
            "recall@5":          recall_at(5),
            "f1":                f1,
            "rr":                rr,
            "latency_ms":        round(latency_ms, 2),
        })

    scored = [q for q in queries if not q["skipped"]]
    avg = lambda key: sum(q[key] for q in scored) / len(scored) if scored else 0.0
    n = len(latencies)
    sorted_lat = sorted(latencies)

    return {
        "avg_precision@1": avg("precision@1"),
        "avg_precision@3": avg("precision@3"),
        "avg_precision@5": avg("precision@5"),
        "avg_recall@1":    avg("recall@1"),
        "avg_recall@3":    avg("recall@3"),
        "avg_recall@5":    avg("recall@5"),
        "avg_f1":          avg("f1"),
        "mrr":             avg("rr"),
        "avg_latency_ms":  sum(latencies) / n,
        "p50_latency_ms":  sorted_lat[min(int(n * 0.50), n - 1)],
        "p95_latency_ms":  sorted_lat[min(int(n * 0.95), n - 1)],
        "max_latency_ms":  max(latencies),
        "skipped_count":   len(queries) - len(scored),
        "scored_count":    len(scored),
        "queries":         queries,
    }


def run_comparison(top_k: int = 5) -> dict:
    """Run all five retrievers and return comparison results."""
    from rag.naive_retriever import get_naive_retriever
    from rag.retriever import get_retriever

    naive    = get_naive_retriever()
    hybrid   = get_retriever()
    bm25rrf  = BM25RRFRetriever()
    stepback = get_stepback_retriever()
    agentic  = get_agentic_retriever()

    # Warmup — loads embedding model + cross-encoder into RAM so cold-start
    # doesn't skew latency measurements (first call can take 20-30s from disk)
    _w = "return policy"
    print("Warming up retrievers...")
    naive.retrieve(_w, top_k)
    bm25rrf.retrieve(_w, top_k)
    hybrid.retrieve(_w, top_k_final=top_k)
    stepback.retrieve(_w, top_k)
    agentic.retrieve(_w, top_k)
    print("Warmup complete. Running evaluation...")

    return {
        "naive":    evaluate(naive.retrieve, top_k),
        "bm25_rrf": evaluate(bm25rrf.retrieve, top_k),
        "stepback": evaluate(stepback.retrieve, top_k),
        "hybrid":   evaluate(lambda q, k: hybrid.retrieve(q, top_k_final=k), top_k),
        "agentic":  evaluate_agentic(agentic, top_k),
        "top_k":    top_k,
    }


# ── CLI output ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_comparison(top_k=5)
    top_k = results["top_k"]

    print(f"\nRAG Evaluation Results  (top_k={top_k})")
    print("=" * 75)

    for name, label in [
        ("naive",    "Naive      (dense-only cosine)"),
        ("bm25_rrf", "BM25+RRF   (no cross-encoder)"),
        ("stepback", "Step-Back  (BM25+dense on broader query)"),
        ("hybrid",   "Hybrid     (BM25 + dense + rerank)"),
        ("agentic",  "Agentic    (decide → retrieve → grade → retry)"),
    ]:
        r = results[name]
        print(f"\n{label}:")
        print(f"  Precision@1: {r['avg_precision@1']:.3f}  "
              f"Recall@1: {r['avg_recall@1']:.3f}")
        print(f"  Precision@3: {r['avg_precision@3']:.3f}  "
              f"Recall@3: {r['avg_recall@3']:.3f}")
        print(f"  Precision@5: {r['avg_precision@5']:.3f}  "
              f"Recall@5: {r['avg_recall@5']:.3f}")
        print(f"  MRR:         {r['mrr']:.3f}")
        print(f"  Avg Latency: {r['avg_latency_ms']:.1f}ms  "
              f"P50: {r['p50_latency_ms']:.1f}ms  "
              f"P95: {r['p95_latency_ms']:.1f}ms  "
              f"Max: {r['max_latency_ms']:.1f}ms")
        if name == "agentic":
            print(f"  Scored queries: {r.get('scored_count', '?')} / "
                  f"{r.get('scored_count', 0) + r.get('skipped_count', 0)}  "
                  f"({r.get('skipped_count', 0)} skipped)")

    print("\n" + "=" * 75)
    print(f"\n{'Query':<45} N-MRR  B-MRR  H-MRR  AG-MRR  N-ms  H-ms  AG-ms")
    print("-" * 95)

    for i, q in enumerate(results["naive"]["queries"]):
        n_mrr  = results["naive"]["queries"][i]["rr"]
        b_mrr  = results["bm25_rrf"]["queries"][i]["rr"]
        h_mrr  = results["hybrid"]["queries"][i]["rr"]
        ag_q   = results["agentic"]["queries"][i]
        ag_mrr = "SKIP" if ag_q.get("skipped") else f"{ag_q['rr']:.2f}"
        n_ms   = results["naive"]["queries"][i]["latency_ms"]
        h_ms   = results["hybrid"]["queries"][i]["latency_ms"]
        ag_ms  = results["agentic"]["queries"][i]["latency_ms"]
        print(
            f"{q['query'][:45]:<45} "
            f"{n_mrr:.2f}   {b_mrr:.2f}   {h_mrr:.2f}   "
            f"{ag_mrr:<6} {n_ms:6.1f} {h_ms:6.1f} {ag_ms:7.1f}"
        )