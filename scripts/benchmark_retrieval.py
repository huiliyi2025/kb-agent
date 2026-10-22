"""合成向量精确检索对比；只测检索，不代表端到端服务容量。"""
import argparse
import json
import statistics
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np

from kb_agent.core.db import Database
from kb_agent.retrieval.dense import search_dense


def benchmark(count=10000, dim=256, repeat=3):
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory(prefix="kb-bench-") as directory:
        db = Database(Path(directory) / "bench.db")
        db.init_schema()
        owner = db.create_user("benchmark", "unused")
        doc, _ = db.insert_document(owner, "synthetic", "text/plain", 0, "synthetic")
        matrix = rng.normal(size=(count, dim)).astype("<f4")
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        query = matrix[0].copy()
        db.add_chunks(doc, [{"ordinal": i, "text": "合成测试片段" * 50, "idx_text": "synthetic",
                             "vec": vector.tobytes()} for i, vector in enumerate(matrix)])
        del matrix
        conn = db.connect()
        def reference():
            rows = conn.execute("SELECT c.*, d.filename FROM chunks c JOIN documents d ON d.id=c.doc_id").fetchall()
            mat = np.frombuffer(b"".join(r["vec"] for r in rows), dtype="<f4").reshape(len(rows), dim)
            scores = mat @ query
            return [rows[int(i)]["id"] for i in np.argsort(-scores)[:20]]
        def batched():
            return [h.chunk_id for h in search_dense(conn, query, limit=20, owner_id=owner)]
        results = {}
        try:
            assert reference() == batched(), "Top-K 与全量精确结果不一致"
            for name, fn in [("full_matrix_reference", reference), ("batched_exact", batched)]:
                times, peaks = [], []
                for _ in range(repeat):
                    tracemalloc.start()
                    start = time.perf_counter()
                    fn()
                    times.append((time.perf_counter() - start) * 1000)
                    peaks.append(tracemalloc.get_traced_memory()[1] / 1024 / 1024)
                    tracemalloc.stop()
                results[name] = {"median_ms": round(statistics.median(times), 2),
                                 "tracked_peak_mb": round(max(peaks), 2)}
        finally:
            conn.close()
        return {"chunks": count, "dimension": dim, "repeat": repeat, "top_k_equal": True,
                "note": "synthetic corpus; tracemalloc peak, not process RSS; warm cache", "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, default=10000)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", default="reports/benchmark.json")
    args = parser.parse_args()
    if min(args.chunks, args.dim, args.repeat) <= 0:
        parser.error("chunks, dim, repeat 必须大于 0")
    report = benchmark(args.chunks, args.dim, args.repeat)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
