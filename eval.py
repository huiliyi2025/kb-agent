"""离线评测：把"我觉得效果还行"变成一张可复现的表。

典型结论长这样：
    "20 条标注集上 Recall@5 是 0.92、拒答准确率 1.0；改 chunk_size 后 Recall 掉到 0.85，
     所以最终定在 420。"
这类结论必须来自可重复的脚本。评测集很小但**必须真实标注**，否则数字毫无意义。

三类指标：
    检索层   Recall@K / MRR     —— 和模型无关，纯看检索
    生成层   引用命中率          —— 回答里引用的文档是不是金标文档
    安全层   拒答准确率          —— 没有答案的问题有没有被拒（防幻觉）
"""

from __future__ import annotations

import json
import time
import math
from pathlib import Path

from kb_agent.gen.rag import RagService


def load_cases(path: str | Path) -> list[dict]:
    cases: list[dict] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} 不是合法 JSON：{exc}") from exc
        if not isinstance(case, dict) or not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError(f"{path}:{line_no} 缺少非空 question")
        if case.get("type", "answerable") not in {"answerable", "unanswerable"}:
            raise ValueError(f"{path}:{line_no} type 只能为 answerable / unanswerable")
        if case.get("type", "answerable") == "answerable" and not case.get("gold_doc"):
            raise ValueError(f"{path}:{line_no} 可回答用例必须标注 gold_doc")
        cases.append(case)
    if not cases:
        raise ValueError("评测集不能为空")
    return cases


def evaluate(rag: RagService, cases: list[dict], top_k: int = 5, user_id: int | None = None) -> dict:
    if not cases:
        raise ValueError("评测集不能为空")
    rows: list[dict] = []
    for case in cases:
        question = case["question"]
        gold = case.get("gold_doc")
        answerable = case.get("type", "answerable") == "answerable"

        response = rag.ask(question, top_k=top_k, user_id=user_id)
        ranked_docs = [h.filename for h in response.retrieved]
        rank = next((i + 1 for i, name in enumerate(ranked_docs) if name == gold), 0)
        recall = 1.0 if rank else 0.0
        mrr = 1.0 / rank if rank else 0.0

        cited_docs = {c.filename for c in response.citations}
        citation_hit = None
        if answerable:
            citation_hit = 1.0 if gold in cited_docs else 0.0
        refusal_ok = None
        if not answerable:
            refusal_ok = 1.0 if response.refused else 0.0

        keyword = case.get("expect_keyword")
        keyword_ok = None
        if keyword:
            keyword_ok = 1.0 if keyword in response.answer else 0.0

        rows.append(
            {
                "question": question,
                "type": case.get("type", "answerable"),
                "gold_doc": gold,
                "rank": rank,
                "recall": recall,
                "mrr": mrr,
                "citation_hit": citation_hit,
                "refusal_ok": refusal_ok,
                "keyword_ok": keyword_ok,
                "confidence": response.confidence,
                "refused": response.refused,
                "latency_ms": response.latency_ms,
                "answer": response.answer,
            }
        )

    answerable_rows = [r for r in rows if r["type"] == "answerable"]
    unanswerable_rows = [r for r in rows if r["type"] != "answerable"]

    def mean(values: list[float | None]) -> float:
        clean = [v for v in values if v is not None]
        return round(sum(clean) / len(clean), 4) if clean else 0.0

    summary = {
        "llm": rag.llm.name,
        "embedder": rag.retriever.embedder.name,
        "cases": len(rows),
        "recall_at_k": mean([r["recall"] for r in answerable_rows]),
        "mrr": mean([r["mrr"] for r in answerable_rows]),
        "citation_hit": mean([r["citation_hit"] for r in answerable_rows]),
        "keyword_hit": mean([r["keyword_ok"] for r in answerable_rows]),
        "refusal_accuracy": mean([r["refusal_ok"] for r in unanswerable_rows]),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in rows) / max(1, len(rows))),
        "p95_latency_ms": sorted(r["latency_ms"] for r in rows)[math.ceil(len(rows) * 0.95) - 1],
        "top_k": top_k,
    }
    return {"summary": summary, "rows": rows, "sweep": threshold_sweep(rows)}


def threshold_sweep(rows: list[dict], thresholds: tuple[float, ...] = (0.3, 0.4, 0.5, 0.55, 0.6, 0.7)) -> list[dict]:
    """阈值敏感性分析：同一个 confidence 分数，换不同阈值会怎样。

    这一步比"调参调到 100 分"重要得多——它告诉你指标的**稳健性**。
    只有 20 条标注集时，相邻两个阈值之间可能只差一条用例，这个表用来如实说明
    "阈值不稳，所以保留可调配置 + 评测脚本"。
    """
    table: list[dict] = []
    answerable = [r for r in rows if r["type"] == "answerable"]
    unanswerable = [r for r in rows if r["type"] != "answerable"]
    for t in thresholds:
        kept = sum(1 for r in answerable if r["confidence"] >= t)
        refused = sum(1 for r in unanswerable if r["confidence"] < t)
        accuracy = (kept + refused) / max(1, len(rows))
        table.append(
            {
                "threshold": t,
                "answerable_kept": kept,
                "answerable_total": len(answerable),
                "unanswerable_refused": refused,
                "unanswerable_total": len(unanswerable),
                "accuracy": round(accuracy, 4),
            }
        )
    return table


def to_markdown(report: dict, title: str = "RAG 评测报告") -> str:
    s = report["summary"]
    lines = [
        f"# {title}",
        "",
        f"生成时间（UTC）：{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}",
        f"LLM：{s['llm']}；Embedding：{s['embedder']}。样例集分数不代表真实业务泛化能力。",
        "",
        "## 汇总",
        "",
        "| 指标 | 数值 | 说明 |",
        "| --- | --- | --- |",
        f"| 用例数 | {s['cases']} | 含可回答与不可回答 |",
        f"| Recall@{s['top_k']} | {s['recall_at_k']} | 金标文档是否进了 Top-K |",
        f"| MRR | {s['mrr']} | 金标文档排得越靠前越高 |",
        f"| 引用命中率 | {s['citation_hit']} | 回答引用的文档是否包含金标 |",
        f"| 关键词命中率 | {s['keyword_hit']} | 答案是否包含预期关键数字/术语 |",
        f"| 拒答准确率 | {s['refusal_accuracy']} | 无答案问题被正确拒答的比例 |",
        f"| 平均延迟 | {s['avg_latency_ms']} ms | 当前模型端到端耗时 |",
        f"| P95 延迟 | {s['p95_latency_ms']} ms | nearest-rank 分位数 |",
        "",
        "## 阈值敏感性",
        "",
        "同一个 confidence 分数，选不同阈值时的取舍（口径：可回答的保留率 vs 不可回答的拒答率）：",
        "这里只模拟检索门槛，不会重新生成答案，也不等价于最终回答正确率。",
        "",
        "| 阈值 | 可回答保留 | 不可回答拒答 | 整体准确率 |",
        "| --- | --- | --- | --- |",
    ]
    for row in report.get("sweep", []):
        lines.append(
            f"| {row['threshold']} | {row['answerable_kept']}/{row['answerable_total']} | "
            f"{row['unanswerable_refused']}/{row['unanswerable_total']} | {row['accuracy']} |"
        )
    lines += [
        "",
        "## 明细",
        "",
        "| # | 问题 | 类型 | 金标 | 排名 | 置信度 | 拒答 | 延迟(ms) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, row in enumerate(report["rows"], start=1):
        lines.append(
            f"| {i} | {row['question'][:26]} | {row['type']} | {row['gold_doc'] or '-'} | "
            f"{row['rank'] or '-'} | {row['confidence']} | {'是' if row['refused'] else '否'} | {row['latency_ms']} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_and_save(rag: RagService, dataset: str | Path, top_k: int = 5, out_dir: str | Path = "reports",
                 user_id: int | None = None) -> dict:
    cases = load_cases(dataset)
    report = evaluate(rag, cases, top_k=top_k, user_id=user_id)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    file_path = out_path / f"eval-{stamp}.md"
    file_path.write_text(to_markdown(report), encoding="utf-8")
    file_path.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(file_path)
    return report
