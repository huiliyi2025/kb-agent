"""命令行入口：init / serve / ingest / ask / agent / eval / demo。

为什么要有 CLI？
    1. 克隆下来一条命令就能看到效果，不用配 Postman；
    2. 评测、批量入库这类任务本来就不该走 HTTP（超时、重试都很难受）；
    3. 它让"领域层"和 HTTP 解耦得到验证——同一套 core/retrieval/gen 被两个入口复用。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from kb_agent.core.config import PACKAGE_ROOT, Settings
from kb_agent.gen.agent import Agent, make_tools

CORPUS_DIR = PACKAGE_ROOT / "data" / "corpus"
DATASET = PACKAGE_ROOT / "data" / "eval_cases.jsonl"


def _state(settings: Settings):
    from kb_agent.api.deps import build_state

    return build_state(settings)


def _admin_id(state) -> int:
    row = state.db.get_user(state.settings.admin_user)
    if row is None:  # pragma: no cover
        raise SystemExit("管理员账号不存在，请先执行 kb_agent.cli init")
    return int(row["id"])


def _ingest_paths(state, paths: list[str], owner_id: int) -> None:
    from kb_agent.ingest.indexer import ingest_document
    from kb_agent.ingest.loaders import UnsupportedFormat

    for raw in paths:
        path = Path(raw)
        files = sorted(path.rglob("*")) if path.is_dir() else [path]
        for file in files:
            if not file.is_file():
                continue
            data = file.read_bytes()
            try:
                doc_id, count, reused = ingest_document(
                    state.db,
                    state.settings,
                    state.embedder,
                    owner_id=owner_id,
                    filename=file.name,
                    mime="text/markdown",
                    data=data,
                )
            except UnsupportedFormat as exc:
                print(f"  跳过 {file.name}: {exc}")
                continue
            tag = "复用" if reused else "新增"
            print(f"  {tag} {file.name} -> {doc_id} ({count} chunks)")


def cmd_init(settings: Settings) -> int:
    state = _state(settings)
    conn = state.db.connect()
    try:
        docs = conn.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"]
        chunks = conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]
    finally:
        conn.close()
    print(f"数据库就绪: {settings.db_abspath}")
    print(f"LLM={state.llm.name}  Embedder={state.embedder.name}  文档={docs}  片段={chunks}")
    print(f"管理员账号: {settings.admin_user}（密码由 KB_ADMIN_PASSWORD 配置，不在日志输出）")
    return 0


def cmd_ingest(settings: Settings, paths: list[str]) -> int:
    state = _state(settings)
    print(f"开始入库 {len(paths)} 个路径 ...")
    _ingest_paths(state, paths, _admin_id(state))
    return 0


def cmd_ask(settings: Settings, question: str, top_k: int | None) -> int:
    state = _state(settings)
    started = time.perf_counter()
    answer = state.rag.ask(question, top_k=top_k, user_id=_admin_id(state))
    print(f"\nQ: {question}")
    print(f"A: {answer.answer}\n")
    if answer.citations:
        print("引用：")
        for cite in answer.citations:
            where = cite.filename + (f" / {cite.heading}" if cite.heading else "")
            print(f"  [{cite.index}] {where}  {cite.snippet[:60]}")
    print(
        f"\nconfidence={answer.confidence} refused={answer.refused} "
        f"latency={answer.latency_ms}ms model={answer.model} 总耗时={int((time.perf_counter() - started) * 1000)}ms"
    )
    return 0


def cmd_agent(settings: Settings, question: str, max_steps: int) -> int:
    state = _state(settings)
    citations: list = []
    tools = make_tools(state.retriever, state.db, citations, owner_id=_admin_id(state))
    result = Agent(state.llm, tools, max_steps=max_steps).run(question, citations=citations)
    print(f"\nQ: {question}\n")
    for step in result.steps:
        print(f"[step {step.step}] thought={step.thought}")
        print(f"           action={step.action} input={step.action_input}")
        print(f"           observation={step.observation[:120]}...")
    print(f"\nA: {result.answer}")
    print(f"\n耗时 {result.elapsed_ms}ms，共 {len(result.steps)} 步，引用 {len(result.citations)} 条")
    return 0


def cmd_eval(settings: Settings, dataset: str, top_k: int) -> int:
    from kb_agent.eval import run_and_save

    state = _state(settings)
    report = run_and_save(state.rag, dataset, top_k=top_k, user_id=_admin_id(state))
    print("\n汇总：")
    for key, value in report["summary"].items():
        print(f"  {key} = {value}")
    print(f"\n报告已写入 {report['report_path']}")
    return 0


def cmd_demo(settings: Settings) -> int:
    state = _state(settings)
    owner = _admin_id(state)
    print("[1/3] 导入示例语料 ...")
    _ingest_paths(state, [str(CORPUS_DIR)], owner)

    print("\n[2/3] 提问（走 RAG 链路）...")
    cmd_ask(settings, "出差住宿费一线城市每晚最多报多少？", None)

    print("\n[3/3] 提问（走 Agent 工具循环）...")
    cmd_agent(settings, "年假有几天，怎么申请？", max_steps=4)

    print("\n下一步：python -m kb_agent.cli serve  然后打开 http://127.0.0.1:8000/")
    return 0


def cmd_serve(settings: Settings, host: str, port: int, reload: bool) -> int:
    import uvicorn

    print(f"启动服务：http://{host}:{port}/   (OpenAPI: /docs)")
    uvicorn.run(
        "kb_agent.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kb-agent", description="企业知识库 RAG + Agent 服务")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="初始化数据库并打印当前配置")
    sub.add_parser("demo", help="一键：导入示例语料 + 跑 RAG + 跑 Agent")

    p_ingest = sub.add_parser("ingest", help="把文件或目录导入知识库")
    p_ingest.add_argument("paths", nargs="+")

    p_ask = sub.add_parser("ask", help="命令行提问（RAG）")
    p_ask.add_argument("question")
    p_ask.add_argument("--top-k", type=int, default=None)

    p_agent = sub.add_parser("agent", help="命令行提问（Agent）")
    p_agent.add_argument("question")
    p_agent.add_argument("--max-steps", type=int, default=4)

    p_eval = sub.add_parser("eval", help="跑离线评测")
    p_eval.add_argument("--dataset", default=str(DATASET))
    p_eval.add_argument("--top-k", type=int, default=5)

    p_serve = sub.add_parser("serve", help="启动 HTTP 服务")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows 控制台默认 GBK，输出中文/符号可能炸；这里强制 UTF-8
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    args = build_parser().parse_args(argv)
    settings = Settings.from_env()

    if args.command == "init":
        return cmd_init(settings)
    if args.command == "demo":
        return cmd_demo(settings)
    if args.command == "ingest":
        return cmd_ingest(settings, args.paths)
    if args.command == "ask":
        return cmd_ask(settings, args.question, args.top_k)
    if args.command == "agent":
        return cmd_agent(settings, args.question, args.max_steps)
    if args.command == "eval":
        return cmd_eval(settings, args.dataset, args.top_k)
    if args.command == "serve":
        return cmd_serve(settings, args.host, args.port, args.reload)
    raise SystemExit(f"未知命令 {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
