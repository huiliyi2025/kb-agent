"""检查层级导入方向：api → gen → retrieval → ingest → core。"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAYERS = {"core": 0, "ingest": 1, "retrieval": 2, "gen": 3, "api": 4}
errors = []
for layer, level in LAYERS.items():
    for path in (ROOT / layer).rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            names = [node.module or ""] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names] if isinstance(node, ast.Import) else []
            for name in names:
                parts = name.split(".")
                if len(parts) > 1 and parts[0] == "kb_agent" and LAYERS.get(parts[1], -1) > level:
                    errors.append(f"{path.relative_to(ROOT)}:{node.lineno}: {name}")
if errors:
    raise SystemExit("违反依赖方向：\n" + "\n".join(errors))
print("Architecture imports OK")
