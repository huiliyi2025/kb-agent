# kb-agent

把企业制度、操作手册等文档变成可检索、带引用的问答服务。技术栈为 Python、FastAPI、SQLite FTS5、NumPy；默认完全离线运行，既可学习 RAG 全链路，也可接入兼容接口的真实模型。

当前版本 **0.2.0**：完成用户隔离、会话校验、索引可靠性、检索内存优化、前端会话恢复及工程配套。定位仍是本地学习与小规模原型，尚未验证生产部署容量。

## 文档入口

- [项目背景、架构与代码导读](docs/PROJECT_GUIDE.md)：先理解项目解决什么问题、各模块如何配合。
- [优化记录与验证报告](docs/UPGRADE_REPORT.md)：原始问题、已完成改进、测试证据和后续优先级。
- [12 周详细学习路线](docs/LEARNING_ROADMAP.md)：逐周目标、逐日实验、验收标准和毕业项目。
- [检索实验原始数据](docs/benchmark-2026-10-22.json)。

## 快速启动

从源码安装（Python 3.11+）：

```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m kb_agent.cli demo
.\.venv\Scripts\python.exe -m kb_agent.cli serve
```

然后打开 [本地网页](http://127.0.0.1:8000/) 或 [接口文档](http://127.0.0.1:8000/docs)。开发默认账号为 `admin / admin123`；也可以在页面注册新账号。演示语料属于管理员，新账号需要上传自己的文档。

如果 `python` 命令不可用，改用 `py -3` 或解释器绝对路径，不需要修改系统 PATH。

Linux/macOS 使用 `python3 -m venv .venv`，后续将 `.\.venv\Scripts\python.exe` 换为 `.venv/bin/python`。

`requirements-lock.txt` 是 Windows / Python 3.13 验证环境的依赖快照（含测试、构建及浏览器工具）。需要精确复现时先 `pip install -r requirements-lock.txt`，再 `pip install --no-deps -e .`。它不是附带包哈希的跨平台锁文件，其他系统以 `pyproject.toml` 的范围约束安装并运行测试。

## 能做什么

| 能力 | 使用方式 | 当前边界 |
| --- | --- | --- |
| 文档入库 | 网页上传；CLI `ingest 文件或目录` | Markdown / TXT / DOCX / 文本型 PDF；不含 OCR |
| 混合检索 | FTS5 + 归一化向量 + RRF + 覆盖率重排 | 默认哈希向量用于验证流程，不具备真正语义能力 |
| RAG 问答 | 网页提问；CLI `ask "问题"` | 先判断资料充分性；检查引用编号；仍需核对结论 |
| 流式输出 | 网页默认 SSE | `meta → token* → done`；异常为 `error` |
| 多轮会话 | 新建、切换、刷新恢复 | 取最近 4 条历史供生成；不包含指代改写模块 |
| 文档范围 | 勾选文档再提问 | 空选集检索不到文档；Agent 使用本人全部文档 |
| Agent | 网页 Agent；CLI `agent "问题"` | 仅检索与目录工具，最多 8 轮；不执行命令、不上网 |
| 离线评测 | CLI `eval` | 12 条教学样例，不代表业务测试集 |

## 常用命令

```powershell
.\.venv\Scripts\python.exe -m kb_agent.cli init
.\.venv\Scripts\python.exe -m kb_agent.cli ingest .\data\corpus
.\.venv\Scripts\python.exe -m kb_agent.cli ask "公司账号密码至少几位？"
.\.venv\Scripts\python.exe -m kb_agent.cli agent "年假有几天，怎么申请？"
.\.venv\Scripts\python.exe -m kb_agent.cli eval --top-k 5
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe scripts/check_arch.py
node --test tests/sse.test.mjs
.\.venv\Scripts\python.exe scripts/benchmark_retrieval.py --chunks 10000 --dim 256
```

Python 负责后端，Node 仅用于前端解析器测试，不是运行服务的必要条件。评测会同时生成 Markdown 和 JSON 文件到 `reports/`。

浏览器验收（可选，需要本机 Edge）：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[browser]"
.\.venv\Scripts\python.exe scripts/browser_smoke.py --channel msedge
```

它使用临时数据库与无头浏览器，覆盖注册、上传、流式问答、引用、刷新恢复、历史切换、手机布局、删除和退出；截图写入 `reports/`。不读取个人浏览器配置。其他平台可使用已安装的 Chrome 或 Playwright Chromium。

## 配置与真实模型

配置优先级：系统环境变量高于 `.env`。源码运行的 `.env` 和相对数据库路径以项目根为基准；安装 wheel 后默认使用启动目录，可通过 `KB_PROJECT_ROOT` 固定数据根目录，或使用绝对 `KB_DB_PATH`。

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `KB_ENV` | `development` | 可选 `test` / `production` |
| `KB_DB_PATH` | `data/kb.db` | SQLite 数据库；不要提交进 Git |
| `KB_LLM_PROVIDER` | `fake` | `openai` 表示兼容 `/chat/completions` 的服务 |
| `KB_EMBED_PROVIDER` | `hashing` | `openai` 为真实向量服务；`none` 只用 FTS |
| `KB_EMBED_DIM` | `1024` | 必须与实际向量维度一致；示例配置选择 256 |
| `KB_CHUNK_SIZE / KB_CHUNK_OVERLAP` | `420 / 60` | 字符数，不是 token 数；新导入生效 |
| `KB_TOP_K` | `5` | 1~20 |
| `KB_MIN_ANSWERABILITY` | `0.55` | 0~1 的启发式阈值，不是正确率 |
| `KB_MAX_UPLOAD_MB` | `20` | 单文件业务层上限 |
| `KB_RATE_LIMIT_PER_MIN` | `30` | 单进程滑动窗口；认证按来源地址限流 |
| `KB_CORS_ORIGINS` | 空 | 同源网页无需配置；跨域时逗号分隔白名单 |

接真实服务时设置相应的 `KB_LLM_BASE_URL/API_KEY/MODEL`、`KB_EMBED_BASE_URL/API_KEY/MODEL/DIM`。兼容性取决于供应商的实际响应格式，本次只验证了离线流程和模拟上游响应，未调用付费模型。

向量模型名称、维度和服务地址参与索引签名。更换模型时使用新的 `KB_DB_PATH`，重新导入原始文件；旧库不会自动转换。旧版数据库如果已有向量但没有模型签名，也会要求重新建库。可先使用 `KB_EMBED_PROVIDER=none` 读取旧库的全文索引。

生产模式拒绝默认管理员密码、过短 JWT 密钥和通配 CORS；必须显式配置强随机密钥及管理员密码。首次启动创建管理员，后续修改环境变量**不会重置已有账号密码**。完整部署仍需 HTTPS、网关请求体限制、备份恢复、监控与实际压测。

## API 速查

| 方法 / 路径 | 用途 |
| --- | --- |
| `POST /api/auth/register`、`POST /api/auth/login` | 获取 Bearer Token |
| `GET /api/me` | 当前用户 |
| `POST /api/documents` | multipart `file` 上传并同步等待索引完成 |
| `GET /api/documents` | 本人文档列表 |
| `GET /api/documents/{id}/chunks` | 检查分块 |
| `DELETE /api/documents/{id}` | 删除本人文档及索引 |
| `POST /api/chat`、`POST /api/chat/stream` | RAG；支持 `session_id`、`doc_ids`、`top_k` |
| `GET /api/sessions`、`GET /api/sessions/{id}/messages` | 会话列表与最近 100 条消息 |
| `POST /api/agent` | `question` 和 `max_steps` |
| `GET /api/health` | 数据库可用性及模型类型；不探测上游健康 |

除注册、登录和健康检查外，API 都需要认证。管理员也不能通过问答、目录、分块、删除接口访问其他用户的文档。CLI 默认只操作配置中管理员的文档；内部 Python 服务允许省略用户 ID 做可信离线操作，不能直接暴露给未认证入口。

## 数据与故障处理

- 同一用户相同内容只建一份记录。成功后重复上传返回 `reused=true`；处于 `indexing` 时返回当前状态；失败后重新上传会重试。
- 进程在索引中途被终止后，文档可能停留在 `indexing`，当前需要删除该记录并重新上传；后台任务租约与自动恢复属于下一阶段。
- 只保存解析片段和索引，不保留原始二进制文件。重建索引需要自己保留原始资料。
- 同步问答和成功完成的 SSE 才保存一对问答。中断、模型失败不保存半轮；SSE token 是暂态，最终以 `done` 为准。
- 同一会话从多个客户端同时发问，会按完成顺序保存；当前没有会话串行队列。网页会阻止本页重复发送。
