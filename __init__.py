"""kb-agent：一个可跑、可测、可评测的企业知识库问答服务（RAG + Agent）。

分层（对应 Java 项目的包结构）：

    core/       = 基础设施层（config / db / security / 限流）
    ingest/     = 文档接入层（loaders 解析 → splitter 分块 → indexer 落库）
    retrieval/  = 检索层（向量 / 字面 / 融合 / 重排）
    gen/        = 生成层（LLM 客户端 / Prompt / RAG 编排 / Agent）
    api/        = 接口层（FastAPI 路由，等价于 @RestController）

依赖方向永远单向向下：api → gen → retrieval → ingest → core。
"""

__version__ = "0.2.0"
