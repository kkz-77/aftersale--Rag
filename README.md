# 售后通 · AfterSale RAG

## 项目简介和演示场景

售后通是一个面向电商售后的智能客服系统，由 FastAPI 提供接口、LangGraph 编排工作流，并使用 React 和 TypeScript 构建对话界面。系统支持带引用的售后政策问答、订单与物流查询，以及需要用户确认的取消订单操作。多轮会话使用 SQLite 保存原始消息、滚动摘要和摘要游标，服务重启后可恢复会话上下文。演示政策与订单使用虚构数据。**用 `u_demo` 查询 `ORD10001` 的物流，用 `ORD10003` 演示取消确认**。

## 五条处理路径

| 路径 | 场景 | 行为 |
| --- | --- | --- |
| DIRECT | 问候和闲聊 | 直接回应，不检索 |
| SINGLE | 单一政策问题 | BM25 与向量召回、RRF 融合、启用时 BGE 精排，提供引用 |
| MULTI | 多条件和跨条款问题 | 查询改写、补充检索和自纠错，受跳数上限约束 |
| AGENT | 订单、物流和售后业务 | 规划并调用工具，校验参数、权限与业务状态 |
| REFUSE | 越权、注入或无法可靠处理 | 拒绝或引导人工处理 |

知识库提供政策证据，业务工具读取或修改结构化状态。高风险写操作先返回 `pending_action`，用户确认后携带 `confirm_action_id` 再执行，通过幂等键避免重复操作。确认不能绕过权限和订单状态检查。见 [接口说明](docs/api.md)。

## 评测结果与复现方式

我的 65 条检索实测中，加入 BGE 精排后，**P@1 从 83.08% 到 89.23%，Recall@3 从 94.62% 到 96.15%**。这是固定演示知识库上的检索成绩，不代表线上业务成功率；本次发布整理未重跑真实模型评测。

```powershell
uv run pytest -q
uv run python -m eval.runner --task retrieval
cd frontend
npm test
npm run build
```

评测使用 [65 条检索数据集](eval/dataset/rag_eval.jsonl)。启用 BGE 和精排后运行，结果写入 `eval/reports/`；检查实际后端，降级结果不能作为 BGE 成绩。`--task intent` 可离线评估规则；`--task all --llm` 会调用模型并产生费用。复现时记录模型、配置及依赖版本；历史评测产物未纳入公开版。

在 65 条售后知识检索样例上，按知识文档 ID 对照标注计算 P@1 和 Recall@3。以下为同一题集上的历史实测结果。

| 检索方案                            |    P@1 | Recall@3 |
| ----------------------------------- | -----: | -------: |
| BM25                                | 80.00% |   90.00% |
| 向量检索                            | 81.54% |   94.62% |
| 混合检索 + RRF                      | 83.08% |   94.62% |
| 混合检索 + RRF + Cross-Encoder 精排 | 89.23% |   96.15% |

## 快速开始

需要 Python 3.11+、uv，以及 Node.js 20.19+ 或 22.12+（推荐 24）。从发布目录执行，PowerShell 示例：

```powershell
uv sync --locked --extra dev
Copy-Item .env.example .env
uv run python -m scripts.seed_demo
uv run uvicorn app.asgi:app --host 127.0.0.1 --port 8000
```

另开终端：

```powershell
cd frontend
npm ci
npm run dev
```

访问 http://127.0.0.1:5173，第一条可试问法：“七天无理由退货需要满足什么条件？” API 文档 http://127.0.0.1:8000/docs；健康检查 `/api/v1/health`，实际能力 `/api/v1/capabilities`。后端启动时初始化 SQLite 并加载 `knowledge/`。演示订单由上面的种子脚本写入。

最少配置就是复制模板：空密钥使用离线替代模型、BM25 和内存后端，可以验证流程，但不代表真实模型质量。接入 Qwen-Flash 时在本地 `.env` 填写 `DEEPSEEK_API_KEY`，保留模板的 `DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL=qwen-flash` 和 `LLM_ENABLE_THINKING=false`。字段名为兼容现有代码保留，实际服务由 URL 和模型名决定。不要提交 `.env`。

启用本地 BGE：执行 `uv sync --locked --extra dev --extra rerank`，将 `LOCAL_EMBEDDING_ENABLED`、`CROSS_ENCODER_ENABLED` 设为 `true`；首次使用需下载模型。PDF/Office 解析另需 `docs` 依赖和系统解析工具，默认关闭。

Docker 演示：`docker compose up --build -d`，访问 http://127.0.0.1:8000。见 [部署说明](docs/deployment.md)。

## 当前适用范围、上游致谢和许可证

适用于本地学习、售后流程原型和离线评测。演示政策和订单不构成真实业务承诺。生产还需可信身份认证、业务授权、隐私治理与人工处理流程，不应直接暴露演示接口。

基于上游 [JEen-wang/Adaptive-rag](https://github.com/JEen-wang/Adaptive-rag) 改造并保留 MIT 版权声明。我的改造包括 Qwen-Flash 接入、可恢复滚动摘要、检索与业务流程评测及公开版整理。沿用 [MIT LICENSE](LICENSE)，依赖与模型按各自许可使用，见 [第三方说明](THIRD_PARTY_NOTICES.md)。
