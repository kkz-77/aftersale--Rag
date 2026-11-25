# 本地发布准备记录

发布目录为当前独立目录；未复制或初始化 .git，未创建 GitHub 仓库或推送。原开发仓库未被写入。保留依赖包内部名称以保持 uv.lock 一致，展示名称为「售后通 · AfterSale RAG」。

导出白名单：app 源码与技能定义；frontend/src、入口、类型和 Vite 配置、package.json 与锁文件；knowledge 演示政策；tests 与 HTML 测试夹具；alembic 迁移；启动/入库/迁移/种子脚本；通用 eval.runner、metrics 和四套演示数据集；CI、Docker/Compose/nginx、Python 依赖及锁文件；占位配置模板；MIT LICENSE 和第三方声明。逐文件路径和 SHA-256 见 PUBLIC_FILES.json。

排除：全部旧 docs（面经、简历、个人面试笔记、教程及截图），practice，面试包脚本，真实 .env，data、数据库、聊天/订单运行记录、日志、缓存、虚拟环境、IDE 配置、历史 eval runs/reports/acceptance、临时诊断和定向实验脚本。仅重写公开所需的接口和部署说明。

内容检查：按白名单筛选纯文本源码、政策与合成数据；扫描密钥特征、私网端点、本机绝对路径，未发现匹配。无截图导出，无原始会话或数据库导出。详见 CONTENT_SCAN.json，检查报告只列文件位置和问题类型。自动规则不能证明不存在所有未知形式的敏感信息，发布前仍应人工审阅白名单。

验证：后端 318 项测试通过，覆盖率 76.95%，超过 CI 70% 门槛；前端 2 个测试文件共 3 项通过；npm ci --offline 成功、npm run build 成功；uv lock --check --offline 通过；ruff lint 与全部格式检查通过；mypy 126 个源文件通过。离线启动上下文内 health、ready、capabilities 和 chat 均为 200，聊天返回非空回答。README 相对链接全部存在，模板变量与 Settings 对应，前端代理统一到 8000。

通用检索评测运行成功，65 个有效样本，使用 hash-vector/词法降级，未启用 BGE，不作为 README 中的真实 BGE 成绩。README 使用用户提供的 65 条历史实测口径；未复测真实 Qwen-Flash、BGE 或多轮线上效果。

限制：后端复用原项目现有 Python 环境执行，但工作目录和导入源码均来自发布副本，未全新安装 Python 环境。Docker 命令在本机不可用，未构建镜像或实启 Compose；未执行真实服务网络调用。CI 配置保留，尚未在远端执行。

仍需确认：发布署名（当前使用“我的改造”，未加入个人身份）；演示政策可公开性及历史实测数值的最终背书；发布前真实 Qwen-Flash/BGE 联调及 Docker 环境验证。生产认证和真实业务接入不在此次范围内。
