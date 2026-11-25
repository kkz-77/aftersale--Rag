# 部署

复制根目录 `.env.example` 后执行 `docker compose up --build -d`，默认 SQLite 和离线模型，API 提供前端。`docker compose down` 停止服务，数据保留在命名卷。

`docker-compose.single.yml` 提供 PostgreSQL、Redis 组合；`docker-compose.multi.yml` 配合 `deploy/nginx.multi.conf` 提供多实例组合。先用 `docker compose -f <配置文件> config` 检查。需在本地环境设置 POSTGRES_PASSWORD。生产需配置独立机密、认证、TLS、隔离和备份。

镜像默认关闭本地模型与非 Markdown 解析。构建参数 INSTALL_RERANK=true 安装精排依赖，运行时仍需开启对应变量。真实 Qwen-Flash 使用模板中的兼容接口配置。
