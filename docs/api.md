# 本地 API

完整 schema 见启动后的 `/docs`。`POST /api/v1/chat` 接收 `message`、`user_id`、可选 `session_id` 和 `stream`。演示用户 `u_demo`。返回 `answer`、`strategy`、`citations`、`trace_id` 及可能的 `pending_action`。

确认操作时继续使用同一用户与会话，传入待确认 `confirm_action_id`、非空消息和 `idempotency_key`，可通过 Web 卡片操作。user_id 参数不构成生产身份认证。

`GET /api/v1/health` 为存活检查，`/api/v1/ready` 检查数据库，`/api/v1/capabilities` 报告实际后端。
