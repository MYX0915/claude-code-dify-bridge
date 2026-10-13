# 私有部署配置

> 本文件包含 Dify 应用密钥和本地代理密钥，仅适用于私有仓库。

## 服务地址

- Dify 管理地址：`http://124.174.7.48`
- Dify 原生对话接口：`http://124.174.7.48/v1/chat-messages`
- 本地 Dify 适配器：`http://127.0.0.1:4010`
- 本地 LiteLLM Anthropic 代理：`http://127.0.0.1:4421`

## Dify 应用密钥

### 销售助手 — DeepSeek-v4-pro

- Base URL：`http://124.174.7.48/v1`
- App API Key：`app-uAvMhefCAn6WgJWdXH0HrbmX`

### 销售助手 — DeepSeek-v4-flash

- Base URL：`http://124.174.7.48/v1`
- App API Key：`app-0A1E9Hq3ZlHpZVyWemL4coIK`

## 本地代理凭据

- Dify 适配器 API Key：`sk-dify-local-bridge`
- LiteLLM master key：`sk-local-claude-bridge`

## Claude Code 环境变量

```json
{
  "ANTHROPIC_BASE_URL": "http://127.0.0.1:4421",
  "ANTHROPIC_AUTH_TOKEN": "sk-local-claude-bridge",
  "ANTHROPIC_MODEL": "claude-sonnet-4-5",
  "ANTHROPIC_SMALL_FAST_MODEL": "claude-sonnet-4-5"
}
```

## 启动

```bash
cd "/Users/mnb/Desktop/测试项目/Dify_ds 代理"
python3 run
```
