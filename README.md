# Claude Code Dify Bridge

基于 LiteLLM 的 Claude Code 与 Dify 桥接服务，将 Anthropic 请求转换为 Dify 原生对话接口。

## 一键启动

```bash
cd "/Users/mnb/Desktop/测试项目/Dify_ds 代理"
cp .env.example .env
# 编辑 .env，填写你自己的 DIFY_BASE_URL 和 DIFY_APP_API_KEY
python3 run
```

服务地址：

- Dify 适配器：`http://127.0.0.1:4010`
- LiteLLM：`http://127.0.0.1:4421`

应用密钥优先从 `.env` 中读取。本私有仓库按部署需要保留 `PRIVATE_CONFIG.md` 和
`xcg_ai_dify_agent.json`；请勿将仓库改为公开，或将这两个文件复制到公共仓库。

Nginx HTTPS 和访问控制模板见 [`deploy/nginx/`](deploy/nginx/)。本机开发环境已经绑定到 `127.0.0.1`，不需要 Nginx。

## 配置优先级

环境变量优先于本地 JSON 配置：

- `DIFY_BASE_URL`
- `DIFY_APP_API_KEY`
- `DIFY_APP_MODEL`
- `DIFY_PARAMETER_CHECK_INTERVAL`

如果没有设置环境变量，适配器会读取 `xcg_ai_dify_agent.json`。该文件包含当前私有部署的
Dify 配置，仅适用于本私有仓库；部署地址和凭据说明见 [`PRIVATE_CONFIG.md`](PRIVATE_CONFIG.md)。
