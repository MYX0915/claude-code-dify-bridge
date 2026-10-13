# Nginx HTTPS 模板

`dify-bridge.conf.example` 是部署模板，不会自动安装或修改系统 Nginx。

使用前需要：

1. 准备证书和域名；
2. 替换 `server_name` 与证书路径；
3. 按实际出口配置 `allow` 白名单；
4. 通过 `nginx -t` 验证后再 reload。

本机开发环境继续使用 `127.0.0.1:4421`，不需要 Nginx。
