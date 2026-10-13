#!/usr/bin/env python3
"""Start the Dify adapter and LiteLLM proxy together."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PORT_BRIDGE = 4010
PORT_LITELLM = 4421


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_for_port(port: int, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if port_in_use(port):
            return True
        time.sleep(0.25)
    return False


def terminate_all(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and any(p.poll() is None for p in processes):
        time.sleep(0.1)
    for process in processes:
        if process.poll() is None:
            process.kill()


def load_local_env(env: dict[str, str]) -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env.setdefault(key.strip(), value.strip().strip("\"'"))


def main() -> int:
    venv_root = ROOT / ".venv"
    if (venv_root / "bin/python").exists() and (venv_root / "bin/litellm").exists():
        python = venv_root / "bin/python"
        litellm = venv_root / "bin/litellm"
    else:
        python = Path(sys.executable)
        litellm = Path("litellm")

    env = os.environ.copy()
    load_local_env(env)
    env.setdefault("DIFY_CONFIG_JSON", str(ROOT / "xcg_ai_dify_agent.json"))
    env.setdefault("DIFY_BRIDGE_API_KEY", "sk-dify-local-bridge")
    env.setdefault("DIFY_USER", "claude-code")
    env.setdefault("DIFY_BRIDGE_PORT", str(PORT_BRIDGE))
    env.setdefault("LITELLM_USE_CHAT_COMPLETIONS_URL_FOR_ANTHROPIC_MESSAGES", "true")

    processes: list[subprocess.Popen] = []
    try:
        if port_in_use(PORT_BRIDGE):
            print(f"[Dify bridge] {PORT_BRIDGE} 已被占用，将复用现有服务。")
        else:
            processes.append(subprocess.Popen([str(python), str(ROOT / "dify_openai_bridge.py")], cwd=ROOT, env=env))
            if not wait_for_port(PORT_BRIDGE):
                print("Dify bridge 启动超时。", file=sys.stderr)
                return 1
            print(f"[Dify bridge] 已启动：http://127.0.0.1:{PORT_BRIDGE}")

        if port_in_use(PORT_LITELLM):
            print(f"[LiteLLM] {PORT_LITELLM} 已被占用，将复用现有服务。")
        else:
            processes.append(subprocess.Popen([
                str(litellm),
                "--config", str(ROOT / "config.yaml"),
                "--host", "127.0.0.1",
                "--port", str(PORT_LITELLM),
            ], cwd=ROOT, env=env))
            if not wait_for_port(PORT_LITELLM):
                print("LiteLLM 启动超时。", file=sys.stderr)
                return 1
            print(f"[LiteLLM] 已启动：http://127.0.0.1:{PORT_LITELLM}")

        print("两个服务正在运行，按 Ctrl+C 可一起停止。")
        while True:
            for process in processes:
                if process.poll() is not None:
                    print(f"子进程已退出，返回码：{process.returncode}", file=sys.stderr)
                    return process.returncode or 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("正在停止两个服务……")
        return 0
    finally:
        terminate_all(processes)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    raise SystemExit(main())
