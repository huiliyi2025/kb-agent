"""使用独立临时数据库和无头 Edge/Chromium 验证网页；不访问个人浏览器配置。"""
import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def run(channel="msedge"):
    with tempfile.TemporaryDirectory(prefix="kb-browser-") as directory:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        env = {**os.environ, "KB_DB_PATH": str(Path(directory) / "browser.db"),
               "KB_ENV": "test", "KB_LLM_PROVIDER": "fake", "KB_EMBED_PROVIDER": "hashing",
               "KB_EMBED_DIM": "32", "KB_JWT_SECRET": "browser-test-secret-32-characters-123",
               "KB_ADMIN_PASSWORD": "browser-test-password"}
        with open(Path(directory) / "server.log", "w", encoding="utf-8") as log:
            server = subprocess.Popen([sys.executable, "-m", "uvicorn", "kb_agent.api.app:create_app", "--factory",
                                       "--host", "127.0.0.1", "--port", str(port)],
                                      cwd=ROOT, env=env, stdout=log, stderr=log,
                                      creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            try:
                base = f"http://127.0.0.1:{port}"
                for _ in range(100):
                    try:
                        urllib.request.urlopen(base + "/api/health", timeout=1).close()
                        break
                    except OSError:
                        if server.poll() is not None:
                            raise RuntimeError("测试服务器启动失败")
                        time.sleep(0.1)
                else:
                    raise TimeoutError("测试服务器启动超时")
                with sync_playwright() as p:
                    browser = p.chromium.launch(channel=channel or None, headless=True)
                    try:
                        page = browser.new_page(viewport={"width": 1280, "height": 900})
                        errors = []
                        page.on("pageerror", lambda error: errors.append(str(error)))
                        page.goto(base)
                        page.locator("#username").fill("browseruser")
                        page.locator("#password").fill("browser-test-password")
                        page.locator("#register-btn").click()
                        expect(page.locator("#who")).to_have_text("browseruser")
                        page.locator("#file").set_input_files({"name": "test.md", "mimeType": "text/markdown",
                            "buffer": "# 住宿标准\n住宿标准是每晚 600 元。".encode()})
                        page.locator("#upload-btn").click()
                        expect(page.locator("#docs")).to_contain_text("ready", timeout=15000)
                        page.locator("#question").fill("住宿标准是多少？")
                        page.locator("#send-btn").click()
                        expect(page.locator(".msg.bot .meta-line").last).to_contain_text("引用编号校验完成", timeout=15000)
                        expect(page.locator(".msg.bot .bubble").last).to_contain_text("600")
                        expect(page.locator(".cite").last).to_contain_text("test.md")
                        page.reload()
                        expect(page.locator(".msg.bot .bubble").last).to_contain_text("600")
                        page.locator("#new-session-btn").click()
                        expect(page.locator(".msg")).to_have_count(0)
                        page.locator("#sessions").select_option(index=1)
                        expect(page.locator(".msg.bot .bubble").last).to_contain_text("600")
                        page.set_viewport_size({"width": 390, "height": 844})
                        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
                        page.screenshot(path=str(ROOT / "reports" / "browser-mobile.png"), full_page=True)
                        page.set_viewport_size({"width": 1280, "height": 900})
                        page.screenshot(path=str(ROOT / "reports" / "browser-desktop.png"), full_page=True)
                        page.on("dialog", lambda dialog: dialog.accept())
                        page.locator("#docs button").click()
                        expect(page.locator("#docs")).to_contain_text("还没有文档")
                        page.locator("#logout-btn").click()
                        expect(page.locator("#who")).to_have_text("未登录")
                        assert not errors, errors
                        print("Browser smoke OK: register/upload/SSE/citations/reload/history/mobile/delete/logout; 0 page errors")
                    finally:
                        browser.close()
            finally:
                server.terminate()
                server.wait(timeout=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", default="msedge", help="msedge / chrome / 空字符串使用 Playwright Chromium")
    args = parser.parse_args()
    (ROOT / "reports").mkdir(exist_ok=True)
    run(args.channel)
