"""Playwright-backed browser control (uses installed Chrome, isolated profile)."""

from __future__ import annotations

from pathlib import Path

from wphr.tools.policy import Audit, Policy


class PlaywrightBrowser:
    def __init__(self, policy: Policy | None = None, audit: Audit | None = None,
                 headless: bool = True, channel: str = "chrome"):
        self.policy = policy or Policy()
        self.audit = audit or Audit()
        self.headless = headless
        self.channel = channel
        self.profile = Path(__file__).resolve().parents[2] / "storage" / "browser_profile"
        self.profile.mkdir(parents=True, exist_ok=True)
        self._pw = None
        self.browser = None
        self.page = None

    def __enter__(self):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(channel=self.channel,
                                                headless=self.headless)
        self.page = self.browser.new_page(viewport={"width": 1280, "height": 800})
        return self

    def __exit__(self, *exc):
        try:
            if self.browser:
                self.browser.close()
        finally:
            if self._pw:
                self._pw.stop()

    def goto(self, url: str) -> dict:
        self.policy.check("net.http_get")
        self.page.goto(url, timeout=30000)
        self.audit.log("browser.goto", url, self.page.title(), ok=True)
        return {"url": self.page.url, "title": self.page.title()}

    def text(self, selector: str = "body") -> str:
        return self.page.inner_text(selector)

    def click(self, selector: str) -> dict:
        self.page.click(selector, timeout=15000)
        self.audit.log("browser.click", selector, "clicked", ok=True)
        return {"ok": True}

    def type_text(self, selector: str, text: str, enter: bool = False) -> dict:
        self.page.fill(selector, text)
        if enter:
            self.page.press(selector, "Enter")
        self.audit.log("browser.type", selector, f"{len(text)} chars", ok=True)
        return {"ok": True}

    def scroll(self, dy: int) -> dict:
        self.page.mouse.wheel(0, dy)
        return {"ok": True}

    def screenshot(self, out: str) -> dict:
        p = Path(out).resolve() if Path(out).is_absolute() else (self.policy.workspace / out).resolve()
        if not self.policy.inside(p):
            raise PermissionError("screenshot must stay inside workspace")
        p.parent.mkdir(parents=True, exist_ok=True)
        self.page.screenshot(path=str(p))
        self.audit.log("browser.screenshot", self.page.url, str(p), ok=True, artifacts=[str(p)])
        return {"ok": p.exists(), "path": str(p), "bytes": p.stat().st_size}
