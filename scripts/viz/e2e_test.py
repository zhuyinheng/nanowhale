"""End-to-end test for the custom architecture viewer.

Loads scripts/viz/index.html via a local HTTP server, then asserts:
  - the header chips render
  - the SVG scene exists and has block-wraps
  - layer 0 is expanded by default (contains MLA + MoE child blocks)
  - clicking a layer header toggles its expanded state
  - clicking a block updates the side panel
  - the legend / controls / panel exist
"""

import asyncio
import http.server
import os
import socketserver
import threading

from playwright.async_api import async_playwright

ROOT = os.path.dirname(os.path.abspath(__file__))
PORT = 8765


def start_server():
    os.chdir(ROOT)
    handler = http.server.SimpleHTTPRequestHandler
    socketserver.TCPServer.allow_reuse_address = True
    srv = socketserver.TCPServer(("127.0.0.1", PORT), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


async def run():
    failures = 0

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1400, "height": 900})
        page = await ctx.new_page()

        console_errors = []
        page_errors = []
        page.on("console", lambda m: console_errors.append((m.type, m.text)) if m.type == "error" else None)
        page.on("pageerror", lambda e: page_errors.append(str(e)))

        url = f"http://127.0.0.1:{PORT}/index.html"
        print(f"navigating to {url}")
        await page.goto(url, wait_until="domcontentloaded", timeout=15_000)
        await page.wait_for_timeout(800)

        # 1) Header chips rendered
        chip_count = await page.locator("header .chip").count()
        print(f"  header chips: {chip_count}")
        if chip_count < 8:
            print(f"  FAIL: expected ≥ 8 header chips, got {chip_count}")
            failures += 1

        # 2) SVG scene has block-wraps
        block_count = await page.locator("#scene .block-wrap").count()
        print(f"  block-wrap count: {block_count}")
        if block_count < 20:
            print(f"  FAIL: expected ≥ 20 blocks, got {block_count}")
            failures += 1

        # 3) Layer 0 is expanded — MLA region label visible inside layer0
        mla_label_count = await page.locator(".region-label").count()
        layer_card_count = await page.locator(".layer-card").count()
        expanded_count = await page.locator(".layer-card.expanded").count()
        print(f"  layer cards: {layer_card_count}, expanded: {expanded_count}, region labels: {mla_label_count}")
        if layer_card_count != 8:
            print(f"  FAIL: expected 8 layer cards, got {layer_card_count}")
            failures += 1
        if expanded_count != 1:
            print(f"  FAIL: expected exactly 1 layer expanded at load, got {expanded_count}")
            failures += 1
        if mla_label_count < 2:
            print(f"  FAIL: expected at least 2 region labels (MLA + MoE), got {mla_label_count}")
            failures += 1

        # 4) Click layer 1 header to expand it
        await page.click('.layer-card[data-id="layer1"] .header', force=True)
        await page.wait_for_timeout(300)
        expanded_after = await page.locator(".layer-card.expanded").count()
        print(f"  expanded after clicking layer1 header: {expanded_after}")
        if expanded_after != 2:
            print(f"  FAIL: expected 2 expanded after click, got {expanded_after}")
            failures += 1

        # 5) Click "Expand all" button
        await page.click("#expand-all", force=True)
        await page.wait_for_timeout(300)
        expanded_all = await page.locator(".layer-card.expanded").count()
        print(f"  expanded after 'Expand all': {expanded_all}")
        if expanded_all != 8:
            print(f"  FAIL: expected 8 expanded after Expand all, got {expanded_all}")
            failures += 1

        # 6) Click a specific block (e.g. embed) and verify panel updates
        await page.click('#expand-all', force=True)  # collapse all
        await page.wait_for_timeout(200)
        await page.click('.block-wrap[data-id="embed"]', force=True)
        await page.wait_for_timeout(200)
        panel_text = await page.locator("#panel-body").inner_text()
        print(f"  panel after embed click [:140]: {panel_text[:140]!r}")
        if "embed_tokens" not in panel_text:
            print(f"  FAIL: panel did not mention 'embed_tokens'")
            failures += 1
        sel_count = await page.locator("#scene .block.selected").count()
        if sel_count == 0:
            print(f"  FAIL: no .selected block after click")
            failures += 1

        # 7) Click an MLA sub-block (needs layer0 expanded which is default — re-expand it)
        await page.click('#expand-all', force=True)  # back to all expanded? Toggle once more
        await page.wait_for_timeout(150)
        # Click an SDPA block in layer 0
        await page.click('.block-wrap[data-id="L0.sdpa"]', force=True)
        await page.wait_for_timeout(200)
        panel2 = await page.locator("#panel-body").inner_text()
        print(f"  panel after L0.sdpa click [:140]: {panel2[:140]!r}")
        if "SDPA" not in panel2 and "scaled" not in panel2.lower():
            print(f"  FAIL: panel did not mention SDPA")
            failures += 1

        # 8) No console errors / page errors
        print(f"  page errors: {len(page_errors)} · console errors: {len(console_errors)}")
        if page_errors:
            for e in page_errors[:3]:
                print(f"    pageerror: {e}")
            failures += 1
        if console_errors:
            for ty, t in console_errors[:3]:
                print(f"    console {ty}: {t}")
            # Console errors are not necessarily fatal — only count fatal-looking ones
            for ty, t in console_errors:
                if "Uncaught" in t or "TypeError" in t:
                    failures += 1
                    break

        # Save screenshots
        await page.screenshot(path=os.path.join(ROOT, "_shot_default.png"), full_page=False)
        # Take a full-page screenshot too for visual inspection
        await page.screenshot(path=os.path.join(ROOT, "_shot_full.png"), full_page=True)

        await browser.close()

    print("\n" + "=" * 60)
    print(f"RESULT: {'PASS' if failures == 0 else f'FAIL ({failures} issue(s))'}")
    return failures


def main():
    srv = start_server()
    try:
        rc = asyncio.run(run())
        raise SystemExit(rc)
    finally:
        srv.shutdown()


if __name__ == "__main__":
    main()
