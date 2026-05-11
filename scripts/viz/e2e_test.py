"""End-to-end test of scripts/viz/index.html in headless Chromium.

Loads index.html via a local HTTP server, waits up to 120s for the Netron
iframe to render g.node elements (Netron renders graphs as inline SVG, not
canvas), and reports counts for each source (ONNX, safetensors).

Uses only Playwright locator APIs because Netron overrides window.eval and
page.evaluate() fails inside the netron.app frame.
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


async def wait_for_netron_render(page, label, screenshot_path, max_wait=120):
    """Wait until the netron iframe renders graph nodes, or max_wait elapses.
    Returns the final node count."""
    iframe_el = page.locator("#netron-frame")
    src = await iframe_el.get_attribute("src")
    print(f"  [{label}] iframe src: {src}")

    # Get the netron Frame object
    netron_frame = None
    for _ in range(15):
        for fr in page.frames:
            if fr.url.startswith("https://netron.app"):
                netron_frame = fr
                break
        if netron_frame:
            break
        await page.wait_for_timeout(1_000)
    if not netron_frame:
        print(f"  [{label}] no netron frame ever appeared")
        await page.screenshot(path=screenshot_path)
        return 0

    last_count = 0
    for t in range(max_wait):
        try:
            n = await netron_frame.locator("svg g.node").count()
            body_class = await netron_frame.locator("body").get_attribute("class")
            if n != last_count:
                print(f"  [{label}] [t={t}s] body={body_class!r}  g.node count={n}")
                last_count = n
            if n > 0 and body_class and "default" in body_class:
                # rendered
                await page.screenshot(path=screenshot_path)
                return n
            if t % 15 == 0 and t > 0:
                print(f"  [{label}] [t={t}s] body={body_class!r}  g.node={n} (still waiting)")
        except Exception as e:
            if t % 10 == 0:
                print(f"  [{label}] [t={t}s] probe exception: {e}")
        await page.wait_for_timeout(1_000)
    await page.screenshot(path=screenshot_path)
    return last_count


async def run():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True, args=["--ignore-certificate-errors"]
        )
        ctx = await browser.new_context(
            viewport={"width": 1400, "height": 900}, ignore_https_errors=True
        )
        page = await ctx.new_page()

        url = f"http://127.0.0.1:{PORT}/index.html"
        print(f"navigating to {url}")
        await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        await page.wait_for_timeout(1_500)

        # === Source 1: ONNX (default) ===
        print("\n=== ONNX source ===")
        onnx_nodes = await wait_for_netron_render(
            page, "onnx", os.path.join(ROOT, "_shot_onnx.png"), max_wait=120
        )

        # === Switch to safetensors ===
        print("\n=== safetensors source ===")
        await page.click("#pane-netron .switch button[data-src='weights']")
        await page.wait_for_timeout(2_000)
        weights_nodes = await wait_for_netron_render(
            page, "weights", os.path.join(ROOT, "_shot_weights.png"), max_wait=120
        )

        # === Other tabs ===
        print("\n=== SVG tab ===")
        await page.click("nav.tabs button[data-tab='graph']")
        await page.wait_for_timeout(1_000)
        svg_count = await page.locator("#pane-graph svg").count()
        nodes_in_svg = await page.locator("#pane-graph svg g.node").count()
        print(f"  svg elements: {svg_count}, g.node: {nodes_in_svg}")
        await page.screenshot(path=os.path.join(ROOT, "_shot_svg.png"))

        print("\n=== Module tree tab ===")
        await page.click("nav.tabs button[data-tab='tree']")
        await page.wait_for_timeout(800)
        tree_nodes = await page.locator(".tree .node").count()
        print(f"  tree nodes rendered: {tree_nodes}")
        await page.screenshot(path=os.path.join(ROOT, "_shot_tree.png"))

        await browser.close()

    print("\n" + "=" * 60)
    print(f"RESULT: onnx={onnx_nodes} g.node  weights={weights_nodes} g.node  "
          f"svg_overview={nodes_in_svg}  tree={tree_nodes}")
    overall = "PASS" if (onnx_nodes > 100 and tree_nodes > 100) else "FAIL"
    print(f"OVERALL: {overall}")


def main():
    srv = start_server()
    try:
        asyncio.run(run())
    finally:
        srv.shutdown()


if __name__ == "__main__":
    main()
