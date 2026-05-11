"""Assemble the static viewer index.html.

Inlines arch.json and the overview SVG into template.html. The detailed SVG
is referenced as a sibling file (too large to inline cheaply).
"""

import os
import json
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
ARCH_JSON = os.path.join(ROOT, "arch.json")
OVERVIEW_SVG = os.path.join(ROOT, "nanowhale_overview.svg")
TEMPLATE = os.path.join(ROOT, "template.html")
OUTPUT = os.path.join(ROOT, "index.html")


def clean_svg(svg: str) -> str:
    """Strip XML declaration and DOCTYPE so the SVG can be embedded inline."""
    svg = re.sub(r"<\?xml[^>]*\?>", "", svg)
    svg = re.sub(r"<!DOCTYPE[^>]*>", "", svg, flags=re.DOTALL)
    return svg.strip()


def main():
    with open(ARCH_JSON) as f:
        arch = json.load(f)
    with open(OVERVIEW_SVG) as f:
        overview_svg = clean_svg(f.read())
    with open(TEMPLATE) as f:
        tpl = f.read()

    out = (tpl
           .replace("/*__ARCH_JSON__*/null", json.dumps(arch, separators=(",", ":")))
           .replace("<!--__OVERVIEW_SVG__-->", overview_svg))

    with open(OUTPUT, "w") as f:
        f.write(out)
    size_kb = os.path.getsize(OUTPUT) / 1024
    print(f"wrote {OUTPUT} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
