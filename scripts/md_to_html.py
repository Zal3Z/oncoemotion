#!/usr/bin/env python
"""Render a Markdown file to a self-contained, styled HTML file (no deps, offline).

Handles the constructs used in the project docs: h1-h3, tables, ordered/unordered
lists, blockquotes, fenced code, bold/italic/inline-code, [text](url) and <url>
links, horizontal rules, emoji. Output is a single HTML file (inline CSS,
light/dark aware) you can open by double-click.

Usage:
    python scripts/md_to_html.py docs/RELAZIONE.md outputs/reports/relazione.html
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

CSS = """
:root{--bg:#f7f8fa;--panel:#fff;--ink:#1a1d24;--muted:#5a6473;--faint:#8b95a7;--line:#dde1e8;--accent:#0e7490;--accent2:#c2410c;--code:#eef1f5;--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;--sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
@media (prefers-color-scheme:dark){:root{--bg:#0e1116;--panel:#151a21;--ink:#e7ebf2;--muted:#9aa4b5;--faint:#6b7688;--line:#283041;--accent:#22d3ee;--accent2:#fb923c;--code:#1c2430;}}
:root[data-theme="light"]{--bg:#f7f8fa;--panel:#fff;--ink:#1a1d24;--muted:#5a6473;--faint:#8b95a7;--line:#dde1e8;--accent:#0e7490;--accent2:#c2410c;--code:#eef1f5;}
:root[data-theme="dark"]{--bg:#0e1116;--panel:#151a21;--ink:#e7ebf2;--muted:#9aa4b5;--faint:#6b7688;--line:#283041;--accent:#22d3ee;--accent2:#fb923c;--code:#1c2430;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.65;-webkit-font-smoothing:antialiased}
.wrap{max-width:820px;margin:0 auto;padding:40px 22px 80px}
h1{font-size:clamp(24px,4vw,34px);line-height:1.15;margin:.2em 0 .5em;font-weight:720;text-wrap:balance;letter-spacing:-.01em}
h2{font-size:22px;margin:1.6em 0 .4em;padding-top:.5em;border-top:1px solid var(--line);font-weight:660}
h3{font-size:17px;margin:1.3em 0 .3em;font-weight:640;color:var(--accent)}
p{margin:.7em 0}
a{color:var(--accent);text-decoration:none;border-bottom:1px solid color-mix(in srgb,var(--accent) 35%,transparent)}
a:hover{border-bottom-color:var(--accent)}
strong{font-weight:680}
code{font-family:var(--mono);font-size:.9em;background:var(--code);padding:.12em .4em;border-radius:5px}
pre{background:var(--code);border:1px solid var(--line);border-radius:10px;padding:14px 16px;overflow-x:auto}
pre code{background:none;padding:0;font-size:.86em;line-height:1.5}
blockquote{margin:1.1em 0;padding:12px 18px;border-left:3px solid var(--accent2);background:color-mix(in srgb,var(--accent2) 7%,transparent);border-radius:0 10px 10px 0;color:var(--ink)}
blockquote p{margin:.35em 0}
hr{border:none;border-top:1px solid var(--line);margin:2em 0}
ul,ol{margin:.7em 0;padding-left:1.4em}
li{margin:.35em 0}
.tablewrap{overflow-x:auto;margin:1em 0}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:8px 12px;border:1px solid var(--line);text-align:left}
th{background:color-mix(in srgb,var(--accent) 8%,transparent);font-weight:640;font-size:12.5px}
tbody tr:nth-child(even){background:color-mix(in srgb,var(--line) 30%,transparent)}
td:not(:first-child),th:not(:first-child){font-variant-numeric:tabular-nums}
"""


def _inline(t: str) -> str:
    t = html.escape(t, quote=False)
    # inline code first (protect its contents)
    codes: list[str] = []
    def stash(m):
        codes.append(m.group(1)); return f"\x00{len(codes)-1}\x00"
    t = re.sub(r"`([^`]+)`", stash, t)
    # links [text](url)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
    # bare <url>
    t = re.sub(r"&lt;(https?://[^&\s]+)&gt;", r'<a href="\1">\1</a>', t)
    # bold first (may contain a nested *italic*), then italic
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", t)
    # restore code
    t = re.sub(r"\x00(\d+)\x00", lambda m: f"<code>{html.escape(codes[int(m.group(1))], quote=False)}</code>", t)
    return t


def convert(md: str) -> str:
    lines = md.split("\n")
    out: list[str] = []
    i, n = 0, len(lines)
    list_stack: list[str] = []  # 'ul' / 'ol'

    def close_lists():
        while list_stack:
            out.append(f"</{list_stack.pop()}>")

    while i < n:
        ln = lines[i]

        # fenced code
        if ln.strip().startswith("```"):
            close_lists()
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(html.escape(lines[i], quote=False)); i += 1
            i += 1
            out.append("<pre><code>" + "\n".join(buf) + "</code></pre>")
            continue

        # table: a line with | and next line is a separator
        if "|" in ln and i + 1 < n and re.match(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", lines[i+1]):
            close_lists()
            def cells(row): return [c.strip() for c in row.strip().strip("|").split("|")]
            header = cells(ln); i += 2
            rows = []
            while i < n and "|" in lines[i] and lines[i].strip():
                rows.append(cells(lines[i])); i += 1
            th = "".join(f"<th>{_inline(c)}</th>" for c in header)
            trs = "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in rows)
            out.append(f'<div class="tablewrap"><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>')
            continue

        # headings
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            close_lists()
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{_inline(m.group(2))}</h{lvl}>")
            i += 1
            continue

        # hr
        if re.match(r"^\s*---+\s*$", ln):
            close_lists(); out.append("<hr>"); i += 1; continue

        # blockquote (possibly multi-line): soft-wrapped lines join into one
        # paragraph; a blank quoted line starts a new paragraph.
        if ln.lstrip().startswith(">"):
            close_lists()
            buf = []
            while i < n and lines[i].lstrip().startswith(">"):
                buf.append(re.sub(r"^\s*>\s?", "", lines[i])); i += 1
            paras, cur = [], []
            for b in buf:
                if b.strip():
                    cur.append(b.strip())
                elif cur:
                    paras.append(" ".join(cur)); cur = []
            if cur:
                paras.append(" ".join(cur))
            out.append("<blockquote>" + "".join(f"<p>{_inline(p)}</p>" for p in paras) + "</blockquote>")
            continue

        # list item (ordered or unordered) with lazy indented continuation
        m_ol = re.match(r"^(\s*)(\d+)\.\s+(.*)$", ln)
        m_ul = re.match(r"^(\s*)[-*]\s+(.*)$", ln)
        if m_ol or m_ul:
            kind = "ol" if m_ol else "ul"
            text = m_ol.group(3) if m_ol else m_ul.group(2)
            i += 1
            # fold indented continuation lines into the same <li>
            while (i < n and lines[i].strip() and lines[i][:1].isspace()
                   and not re.match(r"^\s*(\d+\.|[-*])\s", lines[i])):
                text += " " + lines[i].strip(); i += 1
            if not list_stack or list_stack[-1] != kind:
                close_lists(); out.append(f"<{kind}>"); list_stack.append(kind)
            out.append(f"<li>{_inline(text)}</li>")
            continue

        # blank
        if not ln.strip():
            close_lists(); i += 1; continue

        # paragraph (gather consecutive non-structural lines)
        close_lists()
        buf = [ln]
        i += 1
        while i < n and lines[i].strip() and not re.match(r"^(#{1,6}\s|>\s?|\s*[-*]\s|\s*\d+\.\s|```|\s*---+\s*$)", lines[i]) and "|" not in lines[i]:
            buf.append(lines[i]); i += 1
        out.append("<p>" + _inline(" ".join(buf)) + "</p>")

    close_lists()
    return "\n".join(out)


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: md_to_html.py <in.md> <out.html> [title]"); return 2
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    title = sys.argv[3] if len(sys.argv) > 3 else src.stem
    body = convert(src.read_text(encoding="utf-8"))
    doc = (f"<!doctype html><html lang=it><head><meta charset=utf-8>"
           f"<meta name=viewport content=\"width=device-width,initial-scale=1\">"
           f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
           f"<body><div class=wrap>{body}</div></body></html>")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(doc, encoding="utf-8")
    print(f"Wrote {dst} ({len(doc)//1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
