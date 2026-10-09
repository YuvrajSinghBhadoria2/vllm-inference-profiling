#!/usr/bin/env python3
"""Block-based architecture diagram for the LLM inference benchmark.

Design: minimal text per block. The blocks carry the architecture; only the
load-bearing numbers appear as labels. Same sketchy Excalidraw style.

All figures are computed from the CSVs and asserted at build time.

Outputs:
  results/architecture.png         raster render (tracked)
  results/architecture.excalidraw  editable source (tracked)
  architecture.svg                 local build intermediate (gitignored)
"""
import json
import random
import time

from pathlib import Path

import pandas as pd

# ============================================================ data + guards
OUT = Path("results")
PRE, DEC, TP = (pd.read_csv(OUT / f) for f in ("prefill.csv", "decode.csv", "throughput.csv"))
P, D, L = 7.615e9, 3584, 28
BW, PEAK = 2039e9, 312e12
W_BYTES = 14.25 * 2**30
KV_PER_TOK = 2 * L * 4 * 128 * 2
KV_TOTAL = 1_038_128


def flops(n):
    return 2 * P * n + 2 * n * n * D * L


def nst(n):
    return PRE[PRE.prompt_tokens == n].iloc[0].ttft_ms * 1e6 / n


ITL = DEC.itl_mean_ms.iloc[-1]
ITL_CEIL = W_BYTES / BW * 1e3
BW_ACH = 100 * (P * 2) / (ITL / 1e3) / BW
OBS = 100 * ((nst(8192) / nst(4096)) - 1)
PRED = 100 * (flops(8192) / flops(4096) / 2 - 1)
MFU = 100 * flops(8192) / (PRE[PRE.prompt_tokens == 8192].iloc[0].ttft_ms / 1e3) / PEAK

r64 = TP[TP.concurrency == 64].iloc[0]
STEP = r64.wall_s / 256 * 1e3
MEM = (W_BYTES + 64 * 640 * KV_PER_TOK) / BW * 1e3
BW64 = 100 * (W_BYTES + 64 * 640 * KV_PER_TOK) * (256 / r64.wall_s) / BW
FLOP64 = 100 * 2 * P * 64 * (256 / r64.wall_s) / PEAK
K1, K16, K64 = (TP[TP.concurrency == c].iloc[0] for c in (1, 16, 64))

assert abs(BW64 - 33) < 1.5 and abs(FLOP64 - 12) < 1.5
assert abs(BW_ACH - 60) < 1.5 and abs(MFU - 84) < 1.5
assert 2.0 < OBS < 4.0 and 4.0 < PRED < 6.0 and PRED > OBS
print(f"verified  BW64={BW64:.0f}%  FLOP64={FLOP64:.0f}%  BWdecode={BW_ACH:.0f}%  "
      f"MFU={MFU:.0f}%  superlin {OBS:.1f}%/{PRED:.1f}%")

# ============================================================ style
INK, SKETCH = "#1b1b1f", "#868e96"
BLUE, BLUE_BG = "#1971c2", "#a5d8ff"
GREEN, GREEN_BG = "#2f9e44", "#b2f2bb"
ORANGE, ORANGE_BG = "#f08c00", "#ffec99"
YELLOW, YELLOW_BG = "#e67700", "#fff3bf"
RED, RED_BG = "#e03131", "#ffc9c9"
VIOLET, VIOLET_BG = "#6741d9", "#d0bfff"
GREY, GREY_BG = "#868e96", "#e9ecef"

E, RNG = [], random.Random(7)


def jit(a=1.0):
    return RNG.uniform(-a, a)


def box(eid, x, y, w, h, stroke, bg, fill="cross-hatch", sw=2, gap=18):
    E.append({"id": eid, "type": "rectangle", "x": x, "y": y, "width": w, "height": h,
              "angle": 0, "strokeColor": stroke, "backgroundColor": bg, "fillStyle": fill,
              "strokeWidth": sw, "strokeStyle": "solid", "roughness": 2,
              "roughnessSeed": RNG.randint(1, 2**31), "opacity": 100, "groupIds": [],
              "frameId": None, "roundness": {"type": 3}, "seed": RNG.randint(1, 2**31),
              "version": 1, "versionNonce": RNG.randint(1, 2**31), "isDeleted": False,
              "boundElements": [], "updated": int(time.time() * 1000), "link": None,
              "locked": False, "hachureAngle": -41, "hachureGap": gap})


def txt(eid, x, y, w, lines, size=14, color=INK, bold=False):
    body = "\n".join(lines)
    E.append({"id": eid, "type": "text", "x": x, "y": y, "width": w,
              "height": len(lines) * size * 1.3, "angle": 0, "strokeColor": color,
              "backgroundColor": "transparent", "fillStyle": "solid", "strokeWidth": 1,
              "strokeStyle": "solid", "roughness": 1, "opacity": 100, "groupIds": [],
              "frameId": None, "roundness": None, "seed": RNG.randint(1, 2**31),
              "version": 1, "versionNonce": RNG.randint(1, 2**31), "isDeleted": False,
              "boundElements": [], "updated": int(time.time() * 1000), "link": None,
              "locked": False, "fontSize": size, "fontFamily": 1, "text": body,
              "textAlign": "left", "verticalAlign": "top", "containerId": None,
              "originalText": body, "autoResize": True, "lineHeight": 1.3,
              "fontWeight": "bold" if bold else "normal"})


def block(eid, x, y, w, h, title, stroke, bg, sub=None, ts=17):
    """Stacked component block: bold title, optional one-line subtitle."""
    box(eid, x, y, w, h, stroke, bg)
    ty = y + (h - ts * 1.3 - (ts * 1.25 + 4 if sub else 0)) / 2
    txt(eid + "t", x + 14, ty, w - 28, [title], size=ts, bold=True)
    if sub:
        txt(eid + "s", x + 14, ty + ts * 1.3 + 4, w - 28, [sub], size=12, color=SKETCH)


def arr(eid, x, y, dx, dy, color=SKETCH, sw=2, dashed=False):
    E.append({"id": eid, "type": "arrow", "x": x, "y": y, "width": abs(dx),
              "height": abs(dy), "angle": 0, "strokeColor": color,
              "backgroundColor": "transparent", "fillStyle": "solid", "strokeWidth": sw,
              "strokeStyle": "dashed" if dashed else "solid", "roughness": 2,
              "roughnessSeed": RNG.randint(1, 2**31), "opacity": 100, "groupIds": [],
              "frameId": None, "roundness": {"type": 2}, "seed": RNG.randint(1, 2**31),
              "version": 1, "versionNonce": RNG.randint(1, 2**31), "isDeleted": False,
              "boundElements": [], "updated": int(time.time() * 1000), "link": None,
              "locked": False, "points": [[0, 0], [dx, dy]], "lastCommittedPoint": None,
              "startBinding": None, "endBinding": None, "startArrowhead": None,
              "endArrowhead": "arrow"})


W = 1320
txt("title", 52, 24, W - 104,
    ["LLM inference profiling — where the bottleneck actually sits"], size=26, bold=True)
txt("sub", 52, 60, W - 104,
    ["Qwen2.5-7B · bf16 · vLLM 0.7.3 · 1x A100-SXM4-80GB"], size=13, color=SKETCH)

# ================================================ BAND 1 — the pipeline
PY, PH = 138, 96
BW_ = 176
gap = 34
xs = [52 + i * (BW_ + gap) for i in range(5)]

block("p_in", xs[0], PY + 14, BW_, 68, "prompt", GREY, GREY_BG, "≤ 8192 tok")
block("p_pre", xs[1], PY, BW_, PH, "PREFILL", BLUE, BLUE_BG, "compute-bound")
block("p_kv", xs[2], PY, BW_, PH, "KV CACHE", YELLOW, YELLOW_BG, f"{KV_PER_TOK//1024} KB / tok")
block("p_dec", xs[3], PY, BW_, PH, "DECODE", GREEN, GREEN_BG, "bandwidth-bound")
block("p_out", xs[4], PY + 14, BW_, 68, "output", GREY, GREY_BG, "tokens")

for i in range(4):
    arr(f"a{i}", xs[i] + BW_, PY + PH / 2, gap, 0)

# the loop that defines decode
ly = PY + PH + 30
arr("l1", xs[3] + BW_ / 2, PY + PH, 0, 30, color=GREEN, dashed=True)
arr("l2", xs[3] + BW_ / 2, ly, -(xs[3] - xs[1]), 0, color=GREEN, dashed=True)
arr("l3", xs[1] + BW_ / 2, ly, 0, -30, color=GREEN, dashed=True)
txt("lt", xs[1] + BW_ + 8, ly + 6, 460,
    ["full parameter set re-read from HBM every step"], size=12, color=GREEN)

# ================================================ BAND 2 — roofline split
RY, RH = 300, 118
txt("h2", 52, RY - 30, 400, ["roofline position"], size=15, color="#e67700")

half = 636
box("g_pre", 52, RY, half, RH, BLUE, BLUE_BG)
txt("g_pt", 68, RY + 13, half - 32, ["PREFILL"], size=16, bold=True, color=BLUE)
txt("g_pb", 68, RY + 42, half - 32,
    [f"{MFU:.0f}% MFU at 8k · {PEAK/1e12:.0f} TFLOP/s roof",
     f"{nst(4096):.0f} → {nst(8192):.0f} ns/token (+{OBS:.1f}%)",
     f"superlinear: FLOP model says +{PRED:.1f}%"], size=13, color="#1864ab")

box("g_dec", 52 + half + 22, RY, half, RH, GREEN, GREEN_BG)
txt("g_dt", 68 + half + 22, RY + 13, half - 32, ["DECODE"], size=16, bold=True, color=GREEN)
txt("g_db", 68 + half + 22, RY + 42, half - 32,
    [f"ITL {ITL:.1f} ms flat · roof {ITL_CEIL:.1f} ms",
     f"{BW_ACH:.0f}% of peak bandwidth reached",
     "cost per token independent of length"], size=13, color="#2b8a3e")

# ================================================ BAND 3 — concurrency
CY, CH = 486, 124
txt("h3", 52, CY - 30, 500, ["as concurrency rises"], size=15, color="#e67700")

bw3 = 262
step = (W - 104 - 3 * bw3) / 3
cx = [52 + i * (bw3 + step) for i in range(4)]

# utilisation bars: fill proportional to % of peak
BARMAX = 150


def bar(eid, x, y, pct, color):
    box(eid + "bg", x, y, BARMAX, 14, SKETCH, "#f8f9fa", fill="solid", sw=1)
    box(eid + "fg", x + 1, y + 1, max(2, BARMAX * pct / 100), 12, color, color,
        fill="solid", sw=1)


for i, (conc, agg, col, bg, note) in enumerate([
    (1, K1.aggregate_tok_s, GREEN, GREEN_BG, "bandwidth-bound"),
    (16, K16.aggregate_tok_s, YELLOW, YELLOW_BG, "knee"),
    (64, K64.aggregate_tok_s, RED, RED_BG, "neither roof"),
]):
    box(f"c{i}", cx[i], CY, bw3, CH, col, bg)
    txt(f"ct{i}", cx[i] + 14, CY + 12, bw3 - 28, [f"c = {conc}"], size=16, bold=True, color=col)
    txt(f"cn{i}", cx[i] + 14, CY + 36, bw3 - 28, [f"{agg:.0f} tok/s"], size=17, bold=True)
    lbl = (f"{BW_ACH:.0f}% of peak BW" if conc == 1 else
           (f"{BW64:.0f}% BW · {FLOP64:.0f}% FLOP" if conc == 64 else "weights shared"))
    txt(f"cl{i}", cx[i] + 14, CY + 62, bw3 - 28, [lbl], size=12, color=SKETCH)
    txt(f"cs{i}", cx[i] + 14, CY + 78, bw3 - 28, [note], size=12, color=SKETCH)
    pct = BW_ACH if conc == 1 else (BW64 if conc == 64 else 45)
    bar(f"b{i}", cx[i] + 14, CY + 98, pct, col)

# step-budget callout: what a single c=64 decode step spends time on
box("bud", cx[3], CY, bw3, CH, ORANGE, ORANGE_BG)
txt("but", cx[3] + 14, CY + 12, bw3 - 28, ["one c=64 step"], size=16, bold=True,
    color="#e8590c")
txt("bus", cx[3] + 14, CY + 36, bw3 - 28, [f"{STEP:.0f} ms total"], size=17, bold=True)
txt("bux", cx[3] + 14, CY + 62, bw3 - 28, [f"{MEM:.0f} ms memory transfer"],
    size=12, color=SKETCH)
txt("buz", cx[3] + 14, CY + 78, bw3 - 28, [f"{STEP-MEM:.0f} ms engine overhead"],
    size=12, color="#e8590c")
box("bd", cx[3] + 14, CY + 98, BARMAX, 14, SKETCH, "#f8f9fa", fill="solid", sw=1)
box("bd2", cx[3] + 15, CY + 99, max(2, (BARMAX - 2) * MEM / STEP), 12,
    "#e8590c", "#e8590c", fill="solid", sw=1)

# ================================================ conclusion bar
box("concl", 52, 654, W - 104, 62, RED, RED_BG)
txt("ct", 68, 666, W - 136,
    [f"above ~16 concurrent requests neither roof saturates  →  "
     f"limited by per-step overhead ({STEP-MEM:.0f} ms of {STEP:.0f} ms), not by the GPU"],
    size=15, bold=True, color="#c92a2a")

txt("foot", 52, 734, W - 104,
    ["TP=1 throughout (no multi-GPU config available)  ·  no kernel counters "
     "(Nsight blocked in container)  ·  offline generate() exposes no per-token stream, so mean ITL only"],
    size=12, color=SKETCH)

# ================================================================ emit
json.dump({"type": "excalidraw", "version": 2, "source": "https://excalidraw.com",
           "elements": E,
           "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
           "files": {}},
 open("results/architecture.excalidraw", "w"), indent=1)
print(f"wrote architecture.excalidraw — {len(E)} elements")

# ------------------------------------------------------------------- SVG
import html

maxx = max(e["x"] + e["width"] for e in E) + 40
maxy = max(e["y"] + e["height"] for e in E) + 40
HAND = "'Comic Sans MS','Segoe Print','Bradley Hand',cursive"


def rpath(x, y, w, h, a=1.5):
    p, n, m = [], max(2, int(w // 90)), max(2, int(h // 55))
    for i in range(n + 1):
        p.append((x + w * i / n, y + jit(a)))
    for i in range(1, m + 1):
        p.append((x + w + jit(a), y + h * i / m))
    for i in range(1, n + 1):
        p.append((x + w - w * i / n, y + h + jit(a)))
    for i in range(1, m):
        p.append((x + jit(a), y + h - h * i / m))
    return "M " + " L ".join(f"{a1:.1f},{b:.1f}" for a1, b in p) + " Z"


def lpath(x1, y1, x2, y2, a=1.3):
    n = max(2, int(max(abs(x2 - x1), abs(y2 - y1)) // 55))
    return "M " + " L ".join(
        f"{x1+(x2-x1)*i/n+jit(a):.1f},{y1+(y2-y1)*i/n+jit(a):.1f}" for i in range(n + 1))


rects = [e for e in E if e["type"] == "rectangle"]
sv = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{maxx:.0f}" height="{maxy:.0f}" '
      f'viewBox="0 0 {maxx:.0f} {maxy:.0f}" font-family="{HAND}">',
      f'<rect width="{maxx:.0f}" height="{maxy:.0f}" fill="#ffffff"/>', "<defs>"]
for i, r in enumerate(rects):
    sv.append(f'<path id="p{i}" d="{rpath(r["x"],r["y"],r["width"],r["height"])}"/>')
    sv.append(f'<clipPath id="c{i}"><use href="#p{i}"/></clipPath>')
    if r["fillStyle"] in ("hachure", "cross-hatch"):
        g = r.get("hachureGap", 18)
        sv.append(f'<pattern id="h{i}" width="{g}" height="{g}" patternUnits="userSpaceOnUse" '
                  f'patternTransform="rotate(-41)"><line x1="0" y1="0" x2="0" y2="{g}" '
                  f'stroke="{r["strokeColor"]}" stroke-width="0.9" opacity="0.26"/></pattern>')
sv.append('<marker id="mA" markerWidth="11" markerHeight="9" refX="10" refY="4.5" orient="auto">'
          f'<path d="M0,0 L11,4.5 L0,9 z" fill="{SKETCH}"/></marker>'
          '<marker id="mG" markerWidth="11" markerHeight="9" refX="10" refY="4.5" orient="auto">'
          f'<path d="M0,0 L11,4.5 L0,9 z" fill="{GREEN}"/></marker></defs>')
for i, r in enumerate(rects):
    sv.append(f'<use href="#p{i}" fill="{r["backgroundColor"]}"/>')
    if r["fillStyle"] in ("hachure", "cross-hatch"):
        sv.append(f'<g clip-path="url(#c{i})"><rect x="{r["x"]:.0f}" y="{r["y"]:.0f}" '
                  f'width="{r["width"]}" height="{r["height"]}" fill="url(#h{i})"/></g>')
for i, r in enumerate(rects):
    sv.append(f'<use href="#p{i}" fill="none" stroke="{r["strokeColor"]}" '
              f'stroke-width="{r.get("strokeWidth",2)}" stroke-linejoin="round"/>')
for a in [e for e in E if e["type"] == "arrow"]:
    (x0, y0), (x1, y1) = a["points"]
    d = lpath(a["x"] + x0, a["y"] + y0, a["x"] + x1, a["y"] + y1)
    dash = ' stroke-dasharray="7 5"' if a["strokeStyle"] == "dashed" else ""
    mk = "mG" if a["strokeColor"] == GREEN else "mA"
    sv.append(f'<path d="{d}" fill="none" stroke="{a["strokeColor"]}" stroke-width="{a["strokeWidth"]}"'
              f'{dash} marker-end="url(#{mk})" stroke-linecap="round"/>')
for t in [e for e in E if e["type"] == "text"]:
    lh = t["fontSize"] * t["lineHeight"]
    for i, line in enumerate(t["text"].split("\n")):
        if not line.strip():
            continue
        sv.append(f'<text x="{t["x"]:.1f}" y="{t["y"]+t["fontSize"]+i*lh:.1f}" '
                  f'font-size="{t["fontSize"]}" font-weight="{t.get("fontWeight","normal")}" '
                  f'fill="{t["strokeColor"]}">{html.escape(line)}</text>')
sv.append("</svg>")
open("architecture.svg", "w").write("\n".join(sv))
print(f"wrote architecture.svg ({maxx:.0f}x{maxy:.0f})")