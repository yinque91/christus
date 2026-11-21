"""Create Nature-style research figure files with PIL (300 dpi, Arial)."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "NATURE_FIGURES"
OUT.mkdir(exist_ok=True)
A = json.loads((ROOT / "mirror01" / "analysis.json").read_text(encoding="utf-8"))
R = json.loads((ROOT / "mirror01" / "results.json").read_text(encoding="utf-8"))

FONT = r"C:\Windows\Fonts\arial.ttf"
FONTB = r"C:\Windows\Fonts\arialbd.ttf"
INK = (25, 33, 36)
MUTED = (95, 110, 112)
TEAL = (27, 107, 102)
CORAL = (200, 85, 61)
GREY = (155, 168, 170)
PALE = (228, 237, 236)


def f(size, bold=False):
    return ImageFont.truetype(FONTB if bold else FONT, size)


def text(d, xy, s, size=22, fill=INK, bold=False, anchor="la"):
    d.text(xy, s, font=f(size, bold), fill=fill, anchor=anchor)


def axes(im, d, box, xlim, ylim, xlabel, ylabel, xticks=(), yticks=()):
    x0, y0, x1, y1 = box
    d.rectangle([x0, y0, x1, y1], outline=INK, width=2)
    for yt in yticks:
        yy = y1 - (yt - ylim[0]) / (ylim[1] - ylim[0]) * (y1 - y0)
        d.line([x0, yy, x1, yy], fill=(220, 228, 227), width=1)
        text(d, (x0 - 12, yy), str(yt), 21, MUTED, anchor="rm")
    for xt in xticks:
        xx = x0 + (xt - xlim[0]) / (xlim[1] - xlim[0]) * (x1 - x0)
        d.line([xx, y1, xx, y1 + 7], fill=INK, width=2)
        text(d, (xx, y1 + 12), str(xt), 21, MUTED, anchor="ma")
    text(d, ((x0 + x1) / 2, y1 + 48), xlabel, 24, INK, anchor="ma")
    # rotated y label, pasted onto the figure
    lab = Image.new("RGBA", (700, 58), (0, 0, 0, 0))
    dd = ImageDraw.Draw(lab)
    text(dd, (350, 29), ylabel, 24, INK, anchor="mm")
    lab = lab.rotate(90, expand=True)
    im.paste(lab, (x0 - 82, int((y0 + y1) / 2 - lab.height / 2)), lab)


def panel_label(d, xy, s):
    text(d, xy, s, 40, INK, True)


def bar(d, box, frac, color, width=90, label=None, label_size=20):
    x0, y0, x1, y1 = box
    yy = y1 - max(0.0, min(1.0, frac)) * (y1 - y0)
    d.rectangle([x0, yy, x0 + width, y1], fill=color, outline=color)
    if label is not None:
        text(d, (x0 + width / 2, yy - 10), label, label_size, INK, anchor="mb")


def make_results():
    W, H = 2160, 1360
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)

    # a: M1 seed-level correspondence
    panel_label(d, (70, 45), "a")
    box = (180, 130, 1020, 560)
    axes(im, d, box, (0, 21), (0, 0.60), "Seed", "Action–mirror correspondence",
         xticks=(1, 5, 10, 15, 20), yticks=(0, 0.2, 0.4, 0.6))
    y0, y1 = box[1], box[3]
    th = y1 - 0.5 / 0.6 * (y1 - y0)
    d.line([box[0], th, box[2], th], fill=CORAL, width=4)
    text(d, (box[2] - 8, th - 10), "preregistered 0.50", 17, CORAL, anchor="rb")
    vals = [x["summary"]["correspondence"] for x in R["M1"]]
    for i, v in enumerate(vals, 1):
        xx = box[0] + (i - 0.5) / 21 * (box[2] - box[0])
        yy = y1 - max(0, min(0.6, v)) / 0.6 * (y1 - y0)
        d.ellipse([xx - 8, yy - 8, xx + 8, yy + 8], fill=TEAL)
    m = A["M1"]["correspondence"]["mean"]
    lo, hi = A["M1"]["correspondence"]["ci95"]
    xm = box[0] + 20.5 / 21 * (box[2] - box[0])
    ym = y1 - m / 0.6 * (y1 - y0)
    ylo = y1 - lo / 0.6 * (y1 - y0)
    yhi = y1 - hi / 0.6 * (y1 - y0)
    d.line([xm, ylo, xm, yhi], fill=INK, width=4)
    d.line([xm - 12, ylo, xm + 12, ylo], fill=INK, width=2)
    d.line([xm - 12, yhi, xm + 12, yhi], fill=INK, width=2)
    d.ellipse([xm - 10, ym - 10, xm + 10, ym + 10], fill=INK)
    text(d, (190, 600), f"mean {m:.3f}; 95% CI {lo:.3f}–{hi:.3f}; n = 20", 18, MUTED)

    # b: M2 lag peaks
    panel_label(d, (1110, 45), "b")
    box = (1220, 130, 2080, 560)
    axes(im, d, box, (0, 5), (0, 10.5), "Mirror condition", "Evaluator lag peak (steps)",
         yticks=(0, 2, 4, 6, 8, 10))
    conds = [("Realtime", 1.0), ("Delay 0.5 s", 5.0), ("Delay 2 s", 9.0), ("Other", None), ("None", None)]
    bw = 120
    gap = 155
    for i, (lab, val) in enumerate(conds):
        x = box[0] + 45 + i * gap
        if val is None:
            d.rectangle([x, box[3] - 8, x + bw, box[3]], fill=GREY)
            text(d, (x + bw / 2, box[3] - 18), "no peak", 16, MUTED, anchor="mb")
        else:
            bar(d, (x, box[1], box[2], box[3]), val / 10.5, TEAL if val <= 1 else GREY, bw, label=str(int(val)))
        text(d, (x + bw / 2, box[3] + 12), lab, 16, INK, anchor="ma")

    # c: M3 real/unseen marker contact (no fake-marker condition)
    panel_label(d, (70, 700), "c")
    box = (180, 790, 1020, 1210)
    axes(im, d, box, (0, 4), (0, 1.08), "Marker condition", "Correct contact rate",
         yticks=(0, 0.25, 0.50, 0.75, 1.00))
    modes = [("Real", 1.0), ("Unseen", 1.0), ("No marker", 0.0), ("Other", 0.0)]
    bw = 110
    gap = 200
    for i, (lab, contact) in enumerate(modes):
        x = box[0] + 45 + i * gap
        bar(d, (x, box[1], box[2], box[3]), contact, TEAL, bw,
            label=(f"{contact:.3f}" if contact > 0.01 else None))
        text(d, (x + bw / 2, box[3] + 12), lab, 17, INK, anchor="ma")

    # d: A0-A5
    panel_label(d, (1110, 700), "d")
    box = (1220, 790, 2080, 1210)
    axes(im, d, box, (0, 6), (0, 1.08), "Condition", "Correct contact rate",
         yticks=(0, 0.25, 0.50, 0.75, 1.00))
    vals = [A["conditions"][c]["contact"]["mean"] for c in ("A0", "A1", "A2", "A3", "A4", "A5")]
    labels = ["A0", "A1", "A2", "A3", "A4", "A5"]
    for i, (lab, v) in enumerate(zip(labels, vals)):
        x = box[0] + 40 + i * 123
        bar(d, (x, box[1], box[2], box[3]), v, TEAL if v >= 0.7 else GREY, 84,
            label=(f"{v:.3f}" if v > 0.01 else None))
        text(d, (x + 42, box[3] + 12), lab, 19, INK, anchor="ma")

    # shared footer
    text(d, (70, 1300), "All panels: 20 independent seeds; evaluator ground truth withheld from the subject. "
         "M1 preregistered threshold 0.50.", 16, MUTED)
    im.save(OUT / "NATURE_FIG2_mirror_results.png", dpi=(300, 300))
    return OUT / "NATURE_FIG2_mirror_results.png"


def make_environment():
    W, H = 2160, 1100
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    # mirror
    d.rectangle([120, 180, 780, 860], outline=(45, 111, 108), width=5)
    text(d, (135, 195), "mirror plane", 20, (45, 111, 108))
    d.line([450, 180, 450, 860], fill=(215, 228, 226), width=2)
    # reflected body (schematic)
    for name, (x, y) in {"head": (330, 360), "left hand": (270, 560), "right hand": (390, 560),
                         "torso": (330, 650)}.items():
        col = (120, 190, 185) if name != "head" else (150, 210, 205)
        d.ellipse([x - 42, y - 42, x + 42, y + 42], fill=col, outline=(35, 100, 96), width=3)
    d.ellipse([330 - 12, 360 - 12, 330 + 12, 360 + 12], fill=CORAL, outline=(120, 30, 20), width=2)
    text(d, (500, 330), "reflected body", 19, INK)
    text(d, (500, 365), "marker on forehead", 19, CORAL, True)
    # person
    d.ellipse([1010, 300, 1120, 410], fill=(232, 232, 226), outline=(80, 80, 78), width=3)
    d.rectangle([1000, 420, 1130, 720], fill=(232, 232, 226), outline=(80, 80, 78), width=3)
    d.ellipse([930, 510, 990, 570], fill=(232, 232, 226), outline=(80, 80, 78), width=3)
    d.ellipse([1140, 510, 1200, 570], fill=(232, 232, 226), outline=(80, 80, 78), width=3)
    text(d, (1010, 760), "real body", 20, INK)
    text(d, (1010, 795), "marker not visible in first person", 18, MUTED)
    # arrow
    d.line([800, 520, 920, 520], fill=INK, width=3)
    d.polygon([(920, 520), (900, 508), (900, 532)], fill=INK)
    # inset actual frame
    frame = ROOT / "mirror01" / "frames" / "M3_seed12000_marker.png"
    if frame.exists():
        imf = Image.open(frame).convert("RGB")
        imf.thumbnail((650, 430))
        im.paste(imf, (1450, 250))
        d.rectangle([1450, 250, 1450 + imf.width, 250 + imf.height], outline=INK, width=3)
        text(d, (1450, 700), "actual M3 frame: marker appears only in the reflection", 18, MUTED)
    text(d, (70, 980), "Reflection: mirror_x = 32 − (body_x − 32)·scale; the first-person channel contains no marker.", 18, MUTED)
    im.save(OUT / "NATURE_FIG3_mirror_environment.png", dpi=(300, 300))
    return OUT / "NATURE_FIG3_mirror_environment.png"


def main():
    # Figure 1: remove the in-figure title, then resize to 183 mm at 300 dpi.
    src = ROOT / "manuscript" / "figure1_architecture.png"
    im = Image.open(src).convert("RGB")
    im = im.crop((0, 105, im.width, im.height))
    target_w = 2160
    im = im.resize((target_w, int(im.height * target_w / im.width)), Image.LANCZOS)
    im.save(OUT / "NATURE_FIG1_architecture.png", dpi=(300, 300))
    p2 = make_results()
    p3 = make_environment()
    print(p2)
    print(p3)


if __name__ == "__main__":
    main()
