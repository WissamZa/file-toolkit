#!/usr/bin/env python3
"""Self-test for the document cropper.

Generates synthetic "phone photos" of papers — different photo sizes,
orientations, backgrounds, shadows, tilt, perspective, partly cut-off
sheets, long receipts — with the true paper outline known, then measures
how well the detector finds it and writes a before/after contact sheet so
you can verify the result with your own eyes.

    python crop_selftest.py                 # 48 cases -> selftest_out/
    python crop_selftest.py --n 100 --seed 7
    python crop_selftest.py --photos some_folder   # no truth; sheet only
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

import doc_crop

PHOTO_SIZES = [(2000, 1500), (1512, 2016), (1600, 1200), (1280, 960),
               (960, 1280), (2448, 3264), (1800, 1350), (1080, 1920)]
KINDS = ["normal", "wood", "dark", "lowcontrast", "receipt", "clutter",
         "fullframe", "cutoff", "tilted"]


# --------------------------------------------------------------------------
# Synthetic photo generator
# --------------------------------------------------------------------------
def _background(rng, size, style):
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if style == "wood":
        base = np.array([140, 95, 60], np.float32)
        grain = np.sin(yy / rng.uniform(5, 12) + np.sin(xx / 90) * 4) * 14
        arr = base + grain[..., None] * np.array([1.0, 0.8, 0.6])
    elif style == "dark":
        arr = np.full((h, w, 3), rng.uniform(35, 70), np.float32)
    elif style == "white":
        arr = np.full((h, w, 3), rng.uniform(205, 225), np.float32)
    elif style == "blue":
        arr = np.array([70, 100, 150], np.float32) + np.zeros((h, w, 3), np.float32)
    else:                                              # grey desk
        arr = np.full((h, w, 3), rng.uniform(90, 140), np.float32)
    light = 1.0 + 0.18 * (xx / w - 0.5) * rng.choice([-1, 1]) \
        + 0.12 * (yy / h - 0.5) * rng.choice([-1, 1])
    arr = arr * light[..., None]
    arr += rng.normal(0, 4, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def _paper_texture(rng, pw, ph, receipt=False):
    paper = Image.new("RGB", (pw, ph), tuple(int(v) for v in
                      (rng.uniform(232, 248), rng.uniform(232, 248),
                       rng.uniform(222, 245))))
    d = ImageDraw.Draw(paper)
    margin = int(pw * 0.08)
    y = int(ph * 0.07)
    d.rectangle([margin, y, margin + int(pw * 0.28), y + int(ph * 0.05)],
                fill=(40, 90, 170))
    y += int(ph * 0.1)
    line_h = max(4, int(ph * (0.012 if receipt else 0.016)))
    gap = line_h * 2
    while y < ph - margin:
        width = int((pw - 2 * margin) * rng.uniform(0.35, 1.0))
        d.rectangle([margin, y, margin + width, y + line_h],
                    fill=(rng.integers(20, 70),) * 3)
        y += gap
    d.rectangle([0, 0, pw - 1, ph - 1], outline=(200, 200, 195), width=2)
    return paper


def _persp_coeffs(dst, src):
    """Coefficients mapping output (dst) points to input (src) points."""
    rows, rhs = [], []
    for (x, y), (u, v) in zip(dst, src):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); rhs.append(u)
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y]); rhs.append(v)
    return np.linalg.solve(np.array(rows, float), np.array(rhs, float))


def make_sample(seed, kind=None):
    """-> (PIL image, true quad (4x2 TL,TR,BR,BL), kind)"""
    rng = np.random.default_rng(seed)
    kind = kind or KINDS[seed % len(KINDS)]
    size = PHOTO_SIZES[int(rng.integers(len(PHOTO_SIZES)))]
    # keep the test quick: shrink big sizes (the detector is size-agnostic)
    f = 0.5 if max(size) > 2200 else 1.0
    size = (int(size[0] * f), int(size[1] * f))
    w, h = size
    style = {"wood": "wood", "dark": "dark", "lowcontrast": "white",
             }.get(kind, rng.choice(["wood", "dark", "grey", "blue"]))
    canvas = _background(rng, size, style)

    receipt = kind == "receipt"
    aspect = rng.uniform(0.34, 0.42) if receipt else 1 / 1.414
    if rng.random() < 0.3 and not receipt:
        aspect = 1 / aspect                           # landscape sheet
    target = {"fullframe": 0.93, "receipt": 0.32}.get(kind, rng.uniform(0.35, 0.65))
    ph = np.sqrt(target * w * h / aspect)
    pw = ph * aspect
    scale_fix = min(1.0, 0.97 * w / pw, 0.97 * h / ph) if kind != "cutoff" else 1.0
    pw, ph = pw * scale_fix, ph * scale_fix

    cx = w / 2 + rng.uniform(-0.06, 0.06) * w
    cy = h / 2 + rng.uniform(-0.06, 0.06) * h
    if kind == "cutoff":
        cx += w * 0.22
    tilt = np.deg2rad(rng.uniform(-14, 14) if kind == "tilted"
                      else rng.uniform(-4, 4))
    quad = np.array([[-pw / 2, -ph / 2], [pw / 2, -ph / 2],
                     [pw / 2, ph / 2], [-pw / 2, ph / 2]])
    rot = np.array([[np.cos(tilt), -np.sin(tilt)], [np.sin(tilt), np.cos(tilt)]])
    quad = quad @ rot.T + [cx, cy]
    jitter = 0.012 * max(pw, ph)
    quad = quad + rng.uniform(-jitter, jitter, quad.shape)

    tex = _paper_texture(rng, int(pw), int(ph), receipt).convert("RGBA")
    src = [(0, 0), (tex.width, 0), (tex.width, tex.height), (0, tex.height)]
    coeffs = _persp_coeffs([tuple(p) for p in quad], src)
    warped = tex.transform(size, Image.Transform.PERSPECTIVE, tuple(coeffs),
                           Image.Resampling.BILINEAR)
    alpha = warped.getchannel("A")

    if kind == "clutter":                              # distractor objects
        d = ImageDraw.Draw(canvas)
        for _ in range(3):
            ox, oy = rng.uniform(0, w * 0.8), rng.uniform(0, h * 0.8)
            d.rectangle([ox, oy, ox + w * 0.12, oy + h * 0.08],
                        fill=tuple(int(v) for v in rng.integers(20, 230, 3)))

    shadow = alpha.filter(ImageFilter.GaussianBlur(max(w, h) * 0.008))
    shadow = ImageChops_offset(shadow, int(w * 0.006), int(h * 0.008))
    dark = Image.new("RGB", size, (0, 0, 0))
    canvas = Image.composite(dark, canvas, shadow.point(lambda v: int(v * 0.35)))
    canvas.paste(warped.convert("RGB"), (0, 0), alpha)
    arr = np.asarray(canvas, np.float32)
    arr += rng.normal(0, 3.0, arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(0.8))
    return img, quad, kind


def ImageChops_offset(img, dx, dy):
    out = Image.new(img.mode, img.size, 0)
    out.paste(img, (dx, dy))
    return out


# --------------------------------------------------------------------------
# Metrics + visualization
# --------------------------------------------------------------------------
def _poly_mask(quad, size, scale=0.25):
    w, h = size
    m = Image.new("L", (max(1, int(w * scale)), max(1, int(h * scale))), 0)
    ImageDraw.Draw(m).polygon([(x * scale, y * scale) for x, y in quad], fill=1)
    return np.asarray(m, bool)


def quad_iou(a, b, size):
    ma, mb = _poly_mask(a, size), _poly_mask(b, size)
    union = (ma | mb).sum()
    return float((ma & mb).sum() / union) if union else 0.0


def before_after(img, quad, crop, conf, max_h=360):
    return doc_crop.before_after_sheet(img, quad, crop,
                                       conf >= doc_crop.CONF_OK, max_h)


def run(n=48, seed=0, out="selftest_out"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows, sheets = [], []
    for i in range(n):
        img, truth, kind = make_sample(seed * 1000 + i)
        det = doc_crop.detect_document(img)
        if det.found:
            iou = quad_iou(det.quad, truth, img.size)
            crop = doc_crop.warp_quad(img, det.quad, det.bg)
        else:
            iou, crop = 0.0, img
        rows.append((i, kind, img.size, det.conf, iou, det.warnings))
        sheet = before_after(img, det.quad if det.found else None, crop, det.conf)
        sheet.save(out / f"case_{i:03d}_{kind}.jpg", quality=88)
        sheets.append(sheet)

    print(f"{'#':>3} {'kind':<12} {'size':<11} {'conf':>5} {'IoU':>5}  flags")
    for i, kind, size, conf, iou, warn in rows:
        mark = "OK " if iou >= 0.93 else ("~  " if iou >= 0.85 else "BAD")
        print(f"{i:>3} {kind:<12} {size[0]}x{size[1]:<6} {conf:5.2f} {iou:5.3f} "
              f"{mark} {','.join(warn)}")
    ious = np.array([r[4] for r in rows])
    confs = np.array([r[3] for r in rows])
    good = ious >= 0.90
    flagged = confs < doc_crop.CONF_OK
    print("\n==== summary ====")
    print(f"cases: {n}   mean IoU: {ious.mean():.3f}   median: {np.median(ious):.3f}")
    print(f"IoU >= 0.93: {(ious >= 0.93).mean() * 100:.0f}%   "
          f">= 0.90: {good.mean() * 100:.0f}%   < 0.85: {(ious < 0.85).mean() * 100:.0f}%")
    bad = ~good
    if bad.any():
        print(f"bad results that were flagged for review: "
              f"{(bad & flagged).sum()}/{bad.sum()}")
    print(f"good results wrongly flagged: {(good & flagged).sum()}/{good.sum()}")
    by_kind = {}
    for _, kind, _, _, iou, _ in rows:
        by_kind.setdefault(kind, []).append(iou)
    for kind, vals in sorted(by_kind.items()):
        print(f"  {kind:<12} n={len(vals):<3} mean IoU {np.mean(vals):.3f} "
              f"min {np.min(vals):.3f}")

    # contact sheet (worst first)
    order = np.argsort(ious)
    cols = 2
    cell_w = max(s.width for s in sheets)
    cell_h = sheets[0].height
    pick = list(order[:min(12, n)])
    grid = Image.new("RGB", (cols * cell_w, ((len(pick) + cols - 1) // cols) * cell_h),
                     (16, 19, 24))
    for j, idx in enumerate(pick):
        grid.paste(sheets[idx], ((j % cols) * cell_w, (j // cols) * cell_h))
    grid.save(out / "worst_cases.jpg", quality=88)
    print(f"\nbefore/after images: {out}/case_*.jpg   worst 12: {out}/worst_cases.jpg")
    return ious, confs


def sheet_for_folder(folder, out="selftest_out"):
    """No ground truth — just produce before/after sheets for real photos."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() not in exts:
            continue
        img = ImageOps.exif_transpose(Image.open(p))
        det = doc_crop.detect_document(img)
        crop = doc_crop.warp_quad(img, det.quad, det.bg) if det.found else img
        before_after(img, det.quad if det.found else None, crop,
                     det.conf).save(out / f"{p.stem}_before_after.jpg", quality=90)
        print(f"{p.name}: conf {det.conf:.2f} {','.join(det.warnings)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="selftest_out")
    ap.add_argument("--photos", help="folder of real photos (no ground truth)")
    args = ap.parse_args()
    if args.photos:
        sheet_for_folder(args.photos, args.out)
    else:
        run(args.n, args.seed, args.out)
