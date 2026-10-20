#!/usr/bin/env python3
"""Size-independent document detection for phone photos.

Pure numpy + Pillow (no OpenCV / SciPy), so it adds no dependencies.

Why this exists: photos taken with a phone never share exact dimensions or
paper placement, so any logic keyed to "image is WxH pixels" never matches
the next photo.  Everything here works in *relative* terms: the paper is
segmented from the background, its (possibly tilted) rectangle is fitted,
the edges are snapped to the real gradient at full resolution, and a
confidence score tells the caller whether the result can be trusted.

Public API
    detect_document(img, tolerance)  -> Detection
    expand_quad(quad, l, t, r, b)    -> quad grown outward along its axes
    quad_bounds(quad, w, h)          -> axis-aligned integer box (clamped)
    warp_quad(img, quad, fill)       -> upright crop of a (tilted) quad
"""

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

WORK_SIDE = 480        # long side of the image used for segmentation
REFINE_SIDE = 1600     # long side of the image used for edge snapping
CONF_OK = 0.70         # at/above: trustworthy (below = please review)
CONF_MIN = 0.30        # below: nothing usable was found


@dataclass
class Detection:
    quad: object = None            # 4x2 float array (TL, TR, BR, BL), full-res
    conf: float = 0.0              # 0..1
    method: str = "none"           # which mask found the document
    angle: float = 0.0             # tilt of the rectangle in degrees
    frac: float = 0.0              # share of the photo the document covers
    warnings: list = field(default_factory=list)
    bg: tuple = (255, 255, 255)    # background colour (fill for rotation)

    @property
    def found(self):
        return self.quad is not None


# --------------------------------------------------------------------------
# Small numpy image-processing helpers
# --------------------------------------------------------------------------
def _window_sum(a, k):
    """Sum over a k x k window (same size, edge-replicated borders)."""
    a = np.asarray(a, dtype=np.float64)
    r = k // 2
    p = np.pad(a, ((r, k - 1 - r), (r, k - 1 - r)), mode="edge")
    ii = np.zeros((p.shape[0] + 1, p.shape[1] + 1))
    ii[1:, 1:] = p.cumsum(0).cumsum(1)
    h, w = a.shape
    return (ii[k:k + h, k:k + w] - ii[0:h, k:k + w]
            - ii[k:k + h, 0:w] + ii[0:h, 0:w])


def _blur(arr, k):
    return np.stack([_window_sum(arr[..., c], k) / (k * k)
                     for c in range(arr.shape[2])], axis=-1)


def _erode(mask, k):
    return _window_sum(mask, k) > (k * k - 0.5)


def _dilate(mask, k):
    return _window_sum(mask, k) > 0.5


def _otsu(values):
    hist, edges = np.histogram(values, bins=256, range=(0, 255))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 127.0
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = total - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[int(np.argmax(between))])


def _label(mask):
    """4-connected components by run-length union-find. -> (labels, count)."""
    h, w = mask.shape
    parent = [0]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    labels = np.zeros((h, w), dtype=np.int32)
    prev = []
    for y in range(h):
        row = mask[y]
        if not row.any():
            prev = []
            continue
        d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        cur, j = [], 0
        for s, e in zip(starts, ends):
            label = 0
            while j < len(prev) and prev[j][1] <= s:
                j += 1
            k = j
            while k < len(prev) and prev[k][0] < e:
                other = prev[k][2]
                if label == 0:
                    label = other
                else:
                    ra, rb = find(label), find(other)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
                k += 1
            if label == 0:
                parent.append(len(parent))
                label = len(parent) - 1
            labels[y, s:e] = label
            cur.append((s, e, label))
        prev = cur
    lut = np.array([find(i) for i in range(len(parent))], dtype=np.int32)
    labels = lut[labels]
    uniq = np.unique(labels)
    remap = np.zeros(labels.max() + 1, dtype=np.int32)
    remap[uniq] = np.arange(len(uniq))
    return remap[labels], len(uniq) - 1


def _fill_holes(mask):
    """Fill every background pocket not connected to the image border (the
    text on a paper is background in a 'bright' mask)."""
    labels, n = _label(~mask)
    if n == 0:
        return mask
    border = np.unique(np.concatenate(
        [labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    outside = np.isin(labels, border[border > 0])
    return ~outside


def _largest_component(mask):
    labels, n = _label(mask)
    if n == 0:
        return None
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(np.argmax(counts))


def _hull(points):
    pts = sorted(set(map(tuple, points.tolist())))
    if len(pts) <= 2:
        return np.array(pts, dtype=float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def _extremes(comp):
    """Outer pixel corners of a component: enough points for its hull."""
    pts = []
    rows = np.where(comp.any(axis=1))[0]
    for y in rows:
        xs = np.where(comp[y])[0]
        pts += [(xs[0], y), (xs[-1] + 1, y), (xs[0], y + 1), (xs[-1] + 1, y + 1)]
    cols = np.where(comp.any(axis=0))[0]
    for x in cols:
        ys = np.where(comp[:, x])[0]
        pts += [(x, ys[0]), (x, ys[-1] + 1), (x + 1, ys[0]), (x + 1, ys[-1] + 1)]
    return np.array(pts, dtype=float)


def _rect_in_frame(points, ang):
    """Extents (x0, x1, y0, y1) of `points` in a frame rotated by `ang`."""
    c, s = np.cos(ang), np.sin(ang)
    xr = points[:, 0] * c + points[:, 1] * s
    yr = -points[:, 0] * s + points[:, 1] * c
    return xr.min(), xr.max(), yr.min(), yr.max()


def _min_area_rect(hull):
    """(angle_rad, x0, x1, y0, y1) of the minimum-area rotated rectangle,
    searched over -45..45 degrees."""
    angles = np.deg2rad(np.arange(-45, 45.01, 0.5))
    c, s = np.cos(angles)[:, None], np.sin(angles)[:, None]
    xr = hull[None, :, 0] * c + hull[None, :, 1] * s
    yr = -hull[None, :, 0] * s + hull[None, :, 1] * c
    x0, x1, y0, y1 = xr.min(1), xr.max(1), yr.min(1), yr.max(1)
    i = int(np.argmin((x1 - x0) * (y1 - y0)))
    return float(angles[i]), x0[i], x1[i], y0[i], y1[i]


def _frame_to_quad(ang, x0, x1, y0, y1):
    u = np.array([np.cos(ang), np.sin(ang)])
    v = np.array([-np.sin(ang), np.cos(ang)])
    return np.array([x0 * u + y0 * v, x1 * u + y0 * v,
                     x1 * u + y1 * v, x0 * u + y1 * v])


# --------------------------------------------------------------------------
# Candidate segmentations of "document vs. background"
# --------------------------------------------------------------------------
def _candidate_masks(arr, tolerance):
    h, w, _ = arr.shape
    blur = _blur(arr, 3)
    lum = blur @ np.array([0.299, 0.587, 0.114])
    frame = max(3, int(min(h, w) * 0.04))
    border = np.zeros((h, w), dtype=bool)
    border[:frame] = border[-frame:] = True
    border[:, :frame] = border[:, -frame:] = True

    out = []
    # A) brightness split (Otsu): the document is whichever side the photo
    #    border mostly is NOT.
    bright = lum > _otsu(lum)
    out.append(("bright", bright if bright[border].mean() <= 0.5 else ~bright))

    # B) colour distance from the border colour (strict and sensitive).
    bg = np.median(blur[border], axis=0)
    diff = np.abs(blur - bg).max(axis=2)
    spread = float(np.median(diff[border]))
    out.append(("contrast", diff > max(float(tolerance), 4.0 * spread)))
    out.append(("contrast-low", diff > max(8.0, 0.4 * tolerance, 3.0 * spread)))

    # C) hue/chroma distance: same brightness, different colour.
    chroma = np.stack([blur[..., 0] - blur[..., 1],
                       blur[..., 1] - blur[..., 2]], axis=-1)
    cd = np.abs(chroma - np.median(chroma[border], axis=0)).max(axis=2)
    out.append(("chroma", cd > max(10.0, 4.0 * float(np.median(cd[border])))))

    # D) closed-edge region: strong brightness/colour gradients mark the
    #    sheet's outline; whatever the photo border cannot reach without
    #    crossing an outline is the sheet. Works when paper and table have
    #    nearly the same brightness but a thin visible edge/shadow.
    smooth = _blur(blur, 3)
    gx = np.abs(np.diff(smooth, axis=1, prepend=smooth[:, :1])).max(axis=2)
    gy = np.abs(np.diff(smooth, axis=0, prepend=smooth[:1])).max(axis=2)
    grad = np.maximum(gx, gy)
    for thr in (2.5, 4.0):
        edges = _dilate(grad > thr, 5)
        labels, n = _label(~edges)
        if n:
            touching = np.unique(np.concatenate(
                [labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
            reach = np.isin(labels, touching[touching > 0])
            out.append(("edges", ~reach))
    return out, tuple(int(v) for v in bg)


def _clean_mask(mask):
    h, w = mask.shape
    side = min(h, w)
    k_open = max(3, int(side * 0.015)) | 1
    k_close = max(5, int(side * 0.03)) | 1
    m = _dilate(_erode(mask, k_open), k_open)
    if not m.any():
        return None
    m = _erode(_dilate(m, k_close), k_close)
    m = _fill_holes(m)
    return _largest_component(m)


def _describe(comp, h, w):
    """Fit a minimum-area rectangle to a component -> dict, or None."""
    area = float(comp.sum())
    if area < 0.02 * h * w:
        return None
    pts = _extremes(comp)
    hull = _hull(pts)
    if len(hull) < 3:
        return None
    ang, x0, x1, y0, y1 = _min_area_rect(hull)
    if abs(np.degrees(ang)) < 1.0:              # treat tiny tilt as upright
        ang = 0.0
        x0, x1, y0, y1 = _rect_in_frame(hull, 0.0)
    rect_area = max(1.0, (x1 - x0) * (y1 - y0))
    return {"area": area, "ang": ang, "x0": x0, "x1": x1, "y0": y0, "y1": y1,
            "fill": min(1.0, area / rect_area)}


# --------------------------------------------------------------------------
# Edge snapping at higher resolution + confidence scoring
# --------------------------------------------------------------------------
def _sample(arr, xs, ys):
    h, w = arr.shape[:2]
    xi = np.clip(np.rint(xs).astype(int), 0, w - 1)
    yi = np.clip(np.rint(ys).astype(int), 0, h - 1)
    return arr[yi, xi]


def _refine(lum, rect, win):
    """Snap each side of `rect` (frame coords at `lum` resolution) onto the
    strongest brightness edge within +-win px. Sides without a consistent,
    clear edge are left where they were."""
    ang = rect["ang"]
    u = np.array([np.cos(ang), np.sin(ang)])
    v = np.array([-np.sin(ang), np.cos(ang)])
    ext = {k: rect[k] for k in ("x0", "x1", "y0", "y1")}
    offs = np.arange(-win, win + 1)
    ts = np.linspace(0.15, 0.85, 25)
    for key in ("x0", "x1", "y0", "y1"):
        horizontal = key in ("x0", "x1")           # side is a vertical line
        if horizontal:
            span = ext["y0"] + ts * (ext["y1"] - ext["y0"])
            fx = ext[key] + offs[None, :]
            fy = span[:, None] + 0 * offs[None, :]
        else:
            span = ext["x0"] + ts * (ext["x1"] - ext["x0"])
            fy = ext[key] + offs[None, :]
            fx = span[:, None] + 0 * offs[None, :]
        px = fx * u[0] + fy * v[0]
        py = fx * u[1] + fy * v[1]
        prof = _sample(lum, px, py).astype(np.float64)
        k = 5
        kernel = np.ones(k) / k
        prof = np.apply_along_axis(
            lambda r: np.convolve(np.pad(r, k // 2, mode="edge"), kernel,
                                  mode="valid"), 1, prof)
        grad = np.abs(prof[:, 2:] - prof[:, :-2])
        best = grad.argmax(axis=1)
        strength = grad.max(axis=1)
        good = strength > 6.0
        if good.mean() < 0.5:
            continue
        d = offs[1:-1][best[good]]
        med = float(np.median(d))
        if np.percentile(np.abs(d - med), 75) > max(2.0, win * 0.3):
            continue
        ext[key] += med
    refined = dict(rect)
    refined.update(ext)
    if refined["x1"] - refined["x0"] < 4 or refined["y1"] - refined["y0"] < 4:
        return rect
    return refined


def _edge_contrast(rgb, rect, depth):
    """Per side: colour gap between a strip just inside and just outside the
    rectangle. Returns (list of contrasts for sides fully inside the photo,
    number of sides touching/leaving the photo)."""
    h, w = rgb.shape[:2]
    ang = rect["ang"]
    u = np.array([np.cos(ang), np.sin(ang)])
    v = np.array([-np.sin(ang), np.cos(ang)])
    ts = np.linspace(0.15, 0.85, 25)
    contrasts, touching, inside = [], 0, []
    for key in ("x0", "x1", "y0", "y1"):
        horizontal = key in ("x0", "x1")
        inward = 1.0 if key in ("x0", "y0") else -1.0
        if horizontal:
            span = rect["y0"] + ts * (rect["y1"] - rect["y0"])
            fa, fb = rect[key] + inward * depth, rect[key] - inward * depth
            pts = [(fa + 0 * span, span), (fb + 0 * span, span)]
        else:
            span = rect["x0"] + ts * (rect["x1"] - rect["x0"])
            fa, fb = rect[key] + inward * depth, rect[key] - inward * depth
            pts = [(span, fa + 0 * span), (span, fb + 0 * span)]
        samples = []
        outside = 0.0
        for fx, fy in pts:
            acc = []
            for dmul in (0.5, 1.0, 1.5):          # a few depths per strip
                shift = (dmul - 1.0) * depth * (1.0 if fx is pts[0][0] else -1.0)
                if horizontal:
                    gx, gy = fx + (shift if fx is pts[0][0] else shift), fy
                else:
                    gx, gy = fx, fy + shift
                px, py = gx * u[0] + gy * v[0], gx * u[1] + gy * v[1]
                outside = max(outside, float(np.mean(
                    (px < 0) | (px > w - 1) | (py < 0) | (py > h - 1))))
                acc.append(_sample(rgb, px, py).reshape(-1, 3))
            cols = np.concatenate(acc)
            # brightest half only: compares paper-with-paper-colour and
            # ignores printed text, so the edge of a text block is not
            # mistaken for the edge of the sheet
            lum = cols @ np.array([0.299, 0.587, 0.114])
            samples.append(cols[lum >= np.median(lum)].mean(axis=0))
        inside.append(samples[0])
        if outside > 0.3 or _near_border(rect, key, w, h, depth):
            touching += 1
            continue
        contrasts.append(float(np.abs(samples[0] - samples[1]).max()))
    return contrasts, touching, inside


def _inner_edge(lum, rect, key):
    """For a side lying on the photo border there is no boundary evidence,
    so look INSIDE the rectangle for a long straight edge parallel to that
    side: that is usually the real sheet edge, meaning the rectangle is too
    big. Printed text gives broken edges (gaps between lines); a sheet edge
    is continuous. Returns the share of the span covered by the best edge."""
    ang = rect["ang"]
    u = np.array([np.cos(ang), np.sin(ang)])
    v = np.array([-np.sin(ang), np.cos(ang)])
    vertical_side = key in ("x0", "x1")
    sign = 1.0 if key in ("x0", "y0") else -1.0
    length = (rect["x1"] - rect["x0"]) if vertical_side else (rect["y1"] - rect["y0"])
    ts = np.linspace(0.12, 0.88, 40)
    if vertical_side:
        span = rect["y0"] + ts * (rect["y1"] - rect["y0"])
    else:
        span = rect["x0"] + ts * (rect["x1"] - rect["x0"])
    ds = np.arange(0.04 * length, 0.65 * length, 2.0)[:, None]      # (n,1)
    sp = span[None, :]                                               # (1,m)
    mid = float(span.mean())
    best = 0.0
    for tilt in np.tan(np.deg2rad(np.arange(-6.0, 6.1, 1.5))):
        pos = rect[key] + sign * ds + (sp - mid) * tilt              # (n,m)
        vals = []
        for off in (-2.0, 2.0):
            if vertical_side:
                fx, fy = pos + off, 0 * pos + sp
            else:
                fx, fy = 0 * pos + sp, pos + off
            px, py = fx * u[0] + fy * v[0], fx * u[1] + fy * v[1]
            vals.append(_sample(lum, px, py).astype(np.float64))
        step = np.abs(vals[1] - vals[0])
        # only weak steps: paper-vs-table edges are a few levels; printed
        # ink is a huge step and must not count as a sheet edge
        cover = ((step > 4.0) & (step < 45.0)).mean(axis=1)
        if cover.size:
            best = max(best, float(cover.max()))
    return best


def _near_border(rect, key, w, h, depth):
    ang = rect["ang"]
    u = np.array([np.cos(ang), np.sin(ang)])
    v = np.array([-np.sin(ang), np.cos(ang)])
    if key in ("x0", "x1"):
        mid = 0.5 * (rect["y0"] + rect["y1"])
        p = rect[key] * u + mid * v
    else:
        mid = 0.5 * (rect["x0"] + rect["x1"])
        p = mid * u + rect[key] * v
    m = max(2.0, depth * 0.8)
    return bool(p[0] < m or p[1] < m or p[0] > w - m or p[1] > h - m)


def _score(fill, frac, contrasts, touching, purity=1.0):
    rect_score = float(np.clip((fill - 0.78) / 0.17, 0, 1))
    if contrasts:
        edge_score = 0.5 * np.mean([min(1.0, c / 25.0) for c in contrasts]) \
            + 0.5 * min(1.0, min(contrasts) / 15.0)
    else:
        edge_score = 0.0
    if frac > 0.985:
        size_score = 0.4
    elif frac < 0.08:
        size_score = 0.1
    elif frac < 0.15:
        size_score = 0.6
    else:
        size_score = 1.0
    base = 0.40 * rect_score + 0.40 * float(edge_score) + 0.20 * size_score
    # A sheet is one colour; sheet + part of the table is not.
    return base * (0.45 + 0.55 * float(purity))


# --------------------------------------------------------------------------
# Public: detection
# --------------------------------------------------------------------------
def detect_document(img, tolerance=28):
    """Find the document in `img` (orientation already corrected)."""
    rgb_img = img.convert("RGB")
    W, H = rgb_img.size
    if min(W, H) < 16:
        return Detection(warnings=["too_small"])

    s1 = min(1.0, WORK_SIDE / max(W, H))
    small = rgb_img if s1 == 1.0 else rgb_img.resize(
        (max(16, round(W * s1)), max(16, round(H * s1))), Image.BOX)
    arr = np.asarray(small, dtype=np.float64)
    sh, sw = arr.shape[:2]

    s2 = min(1.0, REFINE_SIDE / max(W, H))
    ref_img = rgb_img if s2 == 1.0 else rgb_img.resize(
        (max(16, round(W * s2)), max(16, round(H * s2))), Image.BOX)
    ref_rgb = np.asarray(ref_img, dtype=np.float64)
    ref_lum = ref_rgb @ np.array([0.299, 0.587, 0.114])
    rh, rw = ref_lum.shape
    win = max(6, int(round(0.025 * min(rh, rw))))
    depth = max(3.0, 0.012 * min(rh, rw))

    candidates, bg = _candidate_masks(arr, tolerance)
    blur = _blur(arr, 3)
    state = {"best": None}

    def evaluate(name, mask):
        comp = _clean_mask(mask)
        if comp is None:
            return
        rect = _describe(comp, sh, sw)
        if rect is None:
            return
        k = rw / sw                       # snap edges at higher resolution
        scaled = dict(rect)
        for key in ("x0", "x1", "y0", "y1"):
            scaled[key] = rect[key] * k
        scaled = _refine(ref_lum, scaled, win)
        quad_ref = _frame_to_quad(scaled["ang"], scaled["x0"], scaled["x1"],
                                  scaled["y0"], scaled["y1"])
        frac = ((scaled["x1"] - scaled["x0"]) * (scaled["y1"] - scaled["y0"])
                / (rw * rh))
        contrasts, touching, inside = _edge_contrast(ref_rgb, scaled, depth)
        # every side's inside strip should be the same paper colour
        spread = float((np.max(inside, axis=0) - np.min(inside, axis=0)).max())
        purity = float(np.clip(1.0 - (spread - 12.0) / 30.0, 0.0, 1.0))
        conf = _score(rect["fill"], min(frac, 1.0), contrasts, touching,
                      purity)
        inner = False
        for key in ("x0", "x1", "y0", "y1"):
            if _near_border(scaled, key, rw, rh, depth) \
                    and _inner_edge(ref_lum, scaled, key) > 0.88:
                inner = True
                break
        if inner:
            conf *= 0.5
        cur = state["best"]
        if cur is None or conf > cur["conf"]:
            state["best"] = {"name": name, "conf": conf, "quad_ref": quad_ref,
                             "ang": scaled["ang"], "frac": frac,
                             "touching": touching, "fill": rect["fill"],
                             "comp": comp, "inner": inner}

    for name, mask in candidates:
        evaluate(name, mask)

    # Second pass: learn the real paper colour from the best region so far
    # (brightest half of its pixels, which skips printed text) and grow the
    # mask to everything that colour covers. Fixes the classic failure where
    # only the text block of a white sheet on a white table is found.
    first = state["best"]
    if first is not None and first["conf"] < 0.97:
        vals = blur[first["comp"]]
        lumv = vals @ np.array([0.299, 0.587, 0.114])
        paper = np.median(vals[lumv >= np.median(lumv)], axis=0)
        gap = np.abs(blur - paper).max(axis=2)
        for thr in (6.0, 10.0, 16.0, 24.0):      # strict -> lenient
            evaluate("paper-colour", gap < thr)
    best = state["best"]

    if best is None:
        return Detection(warnings=["nothing_found"], bg=bg)

    quad = best["quad_ref"] / s2
    warnings = []
    if best["conf"] < CONF_OK:
        warnings.append("low_confidence")
    if best.get("inner"):
        warnings.append("inner_edge")
    if best["frac"] < 0.15:
        warnings.append("small_document")
    if best["frac"] > 0.97 or best["touching"] >= 4:
        warnings.append("full_frame")
    elif best["touching"] >= 1:
        warnings.append("touches_edge")
    if abs(np.degrees(best["ang"])) > 3.0:
        warnings.append("tilted")
    return Detection(quad=quad, conf=float(best["conf"]), method=best["name"],
                     angle=float(np.degrees(best["ang"])),
                     frac=float(min(best["frac"], 1.0)),
                     warnings=warnings, bg=bg)


# --------------------------------------------------------------------------
# Geometry helpers for the caller
# --------------------------------------------------------------------------
def quad_bounds(quad, width, height):
    """Axis-aligned integer box (l, t, r, b) around `quad`, clamped."""
    q = np.asarray(quad, dtype=float)
    l = int(max(0, np.floor(q[:, 0].min())))
    t = int(max(0, np.floor(q[:, 1].min())))
    r = int(min(width, np.ceil(q[:, 0].max())))
    b = int(min(height, np.ceil(q[:, 1].max())))
    return l, t, max(l + 1, r), max(t + 1, b)


def expand_quad(quad, left, top, right, bottom):
    """Grow a rectangle-like quad outward along its own axes (pixels;
    negative values shrink it)."""
    q = np.asarray(quad, dtype=float)
    u = q[1] - q[0]
    v = q[3] - q[0]
    u = u / (np.linalg.norm(u) or 1.0)
    v = v / (np.linalg.norm(v) or 1.0)
    return np.array([q[0] - u * left - v * top,
                     q[1] + u * right - v * top,
                     q[2] + u * right + v * bottom,
                     q[3] - u * left + v * bottom])


def warp_quad(img, quad, fill=(255, 255, 255)):
    """Cut `quad` (TL, TR, BR, BL) out of `img` as an upright image."""
    q = np.asarray(quad, dtype=float)
    width = int(round((np.linalg.norm(q[1] - q[0])
                       + np.linalg.norm(q[2] - q[3])) / 2))
    height = int(round((np.linalg.norm(q[3] - q[0])
                        + np.linalg.norm(q[2] - q[1])) / 2))
    width, height = max(1, width), max(1, height)
    data = (q[0][0], q[0][1], q[3][0], q[3][1],
            q[2][0], q[2][1], q[1][0], q[1][1])        # NW, SW, SE, NE
    base = img.convert("RGB") if img.mode not in ("RGB", "L") else img
    return base.transform((width, height), Image.Transform.QUAD, data,
                          resample=Image.Resampling.BICUBIC, fillcolor=fill)


def before_after_sheet(img, quad, crop, ok=True, max_h=360):
    """Side-by-side picture for verifying a crop: the original with the
    outline that will be cut (green = confident, orange = please check),
    then the cropped result at the same height."""
    from PIL import ImageDraw
    color = (60, 220, 120) if ok else (245, 170, 50)
    s = max_h / img.height
    left = img.convert("RGB").resize((max(1, int(img.width * s)), max_h))
    if quad is not None:
        pts = [(float(x) * s, float(y) * s) for x, y in quad]
        ImageDraw.Draw(left).line(pts + [pts[0]], fill=color, width=4)
    right = crop.convert("RGB").resize(
        (max(1, int(crop.width * max_h / max(1, crop.height))), max_h))
    sheet = Image.new("RGB", (left.width + right.width + 30, max_h + 8),
                      (24, 27, 34))
    sheet.paste(left, (4, 4))
    sheet.paste(right, (left.width + 26, 4))
    ImageDraw.Draw(sheet).rectangle(
        [left.width + 10, 4, left.width + 14, max_h + 4], fill=color)
    return sheet
