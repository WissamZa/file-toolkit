#!/usr/bin/env python3
"""Learns *what the invoice looks like* from the boxes the user drew.

Colour-based detection cannot find a white invoice lying on white marble
next to blank sheets, a calculator and a stapler. What does work is
learning from examples: every taught photo says "these cells are the
invoice, those are not". A small random-forest (extra-trees, pure numpy)
is trained on per-cell features — colour, brightness, texture, ink density,
blue-ink density, gradients, wider context, and where the cell sits in the
photo — and then, for a new photo, the best-scoring rectangle over the
probability map is the crop.

Nothing depends on the photo's pixel size: the photo is always resampled
to a grid of ~96 cells along its long side.

Public API
    cell_features(img)                       -> (gh, gw, F) float32
    train(samples, ...)                      -> Model   samples=[(img, box_frac)]
    predict(model, img)                      -> Prediction
    Model.to_bytes() / Model.from_bytes(b)
"""

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

import doc_crop

CELL = 6              # working pixels per grid cell
LONG_CELLS = 96       # cells along the photo's long side
MIN_EXAMPLES = 3      # fewer taught photos than this -> no model
MODEL_VERSION = 3


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def _grid_for(w, h):
    s = LONG_CELLS / max(w, h)
    return max(8, round(w * s)), max(8, round(h * s))


def _pool(a, gh, gw):
    return a.reshape(gh, CELL, gw, CELL).mean(axis=(1, 3))


def _cell_blur(f, r):
    """Mean over a (2r+1)^2 neighbourhood of cells (edge-replicated)."""
    k = 2 * r + 1
    return doc_crop._window_sum(f, k) / (k * k)


def cell_features(img):
    """Per-cell feature maps for `img` (EXIF orientation already applied)."""
    W, H = img.size
    gw, gh = _grid_for(W, H)
    work = img.convert("RGB").resize((gw * CELL, gh * CELL), Image.BOX)
    a = np.asarray(work, dtype=np.float32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    lum = 0.299 * R + 0.587 * G + 0.114 * B
    mx, mn = a.max(axis=2), a.min(axis=2)
    sat = (mx - mn) / (mx + 1.0)

    # ink = clearly darker than the surrounding paper; blue ink separately
    bg = doc_crop._window_sum(lum, 25) / 625.0
    ink = (lum < bg - 30).astype(np.float32)
    blue = ((B > R + 20) & (B > G + 8) & (lum < 215)).astype(np.float32)
    gx = np.abs(np.diff(lum, axis=1, prepend=lum[:, :1]))
    gy = np.abs(np.diff(lum, axis=0, prepend=lum[:1]))

    m_lum = _pool(lum, gh, gw)
    base = [
        _pool(R, gh, gw), _pool(G, gh, gw), _pool(B, gh, gw), m_lum,
        np.sqrt(np.maximum(_pool(lum * lum, gh, gw) - m_lum ** 2, 0)),
        _pool(sat, gh, gw), _pool(ink, gh, gw), _pool(blue, gh, gw),
        _pool(gx, gh, gw), _pool(gy, gh, gw),
        _pool(np.minimum(lum, 255), gh, gw) - _cell_blur(m_lum, 6),
    ]
    maps = []
    for f in base:
        maps.append(f)
        maps.append(_cell_blur(f, 2))
        maps.append(_cell_blur(f, 6))
    # brightness / saturation / ink relative to the whole photo (lighting)
    for f in (m_lum, base[5], base[6]):
        maps.append(f - np.median(f))
    ys, xs = np.mgrid[0:gh, 0:gw]
    maps.append(xs / max(1, gw - 1))
    maps.append(ys / max(1, gh - 1))
    return np.stack(maps, axis=-1).astype(np.float32)


# --------------------------------------------------------------------------
# Extra-trees (numpy)
# --------------------------------------------------------------------------
class _Tree:
    __slots__ = ("feat", "thr", "left", "right", "val")

    def __init__(self, feat, thr, left, right, val):
        self.feat = np.asarray(feat, dtype=np.int32)
        self.thr = np.asarray(thr, dtype=np.float32)
        self.left = np.asarray(left, dtype=np.int32)
        self.right = np.asarray(right, dtype=np.int32)
        self.val = np.asarray(val, dtype=np.float32)

    def predict(self, X, depth):
        n = len(X)
        node = np.zeros(n, dtype=np.int32)
        rows = np.arange(n)
        for _ in range(depth + 1):
            f = self.feat[node]
            leaf = f < 0
            go_left = X[rows, np.maximum(f, 0)] <= self.thr[node]
            nxt = np.where(go_left, self.left[node], self.right[node])
            node = np.where(leaf, node, nxt)
        return self.val[node]


def _grow(X, y, rng, depth, min_leaf, mtry, tries=5):
    feat, thr, left, right, val = [], [], [], [], []

    def build(idx, d):
        node = len(feat)
        feat.append(-1); thr.append(0.0); left.append(-1); right.append(-1)
        p = float(y[idx].mean())
        val.append(p)
        if d >= depth or len(idx) < 2 * min_leaf or p < 0.01 or p > 0.99:
            return node
        Xi, yi = X[idx], y[idx]
        best = None
        for f in rng.choice(X.shape[1], size=mtry, replace=False):
            col = Xi[:, f]
            lo, hi = float(col.min()), float(col.max())
            if hi - lo < 1e-6:
                continue
            for t in rng.uniform(lo, hi, size=tries):
                m = col <= t
                nl = int(m.sum())
                nr = len(idx) - nl
                if nl < min_leaf or nr < min_leaf:
                    continue
                pl, pr = yi[m].mean(), yi[~m].mean()
                score = nl * pl * (1 - pl) + nr * pr * (1 - pr)
                if best is None or score < best[0]:
                    best = (score, int(f), float(t))
        if best is None:
            return node
        _, f, t = best
        m = X[idx, f] <= t
        feat[node], thr[node] = f, t
        left[node] = build(idx[m], d + 1)
        right[node] = build(idx[~m], d + 1)
        return node

    build(np.arange(len(X)), 0)
    return _Tree(feat, thr, left, right, val)


@dataclass
class Model:
    trees: list
    depth: int
    n_features: int
    n_examples: int
    cv_iou: float = -1.0              # filled in by evaluation

    def prob_map(self, feats):
        gh, gw, F = feats.shape
        X = feats.reshape(-1, F)
        p = np.mean([t.predict(X, self.depth) for t in self.trees], axis=0)
        return p.reshape(gh, gw)

    def to_bytes(self):
        buf = io.BytesIO()
        sizes = np.array([len(t.feat) for t in self.trees], dtype=np.int64)
        np.savez_compressed(
            buf, sizes=sizes,
            feat=np.concatenate([t.feat for t in self.trees]),
            thr=np.concatenate([t.thr for t in self.trees]),
            left=np.concatenate([t.left for t in self.trees]),
            right=np.concatenate([t.right for t in self.trees]),
            val=np.concatenate([t.val for t in self.trees]),
            meta=np.array([self.depth, self.n_features, self.n_examples,
                           MODEL_VERSION], dtype=np.int64),
            cv=np.array([self.cv_iou], dtype=np.float64))
        return buf.getvalue()

    @staticmethod
    def from_bytes(blob):
        z = np.load(io.BytesIO(blob))
        depth, nfeat, nex, ver = (int(v) for v in z["meta"])
        if ver != MODEL_VERSION:
            return None
        trees, o = [], 0
        for s in z["sizes"]:
            s = int(s)
            sl = slice(o, o + s)
            trees.append(_Tree(z["feat"][sl], z["thr"][sl], z["left"][sl],
                               z["right"][sl], z["val"][sl]))
            o += s
        return Model(trees, depth, nfeat, nex, float(z["cv"][0]))


def _cell_labels(box, gw, gh):
    """1 = cell lies inside the box, 0 = outside, -1 = straddles the edge."""
    l, t, r, b = box                       # fractions of the photo
    cx0 = np.arange(gw) / gw
    cx1 = (np.arange(gw) + 1) / gw
    cy0 = np.arange(gh) / gh
    cy1 = (np.arange(gh) + 1) / gh
    ox = np.clip(np.minimum(cx1, r) - np.maximum(cx0, l), 0, None) * gw
    oy = np.clip(np.minimum(cy1, b) - np.maximum(cy0, t), 0, None) * gh
    cov = oy[:, None] * ox[None, :]
    lab = np.full((gh, gw), -1, dtype=np.int8)
    lab[cov > 0.8] = 1
    lab[cov < 0.2] = 0
    return lab


def train(samples, n_trees=24, depth=8, min_leaf=20, per_tree=22000, seed=0,
          feats_cache=None):
    """samples: [(img_or_features, (l,t,r,b) as fractions)]"""
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for item, box in samples:
        f = item if isinstance(item, np.ndarray) else cell_features(item)
        gh, gw, F = f.shape
        lab = _cell_labels(box, gw, gh).reshape(-1)
        keep = lab >= 0
        Xs.append(f.reshape(-1, F)[keep])
        ys.append(lab[keep].astype(np.float32))
    X, y = np.concatenate(Xs), np.concatenate(ys)
    mtry = max(4, int(np.sqrt(X.shape[1])) + 2)
    trees = []
    for _ in range(n_trees):
        idx = rng.choice(len(X), size=min(per_tree, len(X)), replace=False)
        trees.append(_grow(X[idx], y[idx], rng, depth, min_leaf, mtry))
    return Model(trees, depth, X.shape[1], len(samples))


# --------------------------------------------------------------------------
# Prediction: best rectangle over the probability map
# --------------------------------------------------------------------------
@dataclass
class Prediction:
    box_frac: tuple            # (l, t, r, b) as fractions of the photo
    conf: float                # 0..1
    pmap: object               # (gh, gw) probabilities


def _best_rectangle(score):
    """Max-sum axis-aligned rectangle of a 2-D score map (cell indices,
    r/b exclusive)."""
    gh, gw = score.shape
    best, box = -1e18, (0, 0, 1, 1)
    for left in range(gw):
        col = np.zeros(gh)
        for right in range(left, gw):
            col += score[:, right]
            cs = np.concatenate(([0.0], np.cumsum(col)))
            minpref = np.minimum.accumulate(cs)
            gain = cs[1:] - minpref[:-1]
            j = int(np.argmax(gain))
            if gain[j] > best:
                i = int(np.argmin(cs[:j + 1]))
                best, box = float(gain[j]), (left, i, right + 1, j + 1)
    return box


def predict(model, img, feats=None):
    if feats is None:
        feats = cell_features(img)
    p = model.prob_map(feats)
    gh, gw = p.shape
    # smooth a little so isolated cells do not drag the rectangle around
    ps = (doc_crop._window_sum(p, 3) / 9.0)
    l, t, r, b = _best_rectangle(ps - 0.5)
    mask = np.zeros_like(p, dtype=bool)
    mask[t:b, l:r] = True
    high = p > 0.5
    inside_high = float(high[mask].mean()) if mask.any() else 0.0
    outside_share = float((high & ~mask).sum()) / max(1.0, float(high.sum()))
    conf = float(np.clip(inside_high * (1.0 - outside_share), 0.0, 1.0))
    return Prediction((l / gw, t / gh, r / gw, b / gh), conf, p)
