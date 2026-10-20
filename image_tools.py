#!/usr/bin/env python3
"""Local, free image intelligence for the file toolkit: perceptual-hash
similarity grouping, automatic document cropping, and a self-learning loop.

Everything runs offline — no cloud, no LLM, no downloaded models. The
"learning" is feedback-driven: your answers about groups and crops are kept
in the toolkit database and gently adjust the parameters used next time.

Imported lazily by file_toolkit.py, so it reaches back into it for the
shared config, prompts and helpers.
"""

import hashlib
import json
import os
import threading
import re
import shutil
import sqlite3
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import imagehash
import numpy as np
from PIL import Image, ImageOps

import doc_crop
import file_toolkit as tk
import learned_crop as locator


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}

# Image types the user can teach the toolkit (Arabic labels for the GUI).
TYPE_LABELS = {
    "income": "ورقة الدخل",
    "invoice": "فاتورة",
    "receipt": "إيصال",
    "other": "أخرى",
}

# Learnable parameters: defaults, and the range the feedback loop may move
# them into. Keys live in the learning_state table; anything absent simply
# uses the default.
DEFAULT_PARAMS = {
    "similarity_threshold": 10,   # max pHash hamming distance (/64) = similar
    "crop_margin_pct": 1.0,       # padding kept around the paper, in % of the
                                  # photo's short side (size-independent)
    "crop_tolerance": 28,         # sensitivity of the colour-difference test
    "crop_deskew": 0,             # 1 = also straighten tilted papers
    "crop_min_gain": 5.0,         # skip cropping if it frees less than this %
}
PARAM_BOUNDS = {
    "similarity_threshold": (4, 20),
    "crop_margin_pct": (0.0, 8.0),
    "crop_deskew": (0, 1),
    "crop_tolerance": (5, 90),
    "crop_min_gain": (1.0, 30.0),
}


# ==========================================================================
# 1) Database: hash cache + learning state
# ==========================================================================
def _open_db():
    conn = sqlite3.connect(tk.DB_NAME)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS image_hashes (
            full_path TEXT PRIMARY KEY,
            mtime REAL NOT NULL,
            phash TEXT NOT NULL,
            dhash TEXT NOT NULL
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS learning_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS feedback_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            kind TEXT NOT NULL,
            detail TEXT NOT NULL,
            old_value TEXT,
            new_value TEXT
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS operation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            run_id TEXT NOT NULL,
            op TEXT NOT NULL,
            source TEXT,
            destination TEXT,
            detail TEXT,
            status TEXT NOT NULL DEFAULT 'approved'
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS crop_examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            source_path TEXT NOT NULL,
            img_w INTEGER NOT NULL,
            img_h INTEGER NOT NULL,
            type TEXT NOT NULL,
            box_l REAL NOT NULL,
            box_t REAL NOT NULL,
            box_r REAL NOT NULL,
            box_b REAL NOT NULL,
            low_quality INTEGER NOT NULL DEFAULT 0
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS crop_model (
            id INTEGER PRIMARY KEY,
            signature TEXT NOT NULL,
            n INTEGER NOT NULL,
            cv_iou REAL,
            blob BLOB NOT NULL
        )
    ''')
    # Older databases predate the status column — add it in place.
    columns = {row[1] for row in conn.execute(
        "PRAGMA table_info(operation_log)")}
    if "status" not in columns:
        conn.execute("ALTER TABLE operation_log ADD COLUMN "
                     "status TEXT NOT NULL DEFAULT 'approved'")
    # What the detector found for each taught photo (relative units), so
    # the user's correction can be learned independent of photo size.
    columns = {row[1] for row in conn.execute(
        "PRAGMA table_info(crop_examples)")}
    for col in ("det_l", "det_t", "det_r", "det_b", "det_conf"):
        if col not in columns:
            conn.execute(f"ALTER TABLE crop_examples ADD COLUMN {col} REAL")
    conn.commit()
    return conn


def _new_run_id():
    """Short id tying together every operation of one menu action, so a
    whole batch can be undone in one go."""
    return uuid.uuid4().hex[:12]


def _clamp(key, value):
    low, high = PARAM_BOUNDS[key]
    return max(low, min(high, value))


def get_param(conn, key):
    """Learned value for `key`, or the built-in default."""
    default = DEFAULT_PARAMS[key]
    row = conn.execute(
        "SELECT value FROM learning_state WHERE key = ?", (key,)
    ).fetchone()
    if row is None:
        return default
    try:
        return type(default)(row[0])
    except (TypeError, ValueError):
        return default


def set_param(conn, key, new_value, kind, detail):
    """Persist a learned parameter and record why it changed."""
    old_value = get_param(conn, key)
    if new_value == old_value:
        return old_value
    conn.execute(
        "INSERT OR REPLACE INTO learning_state (key, value) VALUES (?, ?)",
        (key, str(new_value)),
    )
    conn.execute(
        "INSERT INTO feedback_log (ts, kind, detail, old_value, new_value) "
        "VALUES (?, ?, ?, ?, ?)",
        (datetime.now().isoformat(timespec="seconds"), kind, detail,
         str(old_value), str(new_value)),
    )
    conn.commit()
    print(f"  [Learned] {key}: {old_value} -> {new_value} ({detail})")
    return old_value


def reset_learning(conn):
    """Forget every learned parameter (the feedback history is kept)."""
    conn.execute("DELETE FROM learning_state")
    conn.commit()


# ==========================================================================
# 2) Image collection + perceptual hashing (cached)
# ==========================================================================
def collect_image_files(target_dir, recursive=True):
    """Every image under `target_dir`, skipping hidden files, the toolkit's
    own output folders and — when not recursive — subfolders entirely."""
    target_dir = Path(target_dir)
    files = []
    if recursive:
        for root, dirs, names in os.walk(target_dir):
            dirs[:] = [d for d in dirs
                       if not d.startswith(".") and d not in tk.OUTPUT_DIR_NAMES]
            for name in names:
                if name.startswith("."):
                    continue
                if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                    files.append(Path(root) / name)
    else:
        for entry in target_dir.iterdir():
            if (entry.is_file() and not entry.name.startswith(".")
                    and entry.suffix.lower() in IMAGE_EXTENSIONS):
                files.append(entry)
    return sorted(files)


def _compute_hashes(path):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        return imagehash.phash(img), imagehash.dhash(img)


def hashes_for_files(conn, files):
    """Map each usable file path to (phash, dhash), recomputing only images
    whose mtime changed since the last run."""
    cache = {
        row[0]: (row[1], row[2], row[3])
        for row in conn.execute(
            "SELECT full_path, mtime, phash, dhash FROM image_hashes")
    }
    hashes = {}
    fresh_rows = []
    for path in files:
        try:
            mtime = path.stat().st_mtime
        except OSError as e:
            print(f"  [Skipped] {path.name}: {e}")
            continue
        cached = cache.get(str(path))
        if cached and abs(cached[0] - mtime) < 1e-6:
            hashes[str(path)] = (
                imagehash.hex_to_hash(cached[1]),
                imagehash.hex_to_hash(cached[2]),
            )
            continue
        try:
            phash, dhash = _compute_hashes(path)
        except Exception as e:
            print(f"  [Skipped] {path.name}: unreadable image ({e})")
            continue
        hashes[str(path)] = (phash, dhash)
        fresh_rows.append((str(path), mtime, str(phash), str(dhash)))

    if fresh_rows:
        conn.executemany(
            "INSERT OR REPLACE INTO image_hashes "
            "(full_path, mtime, phash, dhash) VALUES (?, ?, ?, ?)",
            fresh_rows,
        )
        # Files renamed or moved outside this module leave stale rows
        # behind (e.g. via the Quick Renamer GUI) — drop them.
        stale = [path for path in cache if not os.path.exists(path)]
        for path in stale:
            conn.execute("DELETE FROM image_hashes WHERE full_path = ?",
                         (path,))
        conn.commit()
    return hashes


# ==========================================================================
# 3) Similar-image clustering
# ==========================================================================
def _split_chains(group, hashes, threshold):
    """Union-find can chain A~B~C even when A and C look nothing alike, which
    matters for screenshots sharing one layout. Split a chained group
    greedily so every member ends up within `threshold` of ALL other members
    of its sub-cluster (complete-link criterion)."""
    if len(group) == 2:
        return [list(group)]
    clusters = []
    for path in group:
        for cluster in clusters:
            if all(hashes[path][0] - hashes[other][0] <= threshold
                   for other in cluster):
                cluster.append(path)
                break
        else:
            clusters.append([path])
    return clusters


def cluster_by_similarity(hashes, threshold):
    """Group paths whose pHash hamming distance is <= `threshold` (union-find,
    then chained groups are split so every pair inside a cluster is truly
    similar). Returns a list of (members, max_pair_distance) tuples sorted by
    first member path."""
    paths = list(hashes)
    parent = {p: p for p in paths}

    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            a, b = paths[i], paths[j]
            if hashes[a][0] - hashes[b][0] <= threshold:
                parent[find(a)] = find(b)

    groups = defaultdict(list)
    for p in paths:
        groups[find(p)].append(p)

    result = []
    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort()
        for cluster in _split_chains(members, hashes, threshold):
            if len(cluster) < 2:
                continue
            spread = max(
                hashes[a][0] - hashes[b][0]
                for idx, a in enumerate(cluster)
                for b in cluster[idx + 1:]
            )
            result.append((cluster, spread))
    result.sort(key=lambda item: item[0][0])
    return result


def _adjust_threshold(conn, confirmed, spread, detail):
    """One feedback step for the similarity threshold.

    A rejected group means the threshold let too-distant images together, so
    it drops to just under that group's spread. A confirmed group that sat
    right at the threshold nudges it up — borderline pairs were judged
    genuinely similar, so slightly more distant ones may be worth catching."""
    key = "similarity_threshold"
    threshold = get_param(conn, key)
    low, high = PARAM_BOUNDS[key]
    if confirmed:
        if spread < threshold - 2:
            return                      # comfortably inside; nothing to learn
        new_value = min(high, threshold + 1)
    else:
        new_value = min(threshold - 1, max(low, spread - 2))
        new_value = max(low, new_value)
    set_param(conn, key, new_value,
              "similar_yes" if confirmed else "similar_no", detail)


def _parse_selection(text, maximum):
    """Parse a group selection like '1,3-5' or 'a' into a sorted list of
    group numbers. Raises ValueError on anything else."""
    text = text.strip().lower()
    if text in ("a", "all"):
        return list(range(1, maximum + 1))
    selected = set()
    for token in re.split(r"[,\s]+", text):
        if not token:
            continue
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", token)
        if match is None:
            raise ValueError(token)
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start > end or start < 1 or end > maximum:
            raise ValueError(token)
        selected.update(range(start, end + 1))
    if not selected:
        raise ValueError("nothing selected")
    return sorted(selected)


def _next_group_number(review_root):
    """Smallest free suffix for group_XX folders already in the review dir,
    so planned destinations can be shown before anything moves."""
    numbers = [0]
    if review_root.is_dir():
        for path in review_root.glob("group_*"):
            match = re.fullmatch(r"group_(\d+)", path.name)
            if match and path.is_dir():
                numbers.append(int(match.group(1)))
    return max(numbers) + 1


def group_similar_images():
    """Menu action: find clusters of perceptually similar images, show the
    complete move plan (every source and its destination) and move only the
    groups the user picks. Every move is written to the operation log so the
    batch can be undone."""
    print("Finds groups of similar images via perceptual hashing (offline).")
    print("Nothing moves until you pick the groups, and every move is")
    print("logged so it can be undone.\n")

    conn = _open_db()
    try:
        threshold = get_param(conn, "similarity_threshold")
        files = collect_image_files(tk.TARGET_DIR)
        if not files:
            print("No images found in the working folder.\n")
            return

        print(f"Hashing {len(files)} image(s); cached hashes are reused...")
        hashes = hashes_for_files(conn, files)
        groups = cluster_by_similarity(hashes, threshold)
        if not groups:
            print(f"No similar images found (threshold {threshold}).\n")
            return

        review_root = Path(tk.SIMILAR_REVIEW_DIR)
        first_number = _next_group_number(review_root)
        plan = [
            (number, members, spread,
             review_root / f"group_{first_number + number - 1:02d}")
            for number, (members, spread) in enumerate(groups, 1)
        ]

        print(f"\nFound {len(plan)} group(s) of similar images "
              f"(threshold {threshold}). Move plan:\n")
        for number, members, spread, dest_dir in plan:
            print(f"Group {number} (spread {spread}) -> {dest_dir}/")
            for member in members:
                print(f"    {member}")
        print()

        answer = tk.ask(
            "Move which groups? (e.g. 1,3-5 | a = all | Enter = cancel): ")
        if answer is None or not answer.strip():
            print("  [Cancelled] Nothing was moved.\n")
            return
        try:
            selected = _parse_selection(answer, len(plan))
        except ValueError as e:
            print(f"  [Cancelled] Invalid selection: {e}\n")
            return
        chosen = [entry for entry in plan if entry[0] in selected]

        run_id = _new_run_id()
        moved_total = 0
        for number, members, spread, dest_dir in chosen:
            dest_dir.mkdir(parents=True, exist_ok=True)
            moved = 0
            for member in members:
                try:
                    destination = tk.unique_destination(
                        dest_dir, Path(member).name)
                    shutil.move(member, str(destination))
                    conn.execute(
                        "INSERT INTO operation_log "
                        "(ts, run_id, op, source, destination, detail) "
                        "VALUES (?, ?, 'move', ?, ?, ?)",
                        (datetime.now().isoformat(timespec="seconds"),
                         run_id, member, str(destination),
                         f"similar group {number}"))
                    conn.execute(
                        "DELETE FROM image_hashes WHERE full_path = ?",
                        (member,))
                    moved += 1
                except Exception as e:
                    print(f"  [Error] Failed to move {member}: {e}")
            conn.commit()
            moved_total += moved
            print(f"  Group {number}: moved {moved} image(s) to {dest_dir}")
            _adjust_threshold(conn, confirmed=True, spread=spread,
                              detail="group moved to review")

        answer = tk.ask(
            "\nMark any untouched group as NOT similar? Teaches the "
            "threshold (e.g. 2,4 | Enter = none): ")
        if answer and answer.strip():
            try:
                rejected = _parse_selection(answer, len(plan))
            except ValueError as e:
                print(f"  [Skipped] Invalid selection: {e}")
                rejected = []
            for number, _, spread, _ in plan:
                if number in rejected and number not in selected:
                    _adjust_threshold(conn, confirmed=False, spread=spread,
                                      detail="group marked not similar")

        print(f"\nDone! {moved_total} image(s) moved in batch {run_id} —")
        print("undo it with menu option 12 if the groups look wrong.\n")
    finally:
        conn.close()


# ==========================================================================
# 4) Automatic document cropping
#
# Phone photos never share exact pixel dimensions or paper placement, so
# nothing here is keyed to "image is WxH". The pipeline is:
#   detect the paper (doc_crop)  ->  add the margin / learned correction
#   ->  confidence + warnings    ->  before/after for the human to confirm
# What the user teaches is stored in *relative* units (fractions of the
# photo), so a lesson from one photo helps every other photo.
# ==========================================================================
CONF_OK = doc_crop.CONF_OK
CONF_MIN = doc_crop.CONF_MIN

WARNING_LABELS = {
    "low_confidence": "ثقة منخفضة — راجع القص",
    "small_document": "الورقة صغيرة في الصورة",
    "full_frame": "الورقة تملأ الصورة (لا حاجة للقص)",
    "touches_edge": "الورقة قريبة من حافة الصورة / قد تكون مقطوعة",
    "tilted": "الورقة مائلة",
    "weak_prior": "اعتمد على مثال معلَّم فقط (لم تُكشف الورقة)",
    "prior_disagree": "القص المعلَّم يخالف الكشف التلقائي",
    "nothing_found": "لم يُعثر على ورقة",
    "too_small": "الصورة صغيرة جداً",
}


@dataclass
class CropPlan:
    box: Optional[tuple] = None        # axis-aligned (l, t, r, b) in pixels
    quad: Optional[list] = None        # outline that is cut (TL,TR,BR,BL)
    deskew: bool = False               # quad is rotated and gets straightened
    detect_quad: Optional[list] = None  # raw detection (for the overlay)
    conf: float = 0.0
    method: str = "none"               # auto | calibrated | taught | prior
    type: Optional[str] = None
    warnings: list = field(default_factory=list)
    angle: float = 0.0
    bg: tuple = (255, 255, 255)
    size: tuple = (0, 0)               # source photo size
    reason: Optional[str] = None       # why nothing will be cropped

    @property
    def out_size(self):
        if self.quad is None:
            return None
        if self.deskew:
            q = np.asarray(self.quad, dtype=float)
            return (int(round((np.linalg.norm(q[1] - q[0])
                               + np.linalg.norm(q[2] - q[3])) / 2)),
                    int(round((np.linalg.norm(q[3] - q[0])
                               + np.linalg.norm(q[2] - q[1])) / 2)))
        l, t, r, b = self.box
        return (r - l, b - t)

    @property
    def gain(self):
        """Percent of the photo that cropping removes."""
        out = self.out_size
        if not out or not self.size[0]:
            return 0.0
        return max(0.0, (1 - out[0] * out[1] / (self.size[0] * self.size[1]))
                   * 100)

    @property
    def needs_review(self):
        return self.conf < CONF_OK or "weak_prior" in self.warnings \
            or "prior_disagree" in self.warnings

    def detail_json(self):
        def rnd(q):
            return None if q is None else [[round(float(x), 1),
                                            round(float(y), 1)] for x, y in q]
        return json.dumps({
            "method": self.method, "type": self.type or "unknown",
            "conf": round(self.conf, 3), "warnings": self.warnings,
            "angle": round(self.angle, 2), "deskew": self.deskew,
            "src": list(self.size), "out": list(self.out_size or (0, 0)),
            "quad": rnd(self.quad), "detect_quad": rnd(self.detect_quad),
            "box": list(self.box) if self.box else None,
        }, ensure_ascii=False)


def parse_detail(detail):
    """Operation detail as a dict (older databases stored plain text)."""
    try:
        data = json.loads(detail)
        if isinstance(data, dict):
            return data
    except (TypeError, ValueError):
        pass
    return {"method": "legacy", "note": detail or "", "warnings": [],
            "conf": None}


# -- what the user taught, in relative units --------------------------------
def _norm_box(row):
    """(l,t,r,b) of a crop_examples row as fractions of its photo."""
    return (row["box_l"], row["box_t"], row["box_r"], row["box_b"])


def _example_rows(conn):
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("SELECT * FROM crop_examples "
                            "WHERE low_quality = 0").fetchall()
    finally:
        conn.row_factory = None


def _same_shape(w, h, ew, eh, tol=0.03):
    """Same orientation and aspect ratio within `tol` (log scale). Photo
    *size* is irrelevant: a 4000x3000 and a 1600x1200 shot match."""
    if (w >= h) != (ew >= eh):
        return False
    return abs(np.log((w / h) / (ew / eh))) < tol


def taught_prior(conn, w, h):
    """Average taught box (as fractions) among examples with this photo's
    shape. 'strong' only when 3+ examples agree closely — otherwise it is
    only a fallback for photos where no paper can be detected."""
    if conn is None:
        return None
    rows = [r for r in _example_rows(conn)
            if _same_shape(w, h, r["img_w"], r["img_h"])]
    if not rows:
        return None
    boxes = np.array([_norm_box(r) for r in rows], dtype=float)
    spread = float(boxes.std(axis=0).max()) if len(rows) > 1 else 1.0
    types = [r["type"] for r in rows]
    mean = boxes.mean(axis=0)
    return {"box": (int(round(mean[0] * w)), int(round(mean[1] * h)),
                    int(round(mean[2] * w)), int(round(mean[3] * h))),
            "n": len(rows), "spread": spread,
            "strong": len(rows) >= 3 and spread < 0.012,
            "type": max(set(types), key=types.count)}


def learned_offsets(conn):
    """How far outside (+) or inside (-) the detected paper the user's
    boxes usually are, per side, as a fraction of the photo's short side.
    Needs 2+ examples whose detection was confident."""
    if conn is None:
        return None
    vals = []
    for r in _example_rows(conn):
        if r["det_conf"] is None or r["det_conf"] < CONF_OK:
            continue
        w, h = r["img_w"], r["img_h"]
        side = min(w, h)
        vals.append(((r["det_l"] - r["box_l"]) * w / side,
                     (r["det_t"] - r["box_t"]) * h / side,
                     (r["box_r"] - r["det_r"]) * w / side,
                     (r["box_b"] - r["det_b"]) * h / side))
    if len(vals) < 2:
        return None
    med = np.clip(np.median(np.array(vals), axis=0), -0.06, 0.10)
    return tuple(float(v) for v in med)


def _iou(a, b):
    l, t = max(a[0], b[0]), max(a[1], b[1])
    r, bt = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, r - l) * max(0, bt - t)
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
             - inter)
    return inter / union if union else 0.0


def _box_quad(box):
    l, t, r, b = box
    return [[l, t], [r, t], [r, b], [l, b]]


def _fit_inside(quad, w, h, margins, side):
    """Expand a quad by `margins` (px), shrinking the growth until every
    corner is inside the photo."""
    for k in range(7):
        f = 0.5 ** k
        cand = doc_crop.expand_quad(quad, *[m * f for m in margins])
        if (cand[:, 0].min() >= -1 and cand[:, 1].min() >= -1
                and cand[:, 0].max() <= w + 1 and cand[:, 1].max() <= h + 1):
            return cand
    return np.asarray(quad, dtype=float)


# -- the learned invoice locator --------------------------------------------
LEARNED_OK = 0.60            # model confidence below this -> fall back
_MODEL_CACHE = {}
_FEAT_CACHE = {}
_MODEL_LOCK = threading.Lock()


def _taught_samples(conn):
    """[(path, box_fraction, type)] — the newest example per photo; photos
    that no longer exist are ignored."""
    latest = {}
    for r in _example_rows(conn):
        latest[r["source_path"]] = r
    out = []
    for path, r in latest.items():
        if os.path.exists(path):
            out.append((path, _norm_box(r), r["type"]))
    return sorted(out)


def _features_of_path(path):
    key = (path, os.path.getmtime(path))
    f = _FEAT_CACHE.get(key)
    if f is None:
        with Image.open(path) as im:
            f = locator.cell_features(ImageOps.exif_transpose(im))
        _FEAT_CACHE[key] = f
    return f


def _signature(samples):
    h = hashlib.md5(str(locator.MODEL_VERSION).encode())
    for path, box, _ in samples:
        h.update(f"{path}|{box[0]:.4f},{box[1]:.4f},{box[2]:.4f},"
                 f"{box[3]:.4f}|{os.path.getmtime(path):.0f}".encode())
    return h.hexdigest()


def get_model(conn):
    """The trained locator for the current examples (trained on demand and
    cached in the database), or None when fewer than 3 photos were taught."""
    samples = _taught_samples(conn)
    if len(samples) < locator.MIN_EXAMPLES:
        return None
    sig = _signature(samples)
    key = (tk.DB_NAME, sig)
    with _MODEL_LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        row = conn.execute("SELECT signature, blob FROM crop_model "
                           "WHERE id = 1").fetchone()
        model = None
        if row and row[0] == sig:
            model = locator.Model.from_bytes(row[1])
        if model is None:
            model = locator.train(
                [(_features_of_path(p), b) for p, b, _ in samples])
            old = conn.execute("SELECT cv_iou FROM crop_model WHERE id = 1"
                               ).fetchone()
            conn.execute("INSERT OR REPLACE INTO crop_model "
                         "(id, signature, n, cv_iou, blob) "
                         "VALUES (1, ?, ?, ?, ?)",
                         (sig, len(samples), None, model.to_bytes()))
            conn.commit()
        _MODEL_CACHE[key] = model
        return model


def model_info(conn):
    """{'n': taught photos, 'trained': bool, 'cv_iou': float|None}"""
    samples = _taught_samples(conn)
    row = conn.execute("SELECT signature, cv_iou FROM crop_model "
                       "WHERE id = 1").fetchone()
    fresh = bool(row) and len(samples) >= locator.MIN_EXAMPLES \
        and row[0] == _signature(samples)
    return {"n": len(samples),
            "needed": locator.MIN_EXAMPLES,
            "trained": fresh,
            "cv_iou": row[1] if fresh and row[1] is not None else None}


def evaluate_model(conn, folds=5):
    """Honest accuracy estimate: train on some of the taught photos and
    test on the others (k-fold). Returns {'mean_iou', 'per': [...]}"""
    samples = _taught_samples(conn)
    n = len(samples)
    if n < locator.MIN_EXAMPLES + 1:
        return None
    folds = max(2, min(folds, n))
    feats = [_features_of_path(p) for p, _, _ in samples]
    per = []
    for k in range(folds):
        test = [i for i in range(n) if i % folds == k]
        train_idx = [i for i in range(n) if i % folds != k]
        if len(train_idx) < locator.MIN_EXAMPLES:
            continue
        m = locator.train([(feats[i], samples[i][1]) for i in train_idx])
        for i in test:
            pb = locator.predict(m, None, feats[i]).box_frac
            per.append((os.path.basename(samples[i][0]),
                        _iou(pb, samples[i][1])))
    mean = float(np.mean([v for _, v in per])) if per else 0.0
    sig = _signature(samples)
    conn.execute("UPDATE crop_model SET cv_iou = ? WHERE id = 1 "
                 "AND signature = ?", (mean, sig))
    conn.commit()
    return {"mean_iou": mean, "per": per, "n": n}


def _learned_plan(conn, img, model):
    W, H = img.size
    pred = locator.predict(model, img)
    l, t, r, b = pred.box_frac
    box = (int(round(l * W)), int(round(t * H)),
           int(round(r * W)), int(round(b * H)))
    plan = CropPlan(size=(W, H), box=box, quad=_box_quad(box),
                    conf=pred.conf, method="learned")
    types = [t_ for _, _, t_ in _taught_samples(conn)]
    plan.type = max(set(types), key=types.count) if types else None
    if pred.conf < CONF_OK:
        plan.warnings.append("low_confidence")
    if box[0] <= 0 or box[1] <= 0 or box[2] >= W or box[3] >= H:
        plan.warnings.append("touches_edge")
    return plan


def plan_crop(conn, img, deskew=None):
    """Decide how to crop `img` (EXIF orientation already applied).
    `conn` may be None (defaults, no taught examples)."""
    W, H = img.size
    side = min(W, H)
    tolerance = get_param(conn, "crop_tolerance") if conn else \
        DEFAULT_PARAMS["crop_tolerance"]
    margin_pct = get_param(conn, "crop_margin_pct") if conn else \
        DEFAULT_PARAMS["crop_margin_pct"]
    if deskew is None:
        deskew = bool(get_param(conn, "crop_deskew") if conn
                      else DEFAULT_PARAMS["crop_deskew"])

    # 1) A model trained on the user's own examples beats any generic rule.
    model = get_model(conn) if conn is not None else None
    learned = None
    if model is not None:
        learned = _learned_plan(conn, img, model)
        if learned.conf >= LEARNED_OK:
            return learned

    det = doc_crop.detect_document(img, tolerance)
    plan = CropPlan(size=(W, H), warnings=list(det.warnings),
                    conf=det.conf, angle=det.angle, bg=det.bg,
                    detect_quad=None if not det.found
                    else [[float(x), float(y)] for x, y in det.quad])
    if learned is not None and det.conf < CONF_OK:
        return learned      # a weak generic guess never beats the model

    prior = taught_prior(conn, W, H)

    use_prior = None
    if prior and prior["strong"]:
        use_prior = "taught"
        if det.found and det.conf >= 0.75:
            db = doc_crop.quad_bounds(det.quad, W, H)
            if _iou(prior["box"], db) < 0.7:
                use_prior = None
                plan.warnings.append("prior_disagree")
    elif prior and (not det.found or det.conf < CONF_MIN):
        use_prior = "prior"

    if use_prior:
        plan.box = tuple(prior["box"])
        plan.quad = _box_quad(plan.box)
        plan.method = use_prior
        plan.type = prior["type"]
        plan.deskew = False
        if use_prior == "taught":
            plan.conf = 0.9
            plan.warnings = [w for w in plan.warnings
                             if w not in ("low_confidence", "nothing_found")]
        else:
            plan.conf = 0.45
            plan.warnings = [w for w in plan.warnings
                             if w != "nothing_found"] + ["weak_prior"]
        return plan

    if not det.found or det.conf < CONF_MIN:
        if learned is not None:
            return learned
        plan.reason = WARNING_LABELS.get(
            "nothing_found" if not det.found else "low_confidence")
        return plan

    offsets = learned_offsets(conn)
    if offsets:
        margins = [o * side for o in offsets]
        plan.method = "calibrated"
    else:
        margins = [margin_pct / 100.0 * side] * 4
        plan.method = "auto"

    if deskew and abs(det.angle) >= 0.7 and "full_frame" not in plan.warnings:
        q = _fit_inside(det.quad, W, H, margins, side)
        plan.quad = [[float(x), float(y)] for x, y in q]
        plan.deskew = True
        plan.box = doc_crop.quad_bounds(q, W, H)
    else:
        l, t, r, b = doc_crop.quad_bounds(det.quad, W, H)
        l = int(max(0, l - margins[0]))
        t = int(max(0, t - margins[1]))
        r = int(min(W, r + margins[2]))
        b = int(min(H, b + margins[3]))
        if r - l < 8 or b - t < 8:
            plan.reason = WARNING_LABELS["low_confidence"]
            return plan
        plan.box = (l, t, r, b)
        plan.quad = _box_quad(plan.box)
        plan.deskew = False
    return plan


def apply_plan(img, plan):
    """The cropped image described by `plan`."""
    if plan.deskew:
        return doc_crop.warp_quad(img, plan.quad, plan.bg)
    return img.crop(plan.box)


def before_after_image(img, plan, max_h=360):
    crop = apply_plan(img, plan)
    return doc_crop.before_after_sheet(
        img, plan.detect_quad if plan.deskew or plan.method in
        ("auto", "calibrated") else plan.quad, crop,
        ok=not plan.needs_review, max_h=max_h)


def _save_cropped(img, plan, destination):
    """Write the crop. `plan` is a CropPlan or a plain (l,t,r,b) box."""
    cropped = apply_plan(img, plan) if isinstance(plan, CropPlan) \
        else img.crop(plan)
    kwargs = {}
    if destination.suffix.lower() in (".jpg", ".jpeg"):
        cropped = cropped.convert("RGB")
        kwargs["quality"] = 95
    try:
        # the pixels are already rotated upright — keep the metadata but
        # drop the Orientation tag so viewers do not rotate it twice
        exif = img.getexif()
        if exif:
            exif.pop(0x0112, None)
            kwargs["exif"] = exif
    except Exception:
        pass
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        cropped.save(destination, **kwargs)
    except Exception:
        kwargs.pop("exif", None)
        cropped.save(destination, **kwargs)


def save_crop_example(conn, source_path, img, type_, box, low_quality=False):
    """Remember a box the user drew. Stored as fractions of the photo plus
    what the detector found for that photo, so the *difference* can be
    learned and re-applied to photos of any size."""
    if hasattr(img, "size"):
        w, h = img.size
        det = doc_crop.detect_document(img, get_param(conn, "crop_tolerance"))
        if det.found:
            dl, dt, dr, db = doc_crop.quad_bounds(det.quad, w, h)
            det_vals = (dl / w, dt / h, dr / w, db / h, det.conf)
        else:
            det_vals = (None, None, None, None, None)
    else:
        w, h = img
        det_vals = (None, None, None, None, None)
    left, top, right, bottom = box
    conn.execute(
        "INSERT INTO crop_examples (ts, source_path, img_w, img_h, type, "
        "box_l, box_t, box_r, box_b, low_quality, det_l, det_t, det_r, "
        "det_b, det_conf) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (datetime.now().isoformat(timespec="seconds"), str(source_path),
         w, h, type_, left / w, top / h, right / w, bottom / h,
         1 if low_quality else 0, *det_vals))
    conn.commit()


def learned_crop(conn, img):
    """Compatibility wrapper: (box, method, type) or (None, reason, None)."""
    plan = plan_crop(conn, img)
    if plan.box is None:
        return None, plan.reason or "none", None
    return plan.box, plan.method, plan.type


def _log_crop(conn, run_id, path, destination, plan, status):
    conn.execute(
        "INSERT INTO operation_log (ts, run_id, op, source, destination, "
        "detail, status) VALUES (?, ?, 'crop', ?, ?, ?, ?)",
        (datetime.now().isoformat(timespec="seconds"), run_id, str(path),
         str(destination), plan.detail_json(), status))


def _write_review_report(items, out_root):
    """HTML page with a before/after picture per image (worst first), so a
    whole batch can be verified at a glance."""
    import html
    review = Path(out_root) / "_review"
    review.mkdir(parents=True, exist_ok=True)
    items = sorted(items, key=lambda it: (not it["review"], it["conf"]))
    cards = []
    for i, it in enumerate(items):
        name = f"{i:04d}.jpg"
        it["sheet"].save(review / name, quality=88)
        warns = "، ".join(WARNING_LABELS.get(w, w) for w in it["warnings"])
        cls = "warn" if it["review"] else "ok"
        cards.append(
            f'<div class="card {cls}"><img src="{name}">'
            f'<div class="meta"><b>{html.escape(it["name"])}</b><br>'
            f'{it["src"][0]}×{it["src"][1]} ← → {it["out"][0]}×{it["out"][1]}'
            f' &nbsp;|&nbsp; الثقة {it["conf"] * 100:.0f}%'
            f'{"<br>⚠ " + html.escape(warns) if warns else ""}</div></div>')
    page = ('<!doctype html><html lang="ar" dir="rtl"><meta charset="utf-8">'
            '<title>مراجعة القص</title><style>'
            'body{font-family:system-ui,sans-serif;background:#12151b;'
            'color:#e8ecf2;margin:20px}'
            '.grid{display:grid;grid-template-columns:repeat(auto-fill,'
            'minmax(520px,1fr));gap:14px}'
            '.card{background:#1b2029;border-radius:10px;padding:10px;'
            'border:2px solid #2c3340}.card.warn{border-color:#f5aa32}'
            '.card.ok{border-color:#2c6e49}img{max-width:100%;border-radius:6px}'
            '.meta{font-size:13px;margin-top:6px;direction:ltr;text-align:left}'
            '</style><h2>مراجعة القص — اليسار: قبل (الإطار = حدود القص) · '
            'اليمين: بعد</h2><p>الإطار البرتقالي = يحتاج مراجعتك، '
            'الأخضر = موثوق.</p><div class="grid">' + "".join(cards)
            + "</div></html>")
    path = review / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def auto_crop_images():
    """Menu action: trim the background around every document photo. The
    originals are never touched — results go to the auto_cropped folder,
    with an HTML before/after report to verify them."""
    print("Cuts the background off document photos and saves the result in")
    print(f"'{tk.CROPPED_OUTPUT_DIR}' — original files are never modified.\n")

    answer = tk.ask("Scan subfolders too? (Y/n): ")
    if answer is None:
        print("  [Cancelled]\n")
        return
    recursive = answer.strip().lower() not in ("n", "no")
    answer = tk.ask("Straighten tilted papers too? (y/N): ")
    if answer is None:
        print("  [Cancelled]\n")
        return
    deskew = answer.strip().lower() in ("y", "yes")

    files = collect_image_files(tk.TARGET_DIR, recursive=recursive)
    if not files:
        print("No images found in the working folder.\n")
        return

    conn = _open_db()
    try:
        min_gain = get_param(conn, "crop_min_gain")
        print(f"Using learned settings: margin "
              f"{get_param(conn, 'crop_margin_pct')}% of the photo, "
              f"min gain {min_gain}%.\n")

        out_root = Path(tk.CROPPED_OUTPUT_DIR)
        run_id = _new_run_id()
        cropped = skipped = failed = 0
        items = []
        for path in files:
            try:
                rel = path.relative_to(tk.TARGET_DIR)
            except ValueError:
                rel = Path(path.name)
            try:
                with Image.open(path) as img:
                    img = ImageOps.exif_transpose(img)
                    plan = plan_crop(conn, img, deskew=deskew)
                    if plan.box is None:
                        print(f"  [Skipped] {rel}: {plan.reason}")
                        skipped += 1
                        continue
                    if plan.gain < min_gain:
                        print(f"  [Skipped] {rel}: already tight "
                              f"(cropping would free only {plan.gain:.1f}%)")
                        skipped += 1
                        continue
                    destination = tk.unique_destination(out_root / rel.parent,
                                                        rel.name)
                    _save_cropped(img, plan, destination)
                    _log_crop(conn, run_id, path, destination, plan,
                              "approved")
                    out = plan.out_size
                    flag = "  <-- CHECK" if plan.needs_review else ""
                    print(f"  {rel}: {img.width}x{img.height} -> "
                          f"{out[0]}x{out[1]} (freed {plan.gain:.0f}%, "
                          f"confidence {plan.conf * 100:.0f}%){flag}")
                    items.append({
                        "name": str(rel), "src": plan.size, "out": out,
                        "conf": plan.conf, "warnings": plan.warnings,
                        "review": plan.needs_review,
                        "sheet": before_after_image(img, plan)})
                    cropped += 1
            except Exception as e:
                print(f"  [Error] Failed to crop {rel}: {e}")
                failed += 1

        conn.commit()
        flagged = sum(1 for it in items if it["review"])
        print(f"\nDone! {cropped} image(s) cropped, {skipped} skipped, "
              f"{failed} failed. Results are in '{tk.CROPPED_OUTPUT_DIR}' "
              f"(batch {run_id}).")
        if items:
            report = _write_review_report(items, out_root)
            print(f"Before/after report: {report}")
            if flagged:
                print(f"  {flagged} image(s) are marked CHECK — they appear "
                      f"first in the report.")
            try:
                import webbrowser
                webbrowser.open(report.resolve().as_uri())
            except Exception:
                pass
        print("Review them, then use the crop feedback option to teach the")
        print("cropper what 'good' looks like — or option 12 to undo the")
        print("whole batch.\n")
    finally:
        conn.close()


def run_autonomous_crop(deskew=None):
    """Crop every image in the working folder with the learned/detected
    plan. Results are `pending` until the user approves them (with a
    before/after view in the GUI). Returns (run_id, created, skipped)."""
    files = collect_image_files(tk.TARGET_DIR, recursive=True)
    conn = _open_db()
    try:
        out_root = Path(tk.CROPPED_OUTPUT_DIR)
        min_gain = get_param(conn, "crop_min_gain")
        run_id = _new_run_id()
        created = 0
        skipped = []
        for path in files:
            try:
                rel = path.relative_to(tk.TARGET_DIR)
            except ValueError:
                rel = Path(path.name)
            try:
                with Image.open(path) as img:
                    img = ImageOps.exif_transpose(img)
                    plan = plan_crop(conn, img, deskew=deskew)
                    if plan.box is None:
                        skipped.append((str(rel), plan.reason))
                        continue
                    if plan.gain < min_gain:
                        skipped.append((str(rel), "لا حاجة للقص "
                                        f"(يوفّر {plan.gain:.1f}% فقط)"))
                        continue
                    destination = tk.unique_destination(
                        out_root / rel.parent, rel.name)
                    _save_cropped(img, plan, destination)
                    _log_crop(conn, run_id, path, destination, plan,
                              "pending")
                    created += 1
            except Exception as e:
                skipped.append((str(rel), f"خطأ: {e}"))
        conn.commit()
        return run_id, created, skipped
    finally:
        conn.close()


def pending_operations(conn):
    """Operations recorded but not yet approved or rejected by the user."""
    return conn.execute(
        "SELECT id, ts, op, source, destination, detail FROM operation_log "
        "WHERE status = 'pending' ORDER BY id").fetchall()


def review_operations(conn, approve_ids, reject_ids):
    """Approve or reject pending operations; rejecting a crop deletes the
    generated copy. Returns the number of rejected copies removed."""
    removed = 0
    now = datetime.now().isoformat(timespec="seconds")
    for op_id in reject_ids:
        row = conn.execute(
            "SELECT op, destination, detail FROM operation_log "
            "WHERE id = ?", (op_id,)).fetchone()
        if row is not None and row[0] == "crop":
            if row[1] and os.path.exists(row[1]):
                os.remove(row[1])
                removed += 1
            conn.execute("INSERT INTO feedback_log (ts, kind, detail) "
                         "VALUES (?, 'crop_rejected', ?)", (now, row[2]))
        conn.execute("UPDATE operation_log SET status = 'rejected' "
                     "WHERE id = ?", (op_id,))
    for op_id in approve_ids:
        row = conn.execute(
            "SELECT op, detail FROM operation_log WHERE id = ?",
            (op_id,)).fetchone()
        if row is not None and row[0] == "crop":
            conn.execute("INSERT INTO feedback_log (ts, kind, detail) "
                         "VALUES (?, 'crop_approved', ?)", (now, row[1]))
        conn.execute("UPDATE operation_log SET status = 'approved' "
                     "WHERE id = ?", (op_id,))
    conn.commit()
    return removed


def crop_feedback():
    """Menu action: teach the cropper from what you saw in auto_cropped/."""
    print("Answer based on the results you reviewed in "
          f"'{tk.CROPPED_OUTPUT_DIR}':")
    print(" 1) Crops cut off some content (too tight)")
    print(" 2) Crops left too much background (too loose)")
    print(" 3) Crops look good")
    answer = tk.ask("Select 1/2/3: ")
    if answer is None:
        print("  [Cancelled]\n")
        return
    answer = answer.strip()

    conn = _open_db()
    try:
        if answer == "1":
            margin = _clamp("crop_margin_pct",
                            get_param(conn, "crop_margin_pct") + 0.5)
            set_param(conn, "crop_margin_pct", round(margin, 2),
                      "crop_too_tight", "crops were cutting off content")
        elif answer == "2":
            margin = _clamp("crop_margin_pct",
                            get_param(conn, "crop_margin_pct") - 0.5)
            set_param(conn, "crop_margin_pct", round(margin, 2),
                      "crop_too_loose", "crops left background in")
        elif answer == "3":
            conn.execute(
                "INSERT INTO feedback_log (ts, kind, detail) "
                "VALUES (?, 'crop_good', 'user approved the crops')",
                (datetime.now().isoformat(timespec="seconds"),),
            )
            conn.commit()
            print("  [Learned] Thanks — settings stay as they are.")
        else:
            print("  [Cancelled] Invalid choice.\n")
            return
        print()
    finally:
        conn.close()


# ==========================================================================
# 5) Undo the last move/crop batch
# ==========================================================================
def undo_last_operation(ask_fn=None):
    """Revert the most recent move/crop batch from the operation log — moved
    images go back to their original paths and cropped copies are deleted.
    `ask_fn` overrides the confirmation prompt (GUIs pass their own); returns
    (reverted, failed)."""
    ask_fn = ask_fn or tk.ask
    conn = _open_db()
    try:
        last = conn.execute(
            "SELECT run_id FROM operation_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if last is None:
            print("Nothing to undo — the operation log is empty.\n")
            return 0, 0
        run_id = last[0]
        rows = conn.execute(
            "SELECT id, op, source, destination, detail FROM operation_log "
            "WHERE run_id = ? ORDER BY id", (run_id,)).fetchall()

        counts = defaultdict(int)
        for _, op, _, _, _ in rows:
            counts[op] += 1
        summary = ", ".join(f"{count} {name} operation(s)"
                            for name, count in sorted(counts.items()))
        print(f"Last batch (run {run_id}): {summary}.")
        answer = ask_fn("Undo it? (Y/n): ")
        if answer is None or answer.strip().lower() in ("n", "no"):
            print("  [Cancelled]\n")
            return 0, 0

        reverted = failed = 0
        for row_id, op, source, destination, detail in rows:
            try:
                if op == "move":
                    if not os.path.exists(destination):
                        print(f"  [Skipped] {destination} is already gone")
                    else:
                        Path(source).parent.mkdir(parents=True,
                                                  exist_ok=True)
                        target = tk.unique_destination(
                            Path(source).parent, Path(source).name)
                        shutil.move(destination, str(target))
                        note = ("" if target == source else
                                f" (renamed to {target.name} — path taken)")
                        print(f"  Restored {destination} -> {source}{note}")
                elif op == "crop":
                    if os.path.exists(destination):
                        os.remove(destination)
                        print(f"  Removed cropped copy {destination}")
                    else:
                        print(f"  [Skipped] {destination} is already gone")
                conn.execute("DELETE FROM operation_log WHERE id = ?",
                             (row_id,))
                reverted += 1
            except Exception as e:
                print(f"  [Error] Could not undo {detail or op}: {e}")
                failed += 1
        conn.commit()
        print(f"\nDone! {reverted} operation(s) undone"
              + (f", {failed} failed" if failed else "") + ".\n")
        return reverted, failed
    finally:
        conn.close()


# ==========================================================================
# 6) Learning status
# ==========================================================================
def learning_status():
    """Menu action: show what the toolkit has learned and optionally reset."""
    conn = _open_db()
    try:
        print("Learned parameters (used value; * = still at default):")
        for key, default in DEFAULT_PARAMS.items():
            value = get_param(conn, key)
            marker = "" if value != default else " *"
            bounds = PARAM_BOUNDS[key]
            print(f"  {key:<22} {value}{marker}   (allowed {bounds[0]}..{bounds[1]})")

        n_ex = conn.execute("SELECT COUNT(*) FROM crop_examples").fetchone()[0]
        offs = learned_offsets(conn)
        print(f"\nTaught crop examples: {n_ex}")
        if offs:
            print("  Learned margin around the paper (% of short side): "
                  f"left {offs[0]*100:.1f}, top {offs[1]*100:.1f}, "
                  f"right {offs[2]*100:.1f}, bottom {offs[3]*100:.1f}")
        else:
            print("  (needs 2+ examples with a confident detection to "
                  "learn the margin)")

        counts = conn.execute(
            "SELECT kind, COUNT(*) FROM feedback_log GROUP BY kind "
            "ORDER BY kind").fetchall()
        total = sum(count for _, count in counts)
        print(f"\nFeedback recorded so far: {total} note(s).")
        for kind, count in counts:
            print(f"  {kind:<16} {count}")

        answer = tk.ask("\nReset all learned parameters to defaults? (y/N): ")
        if answer is None:
            return
        if answer.strip().lower() in ("y", "yes"):
            reset_learning(conn)
            print("Learned parameters reset — feedback history kept.\n")
        else:
            print("Kept as they are.\n")
    finally:
        conn.close()
