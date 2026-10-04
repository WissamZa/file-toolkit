#!/usr/bin/env python3
"""Local, free image intelligence for the file toolkit: perceptual-hash
similarity grouping, automatic document cropping, and a self-learning loop.

Everything runs offline — no cloud, no LLM, no downloaded models. The
"learning" is feedback-driven: your answers about groups and crops are kept
in the toolkit database and gently adjust the parameters used next time.

Imported lazily by file_toolkit.py, so it reaches back into it for the
shared config, prompts and helpers.
"""

import os
import re
import shutil
import sqlite3
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image, ImageOps

import file_toolkit as tk


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
    "crop_margin": 12,            # pixels of padding kept around the content
    "crop_tolerance": 28,         # how far a pixel may differ from the border
                                  # colour before it counts as content
    "crop_min_gain": 5.0,         # skip cropping if it frees less than this %
}
PARAM_BOUNDS = {
    "similarity_threshold": (4, 20),
    "crop_margin": (0, 120),
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
    # Older databases predate the status column — add it in place.
    columns = {row[1] for row in conn.execute(
        "PRAGMA table_info(operation_log)")}
    if "status" not in columns:
        conn.execute("ALTER TABLE operation_log ADD COLUMN "
                     "status TEXT NOT NULL DEFAULT 'approved'")
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
# ==========================================================================
def _save_cropped(img, box, destination):
    """Write the cropped region of `img` to `destination` (keeping format
    and EXIF where possible)."""
    cropped = img.crop(box)
    save_kwargs = {}
    if destination.suffix.lower() in (".jpg", ".jpeg"):
        save_kwargs = {"quality": 95}
        if "exif" in img.info:
            save_kwargs["exif"] = img.info["exif"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(destination, **save_kwargs)
    return cropped


def save_crop_example(conn, source_path, img_size, type_, box,
                      low_quality=False):
    """Store a user-taught crop as a normalized example the autonomous mode
    can apply to other images of the same dimensions."""
    width, height = img_size
    left, top, right, bottom = box
    conn.execute(
        "INSERT INTO crop_examples (ts, source_path, img_w, img_h, type, "
        "box_l, box_t, box_r, box_b, low_quality) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (datetime.now().isoformat(timespec="seconds"), str(source_path),
         width, height, type_, left / width, top / height,
         right / width, bottom / height, int(bool(low_quality))))
    conn.commit()


def learned_crop(conn, img):
    """Crop box from what the toolkit was taught, falling back to
    auto-detection. Returns (box or None, method, predicted_type) where
    method is 'taught' or 'auto'.

    Teaching works per image size: screenshots and scans usually share
    exact dimensions per source, so the average normalized box of every
    taught example at this size is the crop."""
    width, height = img.size
    row = conn.execute(
        "SELECT type, AVG(box_l), AVG(box_t), AVG(box_r), AVG(box_b) "
        "FROM crop_examples WHERE img_w = ? AND img_h = ? "
        "GROUP BY type ORDER BY COUNT(*) DESC LIMIT 1",
        (width, height)).fetchone()
    if row is not None:
        type_, nl, nt, nr, nb = row
        left = max(0, min(width - 1, round(nl * width)))
        top = max(0, min(height - 1, round(nt * height)))
        right = max(left + 1, min(width, round(nr * width)))
        bottom = max(top + 1, min(height, round(nb * height)))
        return (left, top, right, bottom), "taught", type_

    margin = get_param(conn, "crop_margin")
    tolerance = get_param(conn, "crop_tolerance")
    arr = np.asarray(img.convert("RGB"), dtype=np.int16)
    bbox = _content_bbox(arr, tolerance)
    if bbox is None:
        return None, "auto", None
    left, top, right, bottom = bbox
    left = max(0, left - margin)
    top = max(0, top - margin)
    right = min(width, right + margin)
    bottom = min(height, bottom + margin)
    return (left, top, right, bottom), "auto", None


def run_autonomous_crop():
    """Work mode: crop every image using the learned model and record each
    copy as a pending operation awaiting user approval. Returns
    (run_id, created, skipped)."""
    conn = _open_db()
    try:
        files = collect_image_files(tk.TARGET_DIR)
        out_root = Path(tk.CROPPED_OUTPUT_DIR)
        run_id = _new_run_id()
        min_gain = get_param(conn, "crop_min_gain")
        created = skipped = 0
        for path in files:
            try:
                rel = path.relative_to(tk.TARGET_DIR)
            except ValueError:
                rel = Path(path.name)
            try:
                with Image.open(path) as img:
                    img = ImageOps.exif_transpose(img)
                    box, method, type_ = learned_crop(conn, img)
                    if box is None:
                        print(f"  [Skipped] {rel}: nothing to crop")
                        skipped += 1
                        continue
                    left, top, right, bottom = box
                    old_area = img.width * img.height
                    gain = (old_area - (right - left) * (bottom - top)) \
                        / old_area * 100
                    if gain < min_gain:
                        skipped += 1
                        continue
                    destination = tk.unique_destination(
                        out_root / rel.parent, rel.name)
                    _save_cropped(img, box, destination)
                    conn.execute(
                        "INSERT INTO operation_log "
                        "(ts, run_id, op, source, destination, detail, "
                        " status) VALUES (?, ?, 'crop', ?, ?, ?, 'pending')",
                        (datetime.now().isoformat(timespec="seconds"),
                         run_id, str(path), str(destination),
                         f"method={method}, type={type_ or 'unknown'}"))
                    print(f"  [Pending approval] {rel} "
                          f"({method}, {TYPE_LABELS.get(type_, type_)})")
                    created += 1
            except Exception as e:
                print(f"  [Error] {rel}: {e}")
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
    for op_id in reject_ids:
        row = conn.execute(
            "SELECT op, destination FROM operation_log WHERE id = ?",
            (op_id,)).fetchone()
        if row is not None and row[0] == "crop" \
                and row[1] and os.path.exists(row[1]):
            os.remove(row[1])
            removed += 1
        conn.execute("UPDATE operation_log SET status = 'rejected' "
                     "WHERE id = ?", (op_id,))
    for op_id in approve_ids:
        conn.execute("UPDATE operation_log SET status = 'approved' "
                     "WHERE id = ?", (op_id,))
    conn.commit()
    return removed


def _content_bbox(arr, tolerance):
    """Bounding box (left, top, right, bottom) of everything that differs
    from the border colour, ignoring sparse noise rows/columns. Returns None
    when the image looks uniformly one colour."""
    height, width, _ = arr.shape
    frame = max(2, min(height, width) // 50)
    samples = np.concatenate([
        arr[:frame].reshape(-1, 3),
        arr[-frame:].reshape(-1, 3),
        arr[:, :frame].reshape(-1, 3),
        arr[:, -frame:].reshape(-1, 3),
    ])
    background = np.median(samples, axis=0)

    diff = np.abs(arr - background).max(axis=2)
    mask = diff > tolerance
    if not mask.any():
        return None

    # A row/column counts as content only if enough of it differs from the
    # background — filters dust specks and sensor noise.
    row_cut = max(3, int(width * 0.005))
    col_cut = max(3, int(height * 0.005))
    rows = np.where(mask.sum(axis=1) > row_cut)[0]
    cols = np.where(mask.sum(axis=0) > col_cut)[0]
    if rows.size == 0 or cols.size == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def compute_crop(img, margin, tolerance):
    """Crop box around the document in `img`, or (None, reason) when there
    is nothing worth cutting. `img` must already be orientation-corrected."""
    arr = np.asarray(img.convert("RGB"), dtype=np.int16)
    bbox = _content_bbox(arr, tolerance)
    if bbox is None:
        return None, "image looks uniformly one colour"
    left, top, right, bottom = bbox
    left = max(0, left - margin)
    top = max(0, top - margin)
    right = min(img.width, right + margin)
    bottom = min(img.height, bottom + margin)
    return (left, top, right, bottom), None


def auto_crop_images():
    """Menu action: trim the background around every document photo. The
    originals are never touched — results go to the auto_cropped folder."""
    print("Cuts the background off document photos and saves the result in")
    print(f"'{tk.CROPPED_OUTPUT_DIR}' — original files are never modified.\n")

    answer = tk.ask("Scan subfolders too? (Y/n): ")
    if answer is None:
        print("  [Cancelled]\n")
        return
    recursive = answer.strip().lower() not in ("n", "no")

    files = collect_image_files(tk.TARGET_DIR, recursive=recursive)
    if not files:
        print("No images found in the working folder.\n")
        return

    conn = _open_db()
    try:
        margin = get_param(conn, "crop_margin")
        tolerance = get_param(conn, "crop_tolerance")
        min_gain = get_param(conn, "crop_min_gain")
        print(f"Using learned settings: margin {margin}px, "
              f"tolerance {tolerance}, min gain {min_gain}%.\n")

        out_root = Path(tk.CROPPED_OUTPUT_DIR)
        run_id = _new_run_id()
        cropped = skipped = failed = 0
        for path in files:
            try:
                rel = path.relative_to(tk.TARGET_DIR)
            except ValueError:
                rel = Path(path.name)
            destination = tk.unique_destination(out_root / rel.parent,
                                                rel.name)
            try:
                with Image.open(path) as img:
                    img = ImageOps.exif_transpose(img)
                    box, reason = compute_crop(img, margin, tolerance)
                    if box is None:
                        print(f"  [Skipped] {rel}: {reason}")
                        skipped += 1
                        continue
                    left, top, right, bottom = box
                    old_area = img.width * img.height
                    new_area = (right - left) * (bottom - top)
                    gain = (old_area - new_area) / old_area * 100
                    if gain < min_gain:
                        print(f"  [Skipped] {rel}: already tight "
                              f"(cropping would free only {gain:.1f}%)")
                        skipped += 1
                        continue

                    _save_cropped(img, box, destination)
                    conn.execute(
                        "INSERT INTO operation_log "
                        "(ts, run_id, op, source, destination, "
                        " detail, status) "
                        "VALUES (?, ?, 'crop', ?, ?, ?, 'approved')",
                        (datetime.now().isoformat(timespec="seconds"),
                         run_id, str(path), str(destination),
                         f"margin {margin}, tolerance {tolerance}"))
                    print(f"  {rel}: {img.width}x{img.height} -> "
                          f"{right - left}x{bottom - top} "
                          f"(freed {gain:.0f}%)")
                    cropped += 1
            except Exception as e:
                print(f"  [Error] Failed to crop {rel}: {e}")
                failed += 1

        print(f"\nDone! {cropped} image(s) cropped, {skipped} skipped, "
              f"{failed} failed. Results are in '{tk.CROPPED_OUTPUT_DIR}' "
              f"(batch {run_id}).")
        print("Review them, then use the crop feedback option to teach the")
        print("cropper what 'good' looks like — or option 12 to undo the")
        print("whole batch.\n")
    finally:
        conn.close()


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
            # Keep more padding and let dimmer pixels (shadows, page edges)
            # count as content.
            margin = _clamp("crop_margin",
                            get_param(conn, "crop_margin") + 4)
            tolerance = _clamp("crop_tolerance",
                               get_param(conn, "crop_tolerance") - 4)
            set_param(conn, "crop_margin", margin, "crop_too_tight",
                      "crops were cutting off content")
            set_param(conn, "crop_tolerance", tolerance, "crop_too_tight",
                      "crops were cutting off content")
        elif answer == "2":
            margin = _clamp("crop_margin",
                            get_param(conn, "crop_margin") - 4)
            tolerance = _clamp("crop_tolerance",
                               get_param(conn, "crop_tolerance") + 4)
            set_param(conn, "crop_margin", margin, "crop_too_loose",
                      "crops left background in")
            set_param(conn, "crop_tolerance", tolerance, "crop_too_loose",
                      "crops left background in")
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
