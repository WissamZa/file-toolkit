#!/usr/bin/env python3
"""Quick Renamer as a local browser GUI — same dark RTL design language as
the image teaching GUI, with every feature of the original tkinter app:

  * folder browsing (any folder, optional subfolders, natural sort)
  * big live preview for images and PDF pages (with a page pager)
  * template renaming ({orig} {n:03} {date} {mtime} {xx}/xx), counter,
    regex find/replace on the current name, extension changer
  * PDF / EXIF / PNG metadata editing
  * saved templates, conflict handling (keep both / overwrite / cancel),
    undo rename, keyboard-first workflow, light/dark theme

The renaming logic itself is imported verbatim from quick_renamer.py
(make_plan, conflicts, unique_path, read_meta, write_meta) so behaviour is
identical. Serves from 127.0.0.1 only; nothing leaves the machine.
"""

import io
import json
import os
import re
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image, ImageOps

import file_toolkit as ftk
import image_tools as it
import quick_renamer as qr

UNDO_STACK = []            # [(log_id, new_path, old_path)] — session only
SESSION = {"folder": None}  # folder the file list came from
LAST_BATCH = {"run_id": None}  # most recent batch rename, for bulk undo
SESSION["hashes"] = None       # lazy {path: phash} map for dup warnings


def _session_hashes(conn):
    """Perceptual hashes for every image in the opened folder, computed on
    first use and kept until the folder changes or files are renamed."""
    if SESSION.get("hashes") is None:
        files = [p for p in _list_files(SESSION["folder"],
                                        SESSION.get("sub", False))
                 if os.path.splitext(p)[1].lower() in it.IMAGE_EXTENSIONS]
        SESSION["hashes"] = it.hashes_for_files(
            conn, [Path(p) for p in files])
    return SESSION["hashes"]


def _find_duplicate(conn, path):
    """Closest perceptually-similar *other* image within the learned
    similarity threshold. Returns (other_path, distance) or None."""
    hashes = _session_hashes(conn)
    mine = hashes.get(str(path))
    if mine is None:
        return None
    threshold = it.get_param(conn, "similarity_threshold")
    best, best_d = None, None
    for other, h in hashes.items():
        if other == str(path):
            continue
        d = mine[0] - h[0]
        if d <= threshold and (best_d is None or d < best_d):
            best, best_d = other, d
    return (best, int(best_d)) if best else None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _config():
    cfg = qr.load_config()
    cfg.setdefault("templates", list(qr.DEFAULT_TEMPLATES))
    cfg.setdefault("last_template", "xx")
    cfg.setdefault("re_find", "")
    cfg.setdefault("re_repl", "")
    cfg.setdefault("re_icase", False)
    return cfg


def _save_config(cfg):
    try:
        qr.CONFIG_PATH.write_text(
            json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _under_session(path):
    """Only serve files that came from the folder the user opened."""
    folder = SESSION.get("folder")
    if not folder:
        return False
    try:
        return Path(folder).resolve() in Path(path).resolve().parents
    except (OSError, ValueError):
        return False


def _list_files(folder, include_sub):
    files = []
    if include_sub:
        for dp, _, fn in os.walk(folder):
            for name in fn:
                if os.path.splitext(name)[1].lower() in qr.SUPPORTED:
                    files.append(os.path.join(dp, name))
    else:
        for name in os.listdir(folder):
            p = os.path.join(folder, name)
            if os.path.isfile(p) and \
                    os.path.splitext(name)[1].lower() in qr.SUPPORTED:
                files.append(p)
    files.sort(key=lambda p: qr.natural_key(os.path.basename(p)))
    return files


def _file_payload(path):
    try:
        stat = os.stat(path)
    except OSError:
        return None
    ext = os.path.splitext(path)[1].lower()
    return {
        "path": path,
        "name": os.path.basename(path),
        "rel": _rel(path),
        "ext": ext,
        "size": qr.human_size(stat.st_size),
        "mtime": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
        "is_pdf": ext == ".pdf",
        "is_image": ext in qr.IMG_EXT,
    }


def _rel(path):
    try:
        return str(Path(path).relative_to(SESSION.get("folder") or path))
    except ValueError:
        return str(path)


EXIF_TOKEN_RE = re.compile(r"\{exif_date(?::([^{}]*))?\}", re.IGNORECASE)


def _exif_date(path, fmt="%Y-%m-%d"):
    """EXIF capture date formatted with `fmt`, or None when the file has
    no EXIF date (never falls back to the mtime)."""
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            dt = None
            try:
                dt = exif.get_ifd(0x8769).get(36867)   # DateTimeOriginal
            except (AttributeError, KeyError):
                pass
            if not dt:
                dt = exif.get(306)                     # DateTime
            if dt:
                return datetime.strptime(str(dt).strip(),
                                         "%Y:%m:%d %H:%M:%S").strftime(fmt)
    except Exception:
        pass
    return None


def _expand_exif_token(path, tpl):
    """Replace {exif_date[:fmt]} in the template with the file's real
    capture date. Returns (template, error)."""
    def sub(m):
        value = _exif_date(path, m.group(1) or "%Y-%m-%d")
        if value is None:
            raise ValueError("no EXIF date")
        return value
    try:
        return EXIF_TOKEN_RE.sub(sub, tpl), None
    except ValueError:
        return tpl, "لا يحتوي هذا الملف تاريخ تصوير EXIF — استخدم {mtime}"


def _move_hash(old, new):
    """Keep the session hash map consistent across renames."""
    hashes = SESSION.get("hashes")
    if hashes and str(old) in hashes:
        hashes[str(new)] = hashes.pop(str(old))


def _image_bytes(path, max_px):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((max_px, max_px))
        import io
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=88)
        return buf.getvalue()


def _pdf_page_bytes(path, page, max_px):
    import pymupdf
    doc = pymupdf.open(path)
    try:
        page = max(0, min(doc.page_count - 1, page))
        pix = doc[page].get_pixmap(dpi=100)
        png = pix.tobytes("png")
        pages = doc.page_count
    finally:
        doc.close()
    from PIL import Image as PILImage
    import io
    img = PILImage.open(io.BytesIO(png))
    img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue(), pages


# --------------------------------------------------------------------------
# HTTP handler
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, content_type="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, payload):
        self._send(code, json.dumps(payload, ensure_ascii=False).encode())

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode())
        except (ValueError, UnicodeDecodeError):
            return {}

    # ---- GET ---------------------------------------------------------------
    def do_GET(self):
        parsed = urlparse(self.path)
        route, query = parsed.path, parse_qs(parsed.query)

        if route in ("/", "/index.html"):
            # '?shot' delays the load event until the boot fetches have
            # painted — used to take reproducible screenshots of the UI.
            page = PAGE
            if "shot" in query:
                page = page.replace(
                    "</body>", '<script src="/api/slow"></script></body>')
            self._send(200, page.encode(), "text/html; charset=utf-8")
        elif route == "/api/slow":
            import time as _time
            _time.sleep(2.5)
            self._send(200, b"", "application/javascript")
        elif route == "/api/config":
            cfg = _config()
            self._json(200, {k: cfg[k] for k in
                             ("templates", "last_template", "re_find",
                              "re_repl", "re_icase", "dark")})
        elif route == "/api/home":
            self._json(200, {"home": str(ftk.TARGET_DIR)})
        elif route == "/api/subdirs":
            folder = (query.get("folder") or [""])[0] or str(ftk.TARGET_DIR)
            folder = os.path.abspath(unquote(folder))
            if not os.path.isdir(folder):
                self._json(400, {"error": "المجلد غير موجود"})
                return
            try:
                dirs = sorted(
                    (d for d in os.listdir(folder)
                     if os.path.isdir(os.path.join(folder, d))
                     and not d.startswith(".")),
                    key=qr.natural_key)
            except OSError as e:
                self._json(500, {"error": str(e)})
                return
            self._json(200, {"folder": folder, "subdirs": dirs})
        elif route == "/api/browse":
            """Directory listing for the folder-picker modal."""
            path = os.path.abspath(unquote((query.get("path") or [""])[0])
                                   or str(Path.home()))
            if not os.path.isdir(path):
                path = str(Path.home())
            try:
                dirs = sorted(
                    (d for d in os.listdir(path)
                     if os.path.isdir(os.path.join(path, d))
                     and not d.startswith(".")),
                    key=qr.natural_key)
            except OSError as e:
                self._json(500, {"error": str(e)})
                return
            parent = os.path.dirname(path)
            self._json(200, {
                "path": path,
                "parent": parent if parent != path else None,
                "dirs": dirs})
        elif route == "/api/list":
            folder = os.path.abspath(unquote((query.get("folder") or [""])[0]))
            include_sub = (query.get("sub") or ["0"])[0] == "1"
            if not os.path.isdir(folder):
                self._json(400, {"error": "المجلد غير موجود"})
                return
            SESSION["folder"] = folder
            SESSION["sub"] = include_sub
            SESSION["hashes"] = None
            UNDO_STACK.clear()
            files = [f for f in (_file_payload(p)
                                 for p in _list_files(folder, include_sub))
                     if f]
            self._json(200, {"folder": folder, "files": files})
        elif route == "/api/file":
            path = unquote((query.get("path") or [""])[0])
            max_px = min(2000, int((query.get("max") or ["1400"])[0]))
            if not _under_session(path) or not os.path.isfile(path):
                self._json(403, {"error": "مسار غير مسموح"})
                return
            try:
                data = _image_bytes(path, max_px)
            except Exception as e:
                self._json(500, {"error": f"تعذر عرض الصورة: {e}"})
                return
            self._send(200, data, "image/jpeg")
        elif route == "/api/page":
            path = unquote((query.get("path") or [""])[0])
            page = int((query.get("page") or ["0"])[0])
            max_px = min(2000, int((query.get("max") or ["1400"])[0]))
            if not _under_session(path) or not os.path.isfile(path):
                self._json(403, {"error": "مسار غير مسموح"})
                return
            try:
                data, pages = _pdf_page_bytes(path, page, max_px)
            except Exception as e:
                self._json(500, {"error": f"تعذر عرض الـPDF: {e}"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Pages", str(pages))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        elif route == "/api/cropped":
            """The image with the crop the teaching GUI learned applied."""
            path = unquote((query.get("path") or [""])[0])
            if not _under_session(path) or not os.path.isfile(path):
                self._json(403, {"error": "مسار غير مسموح"})
                return
            conn = it._open_db()
            try:
                with Image.open(path) as img:
                    img = ImageOps.exif_transpose(img)
                    box, method, type_ = it.learned_crop(conn, img)
                    if box is None:
                        self._json(404, {"error":
                            "لا يوجد قصّ مقترح لهذه الصورة"})
                        return
                    cropped = img.crop(box)
                    buf = io.BytesIO()
                    cropped.convert("RGB").save(buf, "JPEG", quality=88)
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(buf.getvalue())))
                self.send_header("X-Crop-Method", method)
                # HTTP headers are latin-1 only — send the type KEY and let
                # the UI render its Arabic label.
                self.send_header("X-Crop-Type", type_ or "other")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(buf.getvalue())
            except Exception as e:
                self._json(500, {"error": str(e)})
            finally:
                conn.close()
        elif route == "/api/meta":
            path = unquote((query.get("path") or [""])[0])
            if not _under_session(path) or not os.path.isfile(path):
                self._json(403, {"error": "مسار غير مسموح"})
                return
            try:
                vals, keys = qr.read_meta(path)
                self._json(200, {"values": vals,
                                 "keys": sorted(keys),
                                 "fields": qr.FIELDS})
            except Exception as e:
                self._json(500, {"error": str(e)})
        else:
            self._json(404, {"error": "غير موجود"})

    # ---- POST ---------------------------------------------------------------
    def do_POST(self):
        route = urlparse(self.path).path
        data = self._body()

        if route == "/api/config":
            cfg = _config()
            for key in ("templates", "last_template", "re_find", "re_repl",
                        "re_icase", "dark"):
                if key in data:
                    cfg[key] = data[key]
            _save_config(cfg)
            self._json(200, {"ok": True})
        elif route == "/api/meta_batch":
            """Apply the given (non-empty) field values to every file in the
            opened folder, respecting each file's supported metadata keys.
            Existing values for other fields are left untouched."""
            values = data.get("values") or {}
            filled = {k: v for k, v in values.items() if (v or "").strip()}
            if not filled:
                self._json(400, {"error": "لا حقول معبأة للتطبيق"})
                return
            if not SESSION.get("folder"):
                self._json(400, {"error": "افتح مجلداً أولاً"})
                return
            files = _list_files(SESSION["folder"], SESSION.get("sub", False))
            done = skipped = failed = 0
            errors = []
            for path in files:
                try:
                    vals, keys = qr.read_meta(path)
                    if not keys:
                        skipped += 1
                        continue
                    merged = {k: vals.get(k, "") for k in keys}
                    touched = False
                    for k, v in filled.items():
                        if k in keys and merged.get(k, "") != v:
                            merged[k] = v
                            touched = True
                    if touched:
                        qr.write_meta(path, merged)
                        done += 1
                    else:
                        skipped += 1
                except Exception as e:
                    failed += 1
                    errors.append(f"{os.path.basename(path)}: {e}")
            self._json(200, {"done": done, "skipped": skipped,
                             "failed": failed, "errors": errors[:5]})
        elif route == "/api/plan":
            path = data.get("path", "")
            if not _under_session(path):
                self._json(403, {"error": "مسار غير مسموح"})
                return
            tpl, exif_err = _expand_exif_token(path, data.get("tpl", ""))
            new, err = qr.make_plan(
                path,
                typed=data.get("typed", ""),
                tpl=tpl,
                ext_raw=data.get("ext") or qr.KEEP,
                find=data.get("find", ""),
                repl=data.get("repl", ""),
                icase=bool(data.get("icase")),
                counter=int(data.get("counter") or 1),
            )
            err = exif_err or err
            if exif_err:
                new = None
            conn = it._open_db()
            try:
                dup = _find_duplicate(conn, path) if not err else None
            finally:
                conn.close()
            self._json(200, {
                "new": new, "err": err,
                "same": new == path or (new is None and err is None),
                "conflict": bool(new) and qr.conflicts(new, path),
                "dup": {"path": dup[0],
                        "name": os.path.basename(dup[0]),
                        "distance": dup[1]} if dup else None,
            })
        elif route == "/api/commit":
            path = data.get("path", "")
            new = data.get("new")
            if not _under_session(path) or not os.path.isfile(path):
                self._json(403, {"error": "مسار غير مسموح"})
                return
            if new and data.get("keep_both"):
                new = qr.unique_path(new)          # "name (1).ext" scheme
            msgs = []
            meta = data.get("meta")
            if meta is not None:
                try:
                    qr.write_meta(path, meta)
                    msgs.append("حُفظت البيانات الوصفية")
                except Exception as e:
                    self._json(500, {"error": f"فشل حفظ البيانات: {e}"})
                    return
            log_id = None
            if new:
                try:
                    if data.get("overwrite"):
                        os.replace(path, new)
                    else:
                        os.rename(path, new)
                except Exception as e:
                    self._json(500, {"error": f"فشل إعادة التسمية: {e}"})
                    return
                log_id = None
                try:
                    conn = it._open_db()   # creates operation_log if needed
                    try:
                        conn.execute(
                            "INSERT INTO operation_log (ts, run_id, op, "
                            " source, destination, detail, status) "
                            "VALUES (?, ?, 'move', ?, ?, ?, 'approved')",
                        (datetime.now().isoformat(timespec="seconds"),
                         it._new_run_id(), path, new, "quick renamer"))
                        conn.commit()
                        log_id = conn.execute(
                            "SELECT last_insert_rowid()").fetchone()[0]
                    finally:
                        conn.close()
                except Exception as e:
                    print(f"  [Renamer] operation log failed: {e}")
                UNDO_STACK.append((log_id, new, path))
                _move_hash(path, new)
                SESSION["folder"] = os.path.dirname(new)
                msgs.append(f"أُعيدت التسمية إلى {os.path.basename(new)}")
            self._json(200, {"ok": True, "log_id": log_id,
                             "msgs": msgs, "path": new or path})
        elif route == "/api/batch_plan":
            """Plan the same template for every file in the opened folder."""
            if not SESSION.get("folder"):
                self._json(400, {"error": "افتح مجلداً أولاً"})
                return
            files = _list_files(SESSION["folder"], SESSION.get("sub", False))
            rows = []
            counter = int(data.get("counter") or 1)
            for path in files:
                tpl, exif_err = _expand_exif_token(path, data.get("tpl", ""))
                new, err = qr.make_plan(
                    path, typed=data.get("typed", ""), tpl=tpl,
                    ext_raw=data.get("ext") or qr.KEEP,
                    find=data.get("find", ""), repl=data.get("repl", ""),
                    icase=bool(data.get("icase")), counter=counter)
                err = exif_err or err
                if exif_err:
                    new = None
                uses_n = any((m.group("tok") or "").lower() == "n"
                             for m in qr.TPL_RE.finditer(data.get("tpl", "")))
                if err is None and new is not None and new != path:
                    counter += 1
                rows.append({
                    "path": path, "name": os.path.basename(path),
                    "new": new, "err": err,
                    "same": new == path or (new is None and err is None),
                    "conflict": bool(new) and qr.conflicts(new, path),
                })
            self._json(200, {"rows": rows, "n": len(rows)})
        elif route == "/api/batch_commit":
            """Apply a planned batch; every rename shares one run_id so the
            whole batch can be undone at once."""
            if not SESSION.get("folder"):
                self._json(400, {"error": "افتح مجلداً أولاً"})
                return
            ops = data.get("ops") or []
            if not ops:
                self._json(400, {"error": "لا عمليات في الخطة"})
                return
            run_id = it._new_run_id()
            conn = it._open_db()
            done = skipped = failed = 0
            try:
                for op in ops:
                    path, new = op.get("path", ""), op.get("new", "")
                    if not _under_session(path) or not os.path.isfile(path):
                        skipped += 1
                        continue
                    if not new or os.path.abspath(new) == os.path.abspath(path):
                        skipped += 1
                        continue
                    if os.path.exists(new):
                        if data.get("keep_both"):
                            new = qr.unique_path(new)
                        else:
                            skipped += 1
                            continue
                    try:
                        os.rename(path, new)
                        conn.execute(
                            "INSERT INTO operation_log (ts, run_id, op, "
                            " source, destination, detail, status) "
                            "VALUES (?, ?, 'move', ?, ?, ?, 'approved')",
                            (datetime.now().isoformat(timespec="seconds"),
                             run_id, path, new, "batch rename"))
                        log_id = conn.execute(
                            "SELECT last_insert_rowid()").fetchone()[0]
                        conn.commit()
                        UNDO_STACK.append((log_id, new, path))
                        _move_hash(path, new)
                        done += 1
                    except Exception:
                        failed += 1
                LAST_BATCH["run_id"] = run_id if done else None
            finally:
                conn.close()
            self._json(200, {"done": done, "skipped": skipped,
                             "failed": failed, "run_id": run_id})
        elif route == "/api/undo_batch":
            """Undo a whole batch rename (the most recent one by default)."""
            run_id = data.get("run_id") or LAST_BATCH.get("run_id")
            if not run_id:
                self._json(200, {"ok": False, "error": "لا توجد دفعة جماعية"})
                return
            conn = it._open_db()
            try:
                rows = conn.execute(
                    "SELECT id, source, destination FROM operation_log "
                    "WHERE run_id = ? AND op = 'move' ORDER BY id DESC",
                    (run_id,)).fetchall()
                reverted = failed = 0
                reverted_ids = []
                for log_id, source, destination in rows:
                    try:
                        if os.path.exists(destination):
                            target = source
                            if os.path.exists(source):
                                target = qr.unique_path(source)
                            os.rename(destination, target)
                            _move_hash(destination, target)
                        conn.execute("DELETE FROM operation_log WHERE id = ?",
                                     (log_id,))
                        reverted_ids.append(log_id)
                        reverted += 1
                    except Exception:
                        failed += 1
                conn.commit()
            finally:
                conn.close()
            UNDO_STACK[:] = [u for u in UNDO_STACK
                             if u[0] not in set(reverted_ids)]
            if LAST_BATCH.get("run_id") == run_id:
                LAST_BATCH["run_id"] = None
            self._json(200, {"ok": True, "reverted": reverted,
                             "failed": failed})
        elif route == "/api/undo":
            if not UNDO_STACK:
                self._json(200, {"ok": False, "error": "لا شيء للتراجع عنه"})
                return
            log_id, new_path, old_path = UNDO_STACK.pop()
            try:
                os.rename(new_path, old_path)
                _move_hash(new_path, old_path)
            except Exception as e:
                self._json(500, {"error": f"فشل التراجع: {e}"})
                return
            conn = it._open_db()
            try:
                conn.execute("DELETE FROM operation_log WHERE id = ?",
                             (log_id,))
                conn.commit()
            finally:
                conn.close()
            SESSION["folder"] = os.path.dirname(old_path)
            self._json(200, {"ok": True, "restored": old_path,
                             "renamed": new_path})
        else:
            self._json(404, {"error": "غير موجود"})


# --------------------------------------------------------------------------
# The single-page UI
# --------------------------------------------------------------------------
PAGE = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<title>Quick Renamer — إعادة التسمية والبيانات الوصفية</title>
<style>
:root {
  --bg:#101318; --surface:#181c24; --surface2:#20252f; --border:#2c3342;
  --text:#e9ecf2; --muted:#98a2b3; --accent:#4f8cff; --accent-d:#3a6fd8;
  --ok:#3fce8f; --bad:#f0705f; --warn:#f5b04c; --field:#131720;
}
body.light {
  --bg:#eef1f6; --surface:#ffffff; --surface2:#e6eaf1; --border:#c9d1de;
  --text:#1b2130; --muted:#5d6677;
  --accent:#2f6fe0; --accent-d:#2456b3; --ok:#1e9e63; --bad:#cf4a37;
  --warn:#b57a12; --field:#f5f7fb;
}
* { box-sizing:border-box; margin:0; padding:0; }
body { background:var(--bg); color:var(--text); min-height:100vh;
  font-family:"Segoe UI","Noto Sans Arabic","Noto Kufi Arabic",system-ui,sans-serif; }
header { display:flex; justify-content:space-between; align-items:center;
  padding:14px 26px; border-bottom:1px solid var(--border);
  background:var(--surface); position:sticky; top:0; z-index:10; }
h1 { font-size:19px; font-weight:800; display:flex; gap:10px; align-items:center; }
h1 .logo { width:32px; height:32px; border-radius:9px; display:grid; place-items:center;
  background:linear-gradient(135deg,var(--accent),#7c5cff); font-size:16px; }
.muted { color:var(--muted); font-size:12.5px; }
.ltr { direction:ltr; unicode-bidi:isolate; }
main { padding:16px 28px 70px; }
.card { background:var(--surface); border:1px solid var(--border);
  border-radius:14px; padding:14px 16px; margin-bottom:12px; }
.btn { border:1px solid var(--border); border-radius:9px; padding:9px 16px;
  cursor:pointer; font:inherit; font-weight:700; font-size:13.5px;
  background:var(--surface2); color:var(--text); transition:.15s; }
.btn:hover { filter:brightness(1.12); }
.btn.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
.btn.ok { background:var(--ok); border-color:var(--ok); color:#10131a; }
.btn.warnb { background:var(--warn); border-color:var(--warn); color:#10131a; }
.btn.bad { background:var(--bad); border-color:var(--bad); color:#fff; }
.btn.small { padding:6px 12px; font-size:12.5px; }
.row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
.spacer { flex:1; }
.grid { display:grid; grid-template-columns:1fr 350px 210px; gap:14px; align-items:start; }
@media (max-width:1100px){ .grid { grid-template-columns:1fr 350px; }
  #file-list-card { grid-column:1 / -1; order:3; } }
@media (max-width:760px){ .grid { grid-template-columns:1fr; } }
#file-list { max-height:66vh; overflow-y:auto; margin-top:6px; }
.fl-item { display:flex; align-items:center; gap:6px; width:100%; text-align:right;
  padding:6px 9px; border:none; border-radius:8px; cursor:pointer; font:inherit;
  font-size:12px; background:transparent; color:var(--text); direction:ltr;
  overflow:hidden; }
.fl-item:hover { background:var(--surface2); }
.fl-item.cur { background:var(--accent); color:#fff; font-weight:700; }
.fl-item .mark { flex-shrink:0; }
.fl-item.renamed .name { color:var(--ok); text-decoration:line-through; }
.fl-item.cur.renamed .name { color:#fff; }
.preview-wrap { background:#0b0d11; border:1px solid var(--border); border-radius:14px;
  min-height:460px; display:grid; place-items:center; overflow:hidden; position:relative; }
body.light .preview-wrap { background:#dfe4ec; }
#pv-img { max-width:100%; max-height:68vh; display:block; }
.pager { position:absolute; bottom:12px; left:50%; transform:translateX(-50%);
  display:none; gap:8px; background:rgba(10,12,16,.75); padding:6px 10px;
  border-radius:99px; }
.pager button { background:var(--surface2); color:var(--text); border:1px solid var(--border);
  border-radius:99px; padding:5px 14px; cursor:pointer; font:inherit; font-size:13px; }
.pager button:disabled { opacity:.4; cursor:default; }
label.fld { display:block; font-size:12px; color:var(--muted); margin:8px 0 4px; }
input[type=text], input[type=number], select {
  width:100%; background:var(--field); color:var(--text);
  border:1px solid var(--border); border-radius:9px; padding:9px 12px;
  font:inherit; font-size:14px;
}
input:focus, select:focus { outline:2px solid var(--accent); outline-offset:-1px; }
#preview-line { min-height:22px; font-size:14px; font-weight:700; margin-top:8px;
  word-break:break-all; }
#preview-line.ok { color:var(--ok); }
#preview-line.same { color:var(--warn); }
#preview-line.err { color:var(--bad); }
.chip { display:inline-block; background:var(--surface2); border:1px solid var(--border);
  border-radius:99px; padding:4px 12px; margin:3px; cursor:pointer; font-size:12.5px; }
.chip:hover { border-color:var(--accent); }
.meta-fld { display:flex; gap:8px; align-items:center; margin-top:6px; }
.meta-fld label { width:120px; font-size:12.5px; color:var(--muted); flex-shrink:0; }
#status { position:fixed; bottom:0; left:0; right:0; background:var(--surface);
  border-top:1px solid var(--border); padding:9px 26px; font-size:13.5px;
  color:var(--muted); z-index:9; }
.progressbar { height:6px; background:var(--surface2); border-radius:99px; overflow:hidden; }
.progressbar > div { height:100%; background:var(--accent); transition:.2s; }
.modal-bg { display:none; position:fixed; inset:0; background:rgba(5,7,10,.6);
  z-index:50; place-items:center; }
.modal-bg.show { display:grid; }
.modal { background:var(--surface); border:1px solid var(--border); border-radius:16px;
  padding:22px 26px; max-width:560px; width:92%; }
.modal h3 { margin-bottom:10px; }
.modal .actions { display:flex; gap:10px; margin-top:16px; flex-wrap:wrap; }
kbd { background:var(--surface2); border:1px solid var(--border); border-radius:6px;
  padding:1px 7px; font-size:12px; direction:ltr; display:inline-block; }
table.help { width:100%; border-collapse:collapse; margin-top:8px; }
table.help td { padding:6px 8px; border-bottom:1px solid var(--border); font-size:13px; }
#toast { position:fixed; bottom:52px; left:50%; transform:translateX(-50%);
  background:var(--surface2); border:1px solid var(--border); padding:11px 20px;
  border-radius:12px; font-size:13.5px; opacity:0; transition:.25s; pointer-events:none;
  z-index:99; }
#toast.show { opacity:1; }
.empty { text-align:center; color:var(--muted); padding:30px 0; }
</style>
</head>
<body>
<header>
  <h1><span class="logo">✏️</span> Quick Renamer
    <span class="muted" style="font-weight:400">إعادة تسمية بالمعاينة الحية + البيانات الوصفية</span></h1>
  <div class="row">
    <span class="muted ltr" id="folder-label"></span>
    <button class="btn small" id="theme-btn">☀️ فاتح</button>
  </div>
</header>
<main>
  <div class="card">
    <div class="row">
      <span class="muted">المجلد:</span>
      <input type="text" id="folder" style="flex:1; min-width:220px" class="ltr">
      <button class="btn primary" onclick="openBrowser()">📂 اختيار مجلد…</button>
      <button class="btn" onclick="openFolder()" title="فتح المسار المكتوب أعلاه">فتح المسار</button>
      <label class="row" style="gap:5px; cursor:pointer; font-size:13px;">
        <input type="checkbox" id="sub-chk"> تضمين المجلدات الفرعية
      </label>
    </div>
    <div id="subdirs" style="margin-top:6px;"></div>
  </div>

  <div class="grid" id="work-grid" style="display:none;">
    <div class="card" id="file-list-card">
      <div class="row" style="font-weight:800;">📁 الملفات
        <span class="spacer"></span>
        <span class="muted" id="fl-count" style="font-weight:400;"></span></div>
      <div id="file-list"></div>
    </div>
    <div>
      <div class="preview-wrap">
        <img id="pv-img" alt="">
        <div class="pager" id="pager">
          <button id="pg-prev" onclick="pageNav(-1)">▶ الصفحة السابقة</button>
          <span id="pg-label" class="ltr" style="align-self:center; color:var(--muted); font-size:13px;"></span>
          <button id="pg-next" onclick="pageNav(1)">◀ الصفحة التالية</button>
        </div>
      </div>
      <div class="card" style="margin-top:12px;">
        <div class="row">
          <span id="pos" style="font-weight:800;"></span>
          <div class="progressbar" style="flex:1; min-width:140px;"><div id="bar"></div></div>
          <span class="spacer"></span>
          <button class="btn small" id="crop-btn" onclick="toggleCrop()">✂️ معاينة القصّ المتعلم</button>
          <button class="btn small" onclick="nav(-1)">→ السابق</button>
          <button class="btn small" onclick="nav(1)">تخطي ←</button>
        </div>
      </div>
      <div class="card">
        <div class="row">
          <span id="f-name" style="font-weight:800;" class="ltr"></span>
          <span class="spacer"></span>
          <span class="muted" id="f-meta"></span>
        </div>
        <div class="muted ltr" id="f-path" style="margin-top:4px; font-size:12px; word-break:break-all;"></div>
      </div>
    </div>

    <div>
      <div class="card">
        <label class="fld">الاسم الجديد — اكتب النص مكان xx</label>
        <input type="text" id="typed" placeholder="اكتب هنا… (Enter فارغ = تخطي)">
        <div id="preview-line"></div>
        <div id="dup-warning" style="display:none; margin-top:6px; padding:7px 12px;
             border-radius:9px; font-size:12.5px; font-weight:700;
             background:var(--warn); color:#10131a;"></div>
        <label class="fld">القالب</label>
        <div class="row" style="flex-wrap:nowrap;">
          <input type="text" id="tpl" list="tpl-list" class="ltr" style="flex:1;">
          <datalist id="tpl-list"></datalist>
          <button class="btn small" onclick="tplSave()" title="حفظ القالب">💾</button>
          <button class="btn small" onclick="tplDel()" title="حذف القالب المحفوظ">🗑</button>
          <button class="btn small" onclick="showHelp()">؟</button>
        </div>
        <div class="row" style="margin-top:8px;">
          <span class="muted" style="font-size:12.5px;">العدّاد {n} يبدأ من</span>
          <input type="number" id="counter" value="1" style="width:80px;" min="0">
        </div>
        <div class="row" style="margin-top:10px;">
          <button class="btn small" onclick="toggleRegex()">▸ Regex على الاسم الحالي</button>
        </div>
        <div id="regex-box" style="display:none; margin-top:8px;">
          <label class="fld">ابحث (Regex)</label>
          <input type="text" id="re-find" class="ltr" placeholder="مثال: IMG_">
          <label class="fld">استبدل بـ</label>
          <input type="text" id="re-repl" class="ltr">
          <label class="row" style="margin-top:6px; gap:6px; cursor:pointer; font-size:13px;">
            <input type="checkbox" id="re-icase"> تجاهل حالة الأحرف
          </label>
        </div>
        <label class="fld">الامتداد</label>
        <div class="row" style="flex-wrap:nowrap;">
          <select id="ext" style="flex:1;"></select>
          <label class="row" style="gap:5px; cursor:pointer; font-size:12.5px;">
            <input type="checkbox" id="ext-remember"> تثبيت
          </label>
        </div>
      </div>
      <div class="card">
        <button class="btn primary" style="width:100%; padding:12px;" onclick="saveNext()">
          💾 حفظ (+ بيانات) والتالي ⏎</button>
        <button class="btn" style="width:100%; margin-top:8px;" onclick="openBatch()">
          🏷️ إعادة تسمية جماعية — تطبيق القالب على كل الملفات</button>
        <div class="row" style="margin-top:8px;">
          <button class="btn" style="flex:1" onclick="doUndo()">↩︎ تراجع عن آخر إعادة تسمية (Ctrl+U)</button>
          <button class="btn" style="flex:1" onclick="undoBatch()">↩︎ تراجع عن آخر دفعة جماعية</button>
        </div>
      </div>
      <div class="card">
        <div style="font-weight:800; margin-bottom:4px;">البيانات الوصفية</div>
        <div id="meta-note" class="muted" style="font-size:12.5px;"></div>
        <div id="meta-fields"></div>
        <button class="btn" style="width:100%; margin-top:10px;" id="meta-batch-btn"
                onclick="applyMetaBatch()">📦 تطبيق الحقول المعبأة على كل الملفات</button>
        <div class="muted" style="margin-top:6px; font-size:11.5px;">
          يُطبَّق كل حقل معبأ على الملفات التي تدعمه (فارغ = لا يُلمس). لا يمكن التراجع عن هذا.</div>
      </div>
    </div>
  </div>
  <div id="empty-state" class="card empty">اكتب مسار مجلد في الأعلى واضغط «فتح» لبدء إعادة التسمية</div>
</main>
<div id="status">جاهز</div>

<div class="modal-bg" id="conflict-modal">
  <div class="modal">
    <h3>⚠️ الاسم موجود مسبقاً</h3>
    <div class="muted ltr" id="conflict-detail" style="word-break:break-all;"></div>
    <div class="actions">
      <button class="btn ok" onclick="resolveConflict('keep')">إبقاء الاثنين (اسم جديد تلقائي)</button>
      <button class="btn bad" onclick="resolveConflict('overwrite')">استبدال الموجود (لا يمكن التراجع)</button>
      <button class="btn" onclick="resolveConflict('cancel')">إلغاء</button>
    </div>
  </div>
</div>

<div class="modal-bg" id="browse-modal">
  <div class="modal">
    <h3>📂 اختر مجلداً</h3>
    <div class="row" style="margin-bottom:10px;">
      <span class="muted">المسار الحالي:</span>
      <span id="browse-path" class="ltr" style="font-weight:700; word-break:break-all;"></span>
    </div>
    <div id="browse-list" style="max-height:46vh; overflow-y:auto;
         border:1px solid var(--border); border-radius:10px; padding:8px;"></div>
    <div class="actions">
      <button class="btn" onclick="browseUp()">⬆️ للمجلد الأعلى</button>
      <span class="spacer"></span>
      <button class="btn primary" onclick="browseSelect()">✓ فتح هذا المجلد</button>
      <button class="btn" onclick="hide('browse-modal')">إلغاء</button>
    </div>
  </div>
</div>

<div class="modal-bg" id="batch-modal">
  <div class="modal" style="max-width:820px;">
    <h3>🏷️ إعادة تسمية جماعية — معاينة قبل التنفيذ</h3>
    <div class="muted" style="margin-bottom:8px;">
      سيُطبَّق القالب الحالي على كل ملفات المجلد المفتوح. لا شيء يُنفَّذ قبل ضغط «تنفيذ».</div>
    <div style="max-height:46vh; overflow-y:auto; border:1px solid var(--border); border-radius:10px;">
      <table id="batch-table" style="width:100%;">
        <thead><tr><th>الاسم الحالي</th><th></th><th>الاسم الجديد</th><th>الحالة</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
    <div class="row" style="margin-top:12px;">
      <label class="row" style="gap:6px; cursor:pointer; font-size:13px;">
        <input type="checkbox" id="batch-keep-both"> المتعارضة: إبقاء الاثنين بدل تخطيها
      </label>
      <span class="spacer"></span>
      <button class="btn ok" id="batch-go" onclick="runBatch()">✓ تنفيذ الدفعة</button>
      <button class="btn" onclick="hide('batch-modal')">إلغاء</button>
    </div>
  </div>
</div>

<div class="modal-bg" id="help-modal">
  <div class="modal">
    <h3>؟ دليل القوالب والاختصارات</h3>
    <table class="help">
      <tr><td class="ltr">xx أو {xx}</td><td>النص الذي تكتبه في خانة الاسم</td></tr>
      <tr><td class="ltr">{orig}</td><td>الاسم الحالي للملف (بعد الـRegex إن وُجد)</td></tr>
      <tr><td class="ltr">{n} / {n:03}</td><td>العدّاد التلقائي — :03 يعني ثلاث خانات (001)</td></tr>
      <tr><td class="ltr">{date}</td><td>تاريخ اليوم — بتنسيق مثل {date:%m-%Y}</td></tr>
      <tr><td class="ltr">{mtime}</td><td>تاريخ آخر تعديل للملف — مثل {mtime:%d-%m-%Y}</td></tr>
      <tr><td class="ltr">{exif_date}</td><td>تاريخ التصوير من EXIF (وليس تاريخ التعديل) — مثل {exif_date:%d-%m-%Y}. مثالي لصور الواتساب</td></tr>
    </table>
    <div class="muted" style="margin-top:10px; line-height:1.9;">
      الـRegex يُطبَّق على الاسم الحالي قبل القالب (يؤثر في {orig}).<br>
      <kbd>Enter</kbd> حفظ والتالي • <kbd>Alt</kbd>+<kbd>→/←</kbd> تنقّل دائماً •
      <kbd>↑/↓</kbd> صفحات الـPDF • <kbd>Ctrl+U</kbd> تراجع
    </div>
    <div class="actions"><button class="btn primary" onclick="hide('help-modal')">فهمت</button></div>
  </div>
</div>
<div id="toast"></div>

<script>
window.onerror = (msg, src, line) => { const s = document.getElementById('status');
  if (s) s.textContent = 'JS ERROR: ' + msg + ' @' + line; };
const $ = id => document.getElementById(id);
let FILES = [], IDX = 0, PAGES = 1, PAGE = 0, META = null, META_KEYS = [],
    META_VALUES = {}, TEMPLATES = [], PENDING_PLAN = null, EXT_PINNED = null;

function toast(msg) {
  const t = $('toast'); t.textContent = msg; t.classList.add('show');
  clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove('show'), 2600);
}
function status(msg) { $('status').textContent = msg; }
async function api(path, body) {
  const res = await fetch(path, body ? {method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)} : {});
  const d = await res.json();
  if (d.error) throw new Error(d.error);
  return d;
}
function hide(id) { $(id).classList.remove('show'); }

/* ---------- theme ---------- */
$('theme-btn').onclick = () => {
  document.body.classList.toggle('light');
  const light = document.body.classList.contains('light');
  $('theme-btn').textContent = light ? '🌙 داكن' : '☀️ فاتح';
  api('/api/config', {dark: !light});
};

/* ---------- folder ---------- */
$('sub-chk').onchange = () => openFolder();
async function openFolder() {
  const folder = $('folder').value.trim();
  try {
    const d = await api(`/api/list?folder=${encodeURIComponent(folder)}&sub=${$('sub-chk').checked ? 1 : 0}`);
    FILES = d.files; IDX = 0; PAGE = 0; PAGES = 1;
    RENAMED.clear();
    $('folder-label').textContent = d.folder;
    $('work-grid').style.display = 'grid';
    $('empty-state').style.display = 'none';
    const sd = await api(`/api/subdirs?folder=${encodeURIComponent(d.folder)}`);
    const subwrap = $('subdirs'); subwrap.innerHTML = '';
    for (const s of sd.subdirs) {
      const chip = document.createElement('span');
      chip.className = 'chip ltr';
      chip.textContent = s;
      chip.onclick = () => { $('folder').value = sd.folder + '/' + s; openFolder(); };
      subwrap.appendChild(chip);
    }
    status(`تم فتح ${d.folder} — ${FILES.length} ملف مدعوم`);
    if (FILES.length) load(0);
    else { $('pos').textContent = ''; toast('لا توجد ملفات مدعومة في هذا المجلد'); }
  } catch (e) { toast('خطأ: ' + e.message); }
}

/* ---------- side file list ---------- */
const RENAMED = new Set();               // names renamed this session
function renderFileList() {
  const box = $('file-list');
  if (!box) return;
  box.innerHTML = '';
  $('fl-count').textContent = FILES.length ? `${FILES.length}` : '';
  for (let i = 0; i < FILES.length; i++) {
    const b = document.createElement('button');
    b.className = 'fl-item' + (i === IDX ? ' cur' : '') +
                  (RENAMED.has(FILES[i].name) ? ' renamed' : '');
    b.title = FILES[i].path;
    b.innerHTML = `<span class="mark">${i === IDX ? '▶' :
                   (RENAMED.has(FILES[i].name) ? '✓' : '')}</span>
                   <span class="name">${FILES[i].name}</span>`;
    b.onclick = () => load(i);
    box.appendChild(b);
  }
}

/* ---------- file loading / preview ---------- */
async function load(i) {
  IDX = i;
  const f = FILES[i];
  if (!f) return;
  CROP_ON = false;
  if ($('crop-btn')) $('crop-btn').textContent = '✂️ معاينة القصّ المتعلم';
  $('f-name').textContent = f.name;
  $('f-path').textContent = f.path;
  $('f-meta').textContent = `${f.ext} • ${f.size} • ${f.mtime}`;
  $('pos').textContent = `ملف ${i + 1} من ${FILES.length}`;
  $('bar').style.width = ((i + 1) / FILES.length * 100) + '%';
  PAGE = 0; PAGES = 1;
  if (f.is_pdf) {
    try { await showPdfPage(f); }
    catch (e) { $('pager').style.display = 'none'; toast('تعذر عرض الـPDF: ' + e.message); }
  }
  else {
    $('pager').style.display = 'none';
    $('pv-img').src = `/api/file?max=1400&path=${encodeURIComponent(f.path)}`;
  }
  await loadMeta(f);
  renderFileList();
  refreshPreview();
}
async function showPdfPage(f) {
  const res = await fetch(`/api/page?max=1400&page=${PAGE}&path=${encodeURIComponent(f.path)}`);
  if (!res.ok) { const e = await res.json(); throw new Error(e.error); }
  PAGES = parseInt(res.headers.get('X-Pages') || '1');
  $('pv-img').src = URL.createObjectURL(await res.blob());
  $('pager').style.display = PAGES > 1 ? 'flex' : 'none';
  $('pg-label').textContent = `${PAGE + 1} / ${PAGES}`;
  $('pg-prev').disabled = PAGE === 0;
  $('pg-next').disabled = PAGE >= PAGES - 1;
}
async function pageNav(step) {
  PAGE = Math.max(0, Math.min(PAGES - 1, PAGE + step));
  try { await showPdfPage(FILES[IDX]); } catch (e) { toast(e.message); }
}

/* ---------- learned-crop preview ---------- */
let CROP_ON = false, CROP_FULL_SRC = '';
async function toggleCrop() {
  const img = $('pv-img'), btn = $('crop-btn');
  if (CROP_ON) {                          // back to the full image
    CROP_ON = false;
    img.src = CROP_FULL_SRC;
    btn.textContent = '✂️ معاينة القصّ المتعلم';
    return;
  }
  const f = FILES[IDX];
  if (!f) return;
  try {
    const res = await fetch(`/api/cropped?path=${encodeURIComponent(f.path)}`);
    if (!res.ok) { const e = await res.json(); throw new Error(e.error); }
    CROP_FULL_SRC = img.src;
    CROP_ON = true;
    img.src = URL.createObjectURL(await res.blob());
    const method = res.headers.get('X-Crop-Method');
    const typeKey = res.headers.get('X-Crop-Type');
    const typeLabel = TYPES[typeKey] ? TYPES[typeKey].ar : '';
    btn.textContent = '↩︎ عرض الصورة كاملة' +
      (method === 'taught' ? ` (قصّ متعلَّم ✓ ${typeLabel})` : ' (تلقائي)');
  } catch (e) { toast(e.message); }
}
function nav(step) {
  const i = IDX + step;
  if (i < 0 || i >= FILES.length) return toast('لا يوجد ملف آخر في هذا الاتجاه');
  load(i);
}

/* ---------- metadata ---------- */
async function loadMeta(f) {
  const d = await api('/api/meta?path=' + encodeURIComponent(f.path));
  META_VALUES = d.values; META_KEYS = d.keys;
  const box = $('meta-fields'); box.innerHTML = '';
  if (!META_KEYS.length) {
    $('meta-note').textContent = 'لا توجد بيانات قابلة للتعديل لهذا النوع';
    return;
  }
  $('meta-note').textContent = f.is_pdf ? 'بيانات PDF (تُحفظ داخل الملف)' :
      (f.ext === '.png' ? 'أقسام نصية PNG' : 'حقول EXIF للصورة');
  for (const [key, label] of d.fields) {
    if (!META_KEYS.includes(key)) continue;
    const row = document.createElement('div');
    row.className = 'meta-fld';
    row.innerHTML = `<label>${label}</label>`;
    const inp = document.createElement('input');
    inp.type = 'text'; inp.value = META_VALUES[key] || '';
    inp.oninput = () => { META_VALUES[key] = inp.value; };
    row.appendChild(inp); box.appendChild(row);
  }
}
function metaIfDirty() {
  if (!META_KEYS.length) return null;
  return Object.fromEntries(META_KEYS.map(k => [k, META_VALUES[k] ?? '']));
}
async function applyMetaBatch() {
  const values = metaIfDirty();
  if (!values || !Object.values(values).some(v => (v || '').trim()))
    return toast('عبّئ حقلاً واحداً على الأقل أولاً');
  if (!confirm('تطبيق الحقول المعبأة على كل ملفات المجلد المفتوح؟\\n' +
               'الحقول الفارغة لن تُلمس، ولا يمكن التراجع.')) return;
  try {
    const d = await api('/api/meta_batch', { values });
    let msg = `تم: ${d.done} ملف حُدّث، ${d.skipped} لم يحتج تغييراً`;
    if (d.failed) msg += `، ${d.failed} فشلت`;
    status(msg); toast(msg);
    if (d.errors && d.errors.length) console.log('meta errors:', d.errors);
  } catch (e) { toast('خطأ: ' + e.message); }
}

/* ---------- planning / live preview ---------- */
function planInput() {
  return { path: FILES[IDX]?.path, typed: $('typed').value,
    tpl: $('tpl').value, ext: $('ext').value || null,
    find: $('re-find').value, repl: $('re-repl').value,
    icase: $('re-icase').checked, counter: parseInt($('counter').value || '1') };
}
let planTimer = null;
function refreshPreview() {
  clearTimeout(planTimer);
  if (!FILES.length) return;
  planTimer = setTimeout(async () => {
    const line = $('preview-line');
    try {
      const d = await api('/api/plan', planInput());
      PENDING_PLAN = d;
      const dw = $('dup-warning');
      if (d.dup && !d.same) {
        dw.style.display = 'block';
        dw.textContent = `⚠ تحذير تكرار: هذه الصورة تشبه «${d.dup.name}» بصرياً (بعد ${d.dup.distance}) — تأكد قبل التسمية`;
      } else { dw.style.display = 'none'; }
      if (d.err) { line.className = 'err'; line.textContent = '✗ ' + d.err; }
      else if (d.same) { line.className = 'same'; line.textContent = '— لا تغيير (سيتم التخطي)'; }
      else {
        const base = `→ ${d.new.split('/').pop()}`;
        if (d.conflict) { line.className = 'err'; line.textContent = '⚠ يوجد ملف بنفس الاسم: ' + base; }
        else { line.className = 'ok'; line.textContent = '✓ ' + base; }
      }
    } catch (e) { line.className = 'err'; line.textContent = e.message; }
  }, 220);
}
for (const id of ['typed', 'tpl', 're-find', 're-repl', 'counter'])
  $(id).addEventListener('input', refreshPreview);
for (const id of ['re-icase']) $(id).addEventListener('change', refreshPreview);
$('ext').addEventListener('change', () => {
  if ($('ext-remember').checked) EXT_PINNED = $('ext').value;
  refreshPreview();
});

/* ---------- commit ---------- */
async function saveNext() {
  if (!FILES.length) return;
  const f = FILES[IDX];
  const d = await api('/api/plan', planInput());
  PENDING_PLAN = d;
  if (d.err) { toast('✗ ' + d.err); return; }
  if (d.conflict) {
    $('conflict-detail').textContent = d.new;
    $('conflict-modal').classList.add('show');
    return;                                  // resolved by resolveConflict()
  }
  await commit(f, d.new, false);
}
async function resolveConflict(choice) {
  hide('conflict-modal');
  if (choice === 'cancel') { status('أُلغي — لم يتغير شيء'); return; }
  const f = FILES[IDX];
  await commit(f, PENDING_PLAN.new, choice === 'overwrite',
               choice === 'keep');
}
async function commit(f, new_name, overwrite, keep_both) {
  try {
    const d = await api('/api/commit', { path: f.path, new: new_name === f.path ? null : new_name,
      overwrite, keep_both: !!keep_both, meta: metaIfDirty() });
    status(d.msgs.join(' • ') || 'لا تغيير؛ انتقلنا للملف التالي');
    toast(d.msgs.join(' • ') || 'لا تغيير');
    f.path = d.path; f.name = d.path.split('/').pop();
    RENAMED.add(f.name);
    if ($('tpl').value && /{n(?::[^}]*)?}/i.test($('tpl').value))
      $('counter').value = (parseInt($('counter').value || '1')) + 1;
    if (IDX + 1 < FILES.length) load(IDX + 1);
    else { toast('انتهت قائمة الملفات ✓'); $('preview-line').textContent = ''; }
  } catch (e) { toast('خطأ: ' + e.message); }
}

/* ---------- folder picker ---------- */
let BROWSE_PATH = '';
async function openBrowser() {
  $('browse-modal').classList.add('show');
  await browseTo($('folder').value.trim() || '');
}
async function browseTo(p) {
  try {
    const d = await api('/api/browse?path=' + encodeURIComponent(p));
    BROWSE_PATH = d.path;
    $('browse-path').textContent = d.path;
    const list = $('browse-list');
    list.innerHTML = '';
    if (d.parent) {
      const up = document.createElement('div');
      up.className = 'chip';
      up.textContent = '⬆️ .. (المجلد الأعلى)';
      up.style.width = '100%';
      up.onclick = () => browseTo(d.parent);
      list.appendChild(up);
    }
    for (const dir of d.dirs) {
      const item = document.createElement('div');
      item.className = 'chip ltr';
      item.style.width = '100%';
      item.textContent = '📁 ' + dir;
      item.onclick = () => browseTo(
        BROWSE_PATH.endsWith('/') ? BROWSE_PATH + dir
                                  : BROWSE_PATH + '/' + dir);
      list.appendChild(item);
    }
    if (!list.children.length)
      list.innerHTML = '<div class="empty">لا توجد مجلدات فرعية هنا</div>';
  } catch (e) { toast('خطأ: ' + e.message); }
}
function browseUp() {
  const p = $('browse-path').textContent;
  browseTo(p.slice(0, p.lastIndexOf('/')) || '/');
}
function browseSelect() {
  hide('browse-modal');
  $('folder').value = BROWSE_PATH;
  openFolder();
}

/* ---------- batch rename ---------- */
let BATCH_ROWS = [];
async function openBatch() {
  if (!FILES.length) return toast('افتح مجلداً فيه ملفات أولاً');
  try {
    const d = await api('/api/batch_plan', { ...planInput() });
    BATCH_ROWS = d.rows;
    const tb = $('batch-table').querySelector('tbody');
    tb.innerHTML = '';
    for (const r of d.rows) {
      const tr = document.createElement('tr');
      let badge, color;
      if (r.err) { badge = '✗ ' + r.err; color = 'var(--bad)'; }
      else if (r.same) { badge = '— لا تغيير'; color = 'var(--muted)'; }
      else if (r.conflict) { badge = '⚠ يوجد ملف بنفس الاسم'; color = 'var(--warn)'; }
      else { badge = '✓'; color = 'var(--ok)'; }
      tr.innerHTML = `<td class="ltr">${r.name}</td>
        <td class="muted">←</td>
        <td class="ltr" style="color:${r.err || r.same ? 'var(--muted)' : 'var(--ok)'}">${r.new ? r.new.split('/').pop() : '—'}</td>
        <td style="color:${color}; font-size:12.5px;">${badge}</td>`;
      tb.appendChild(tr);
    }
    $('batch-modal').classList.add('show');
  } catch (e) { toast('خطأ: ' + e.message); }
}
async function runBatch() {
  const ops = BATCH_ROWS
    .filter(r => !r.err && !r.same && r.new)
    .map(r => ({ path: r.path, new: r.new, conflict: r.conflict }));
  if (!ops.length) return toast('لا عمليات قابلة للتنفيذ');
  try {
    const d = await api('/api/batch_commit', { ops,
      keep_both: $('batch-keep-both').checked });
    hide('batch-modal');
    status(`الدفعة: ${d.done} نُفّذت، ${d.skipped} تخطّت، ${d.failed} فشلت — التراجع متاح بضغطة`);
    toast(`نُفّذت ${d.done} إعادة تسمية ✓`);
    for (const op of ops) {
      const f = FILES.find(x => x.path === op.path);
      if (f) { f.path = op.new; f.name = op.new.split('/').pop(); }
    }
    await openFolderSoft();
  } catch (e) { toast('خطأ: ' + e.message); }
}
async function undoBatch() {
  try {
    const d = await api('/api/undo_batch', {});
    if (!d.ok) return toast(d.error || 'لا توجد دفعة جماعية');
    toast(`تراجعت الدفعة: عاد ${d.reverted} ملف لمكانه ✓`);
    status(`تراجعت الدفعة الجماعية: ${d.reverted} ملف` +
           (d.failed ? `، ${d.failed} فشلت` : ''));
    await openFolderSoft();
  } catch (e) { toast('خطأ: ' + e.message); }
}
async function openFolderSoft() {
  // refresh the in-memory list from disk without clearing the folder input
  const folder = $('folder').value.trim();
  try {
    const d = await api(`/api/list?folder=${encodeURIComponent(folder)}&sub=${$('sub-chk').checked ? 1 : 0}`);
    FILES = d.files;
    IDX = Math.min(IDX, Math.max(0, FILES.length - 1));
    RENAMED.clear();
    if (FILES.length) load(IDX); else { $('pos').textContent = ''; }
  } catch (e) { /* keep current state */ }
}

/* ---------- undo ---------- */
async function doUndo() {
  try {
    const d = await api('/api/undo', {});
    if (!d.ok) return toast(d.error || 'لا شيء للتراجع عنه');
    for (let i = 0; i < FILES.length; i++)
      if (FILES[i].path === d.renamed) { FILES[i].path = d.restored;
        FILES[i].name = d.restored.split('/').pop(); break; }
    toast(`استُعيد ${d.restored.split('/').pop()}`);
    status(`استُعيد ${d.restored.split('/').pop()} (تغييرات البيانات الوصفية لا تُتراجع)`);
    load(IDX);
  } catch (e) { toast('خطأ: ' + e.message); }
}

/* ---------- templates / regex / extension ---------- */
async function tplSave() {
  const v = $('tpl').value.trim();
  if (!v) return;
  if (!TEMPLATES.includes(v)) TEMPLATES.push(v);
  $('tpl-list').innerHTML = TEMPLATES.map(t => `<option value="${t}">`).join('');
  await api('/api/config', { templates: TEMPLATES, last_template: v });
  toast('حُفظ القالب ✓');
}
async function tplDel() {
  const v = $('tpl').value.trim();
  TEMPLATES = TEMPLATES.filter(t => t !== v);
  $('tpl-list').innerHTML = TEMPLATES.map(t => `<option value="${t}">`).join('');
  await api('/api/config', { templates: TEMPLATES });
  toast('حُذف القالب المحفوظ');
}
function toggleRegex() {
  const box = $('regex-box');
  box.style.display = box.style.display === 'none' ? 'block' : 'none';
  refreshPreview();
}
function showHelp() { $('help-modal').classList.add('show'); }

/* ---------- keyboard ---------- */
window.addEventListener('keydown', e => {
  if (document.querySelector('.modal-bg.show')) return;
  const typing = ['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)
                 && document.activeElement.type !== 'checkbox';
  if (e.key === 'Enter' && !document.activeElement.closest('.modal')) {
    if (document.activeElement.id === 'folder') { openFolder(); e.preventDefault(); return; }
    saveNext(); e.preventDefault(); return;
  }
  if (e.ctrlKey && e.key.toLowerCase() === 'u') { doUndo(); e.preventDefault(); return; }
  if (e.altKey && e.key === 'ArrowLeft') { nav(1); e.preventDefault(); return; }
  if (e.altKey && e.key === 'ArrowRight') { nav(-1); e.preventDefault(); return; }
  if (typing) return;
  if (e.key === 'ArrowLeft') { nav(1); e.preventDefault(); }
  if (e.key === 'ArrowRight') { nav(-1); e.preventDefault(); }
  if (e.key === 'ArrowDown' || e.key === 'PageDown') { pageNav(1); e.preventDefault(); }
  if (e.key === 'ArrowUp' || e.key === 'PageUp') { pageNav(-1); e.preventDefault(); }
});

/* ---------- boot ---------- */
(async function init() {
  try {
    // extension dropdown
    const sel = $('ext');
    sel.innerHTML = `<option value="">(إبقاء الامتداد الحالي)</option>` +
      qr_exts().map(e => `<option value="${e}">${e}</option>`).join('');
    const cfg = await api('/api/config');
    TEMPLATES = cfg.templates || [];
    $('tpl-list').innerHTML = TEMPLATES.map(t => `<option value="${t}">`).join('');
    $('tpl').value = cfg.last_template || 'xx';
    $('re-find').value = cfg.re_find || '';
    $('re-repl').value = cfg.re_repl || '';
    $('re-icase').checked = !!cfg.re_icase;
    if (cfg.dark === false) { document.body.classList.add('light');
      $('theme-btn').textContent = '🌙 داكن'; }
    const home = await api('/api/home');
    $('folder').value = home.home;
    await openFolder();                 // start inside the toolkit's folder
  } catch (e) {
    status('BOOT ERROR: ' + e.message);
  }
})();
function qr_exts() { return ['.jpg', '.jpeg', '.png', '.webp', '.tif', '.bmp', '.gif', '.pdf']; }
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def run():
    """Entry point used by the file toolkit menu."""
    SESSION["folder"] = str(ftk.TARGET_DIR)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}"
    print(f"\n  Quick Renamer GUI: {url}  (Ctrl-C to stop)\n")
    threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  [Stopped] Renamer closed.\n")
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
