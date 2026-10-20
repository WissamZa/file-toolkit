#!/usr/bin/env python3
"""Local browser GUI for teaching the image toolkit.

Serves a single dark-themed RTL page from 127.0.0.1 (stdlib http.server,
no extra dependencies) and opens it in the user's browser. Three sections:

  * التعليم — browse images, drag the correct crop rectangle, label the type
    (ورقة الدخل / فاتورة / إيصال / أخرى), save teaching examples.
  * وضع العمل — one click runs the autonomous pass; every crop lands in a
    pending queue with thumbnails for the user to approve or reject.
  * ما تعلمه — learned parameters, teaching examples, reset.

Everything stays local: no cloud, no LLM, nothing leaves the machine.
"""

import io
import json
import os
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image, ImageOps

import file_toolkit as ftk
import image_tools as it

AUTO_YES = lambda prompt: "y"          # noqa: E731  (web UI confirms itself)

_THUMB_CACHE = {}
_THUMB_LOCK = threading.Lock()


# --------------------------------------------------------------------------
# Helpers shared by the endpoints
# --------------------------------------------------------------------------
def _safe_path(raw):
    """Resolve a client-supplied path and allow only files inside the
    toolkit's working folder."""
    try:
        path = Path(unquote(raw)).resolve()
    except (OSError, ValueError):
        return None
    target = Path(ftk.TARGET_DIR).resolve()
    if target not in path.parents and path != target:
        return None
    return path


def _thumbnail(path, max_px=340):
    """JPEG thumbnail bytes, cached by (path, mtime, size)."""
    path = str(path)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return b""
    key = (path, mtime, max_px)
    with _THUMB_LOCK:
        if key in _THUMB_CACHE:
            return _THUMB_CACHE[key]
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail((max_px, max_px))
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=85)
            data = buf.getvalue()
    except Exception:
        return b""
    with _THUMB_LOCK:
        if len(_THUMB_CACHE) > 300:
            _THUMB_CACHE.clear()
        _THUMB_CACHE[key] = data
    return data


def _rel(path):
    try:
        return str(Path(path).relative_to(ftk.TARGET_DIR))
    except ValueError:
        return str(path)


def _state_snapshot(conn):
    return {
        "images": len(it.collect_image_files(ftk.TARGET_DIR)),
        "examples": conn.execute(
            "SELECT COUNT(*) FROM crop_examples").fetchone()[0],
        "pending": conn.execute(
            "SELECT COUNT(*) FROM operation_log "
            "WHERE status='pending'").fetchone()[0],
        "approved": conn.execute(
            "SELECT COUNT(*) FROM operation_log "
            "WHERE status='approved'").fetchone()[0],
        "rejected": conn.execute(
            "SELECT COUNT(*) FROM operation_log "
            "WHERE status='rejected'").fetchone()[0],
        "target_dir": str(ftk.TARGET_DIR),
    }


# --------------------------------------------------------------------------
# HTTP handler
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass                                    # keep the console quiet

    # ---- plumbing ---------------------------------------------------------
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
        route = parsed.path
        query = parse_qs(parsed.query)

        if route == "/" or route == "/index.html":
            # '?shot' delays the load event until the boot fetches have
            # painted — used to take reproducible screenshots of the UI.
            page = PAGE
            if "shot" in query:
                page = page.replace(
                    "</body>", '<script src="/api/slow"></script></body>')
            self._send(200, page.encode(), "text/html; charset=utf-8")
        elif route == "/api/slow":
            import time as _time
            _time.sleep(1.0)
            self._send(200, b"", "application/javascript")
        elif route == "/img":
            raw = (query.get("path") or [""])[0]
            max_px = min(1600, int((query.get("max") or ["340"])[0]))
            path = _safe_path(raw)
            if path is None or not path.is_file():
                self._json(403, {"error": "مسار غير مسموح"})
                return
            data = _thumbnail(path, max_px)
            if not data:
                self._json(404, {"error": "تعذر قراءة الصورة"})
                return
            self._send(200, data, "image/jpeg")
        elif route == "/api/state":
            conn = it._open_db()
            try:
                self._json(200, _state_snapshot(conn))
            finally:
                conn.close()
        elif route == "/api/images":
            files = it.collect_image_files(ftk.TARGET_DIR)
            self._json(200, {
                "images": [{"path": str(p), "name": p.name} for p in files]})
        elif route == "/api/suggest":
            path = _safe_path((query.get("path") or [""])[0])
            if path is None or not path.is_file():
                self._json(403, {"error": "مسار غير مسموح"})
                return
            conn = it._open_db()
            try:
                with Image.open(path) as img:
                    img = ImageOps.exif_transpose(img)
                    plan = it.plan_crop(conn, img)
                self._json(200, {
                    "box": list(plan.box) if plan.box else None,
                    "detect_quad": plan.detect_quad,
                    "method": plan.method,
                    "type": plan.type or "other",
                    "conf": round(plan.conf, 3),
                    "warnings": plan.warnings,
                    "warning_labels": [it.WARNING_LABELS.get(w, w)
                                       for w in plan.warnings],
                    "angle": round(plan.angle, 1),
                    "reason": plan.reason,
                    "w": img.width, "h": img.height,
                    "model": it.model_info(conn),
                })
            except Exception as e:
                self._json(500, {"error": str(e)})
            finally:
                conn.close()
        elif route == "/api/pending":
            conn = it._open_db()
            try:
                rows = it.pending_operations(conn)
                pending = [{
                    "id": row[0], "source": _rel(row[3]),
                    "source_full": row[3], "destination": row[4],
                    "detail": row[5],
                    "info": it.parse_detail(row[5]),
                } for row in rows]
                self._json(200, {"ops": pending,
                                 **_state_snapshot(conn)})
            finally:
                conn.close()
        elif route == "/api/learned":
            conn = it._open_db()
            try:
                params = [{
                    "key": key, "name": name, "desc": desc,
                    "value": it.get_param(conn, key),
                    "is_default": it.get_param(conn, key)
                                  == it.DEFAULT_PARAMS[key],
                } for key, name, desc in LEARNED_PARAM_INFO]
                examples = conn.execute(
                    "SELECT type, img_w, img_h, COUNT(*) FROM crop_examples "
                    "GROUP BY type, img_w, img_h ORDER BY COUNT(*) DESC"
                ).fetchall()
                offs = it.learned_offsets(conn)
                self._json(200, {
                    "model": it.model_info(conn),
                    "params": params,
                    "offsets": None if offs is None else
                    [round(o * 100, 2) for o in offs],
                    "examples_list": [{
                        "type": it.TYPE_LABELS.get(r[0], r[0]),
                        "size": f"{r[1]}×{r[2]}", "count": r[3],
                    } for r in examples],
                    **_state_snapshot(conn)})
            finally:
                conn.close()
        else:
            self._json(404, {"error": "غير موجود"})

    # ---- POST ---------------------------------------------------------------
    def do_POST(self):
        route = urlparse(self.path).path
        data = self._body()

        if route == "/api/teach":
            path = _safe_path(data.get("path", ""))
            if path is None or not path.is_file():
                self._json(403, {"error": "مسار غير مسموح"})
                return
            box = data.get("box") or []
            type_ = data.get("type") if data.get("type") in it.TYPE_LABELS \
                else "other"
            if len(box) != 4:
                self._json(400, {"error": "مربع القص غير صالح"})
                return
            conn = it._open_db()
            try:
                with Image.open(path) as img:
                    img = ImageOps.exif_transpose(img)
                    l, t, r, b = (int(round(v)) for v in box)
                    l = max(0, min(img.width - 1, l))
                    t = max(0, min(img.height - 1, t))
                    r = max(l + 1, min(img.width, r))
                    b = max(t + 1, min(img.height, b))
                    it.save_crop_example(
                        conn, path, img, type_, (l, t, r, b),
                        low_quality=bool(data.get("low_quality")))
                    copied = None
                    if data.get("also_crop"):
                        rel = _rel(path)
                        destination = ftk.unique_destination(
                            Path(ftk.CROPPED_OUTPUT_DIR) / Path(rel).parent,
                            Path(rel).name)
                        it._save_cropped(img, (l, t, r, b), destination)
                        conn.execute(
                            "INSERT INTO operation_log (ts, run_id, op, "
                            " source, destination, detail, status) "
                            "VALUES (?, ?, 'crop', ?, ?, ?, 'approved')",
                            (datetime.now().isoformat(timespec="seconds"),
                             it._new_run_id(), str(path), str(destination),
                             "taught in web GUI"))
                        conn.commit()
                        copied = str(destination)
                threading.Thread(target=_warm_model, daemon=True).start()
                self._json(200, {"ok": True, "copied": copied,
                                 "model": it.model_info(conn),
                                 **_state_snapshot(conn)})
            except Exception as e:
                self._json(500, {"error": str(e)})
            finally:
                conn.close()
        elif route == "/api/work":
            run_id, created, skipped = it.run_autonomous_crop(
                deskew=bool(data.get("deskew")))
            conn = it._open_db()
            try:
                self._json(200, {"run_id": run_id, "created": created,
                                 "skipped": skipped,
                                 **_state_snapshot(conn)})
            finally:
                conn.close()
        elif route == "/api/evaluate":
            conn = it._open_db()
            try:
                it.get_model(conn)
                res = it.evaluate_model(conn)
                if res is None:
                    self._json(200, {"error": "يلزم 4 صور معلَّمة على الأقل "
                                              "لقياس الدقة"})
                else:
                    self._json(200, {
                        "mean_iou": round(res["mean_iou"], 3), "n": res["n"],
                        "per": [[n, round(v, 3)] for n, v in res["per"]],
                        "model": it.model_info(conn)})
            except Exception as e:
                self._json(500, {"error": str(e)})
            finally:
                conn.close()
        elif route == "/api/review":
            conn = it._open_db()
            try:
                removed = it.review_operations(
                    conn,
                    approve_ids=[int(i) for i in data.get("approve", [])],
                    reject_ids=[int(i) for i in data.get("reject", [])])
                self._json(200, {"removed": removed,
                                 **_state_snapshot(conn)})
            except Exception as e:
                self._json(500, {"error": str(e)})
            finally:
                conn.close()
        elif route == "/api/undo":
            reverted, failed = it.undo_last_operation(ask_fn=AUTO_YES)
            conn = it._open_db()
            try:
                self._json(200, {"reverted": reverted, "failed": failed,
                                 **_state_snapshot(conn)})
            finally:
                conn.close()
        elif route == "/api/reset":
            conn = it._open_db()
            try:
                it.reset_learning(conn)
                self._json(200, {"ok": True, **_state_snapshot(conn)})
            finally:
                conn.close()
        else:
            self._json(404, {"error": "غير موجود"})


def _warm_model():
    """Retrain in the background right after an example is saved, so the
    next suggestion does not have to wait for it."""
    conn = it._open_db()
    try:
        it.get_model(conn)
    except Exception:
        pass
    finally:
        conn.close()


LEARNED_PARAM_INFO = [
    ("similarity_threshold", "عتبة التشابه",
     "أقصى مسافة بصمة يُعد عندها صورتان متشابهتين (المسموح 4–20)"),
    ("crop_margin_pct", "هامش القص",
     "نسبة مئوية من ضلع الصورة القصير تُترك حول الورقة (المسموح 0–8) — "
     "لا تعتمد على أبعاد الصورة"),
    ("crop_tolerance", "تسامح الخلفية",
     "كم قد يختلف البكسل عن لون الخلفية قبل عدّه محتوى (المسموح 5–90)"),
    ("crop_deskew", "تعديل الميل",
     "1 = تعديل ميل الورقة أثناء القص، 0 = قص مستطيل فقط"),
    ("crop_min_gain", "أقل مكسب للقصّ",
     "تجاهل الصور التي لا يوفر قصّها إلا أقل من هذه النسبة (المسموح 1–30)"),
]


# --------------------------------------------------------------------------
# The single-page UI (dark, RTL)
# --------------------------------------------------------------------------
PAGE = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<title>File Toolkit — مركز تعليم الصور</title>
<style>
:root {
  --bg:#101318; --surface:#181c24; --surface2:#20252f; --border:#2c3342;
  --text:#e9ecf2; --muted:#98a2b3; --accent:#4f8cff; --accent-d:#3a6fd8;
  --ok:#3fce8f; --bad:#f0705f; --warn:#f5b04c;
}
* { box-sizing:border-box; margin:0; padding:0; }
body {
  background:var(--bg); color:var(--text); min-height:100vh;
  font-family:"Segoe UI","Noto Sans Arabic","Noto Kufi Arabic",system-ui,sans-serif;
}
header {
  display:flex; justify-content:space-between; align-items:center;
  padding:18px 28px; border-bottom:1px solid var(--border);
  background:linear-gradient(180deg,#161a22,#101318);
  position:sticky; top:0; z-index:10;
}
h1 { font-size:20px; font-weight:800; display:flex; gap:10px; align-items:center; }
h1 .logo { width:34px; height:34px; border-radius:9px; display:grid; place-items:center;
  background:linear-gradient(135deg,var(--accent),#7c5cff); font-size:17px; }
.muted { color:var(--muted); font-size:12.5px; line-height:1.7; text-align:left; }
.ltr { direction:ltr; unicode-bidi:isolate; }
nav { display:flex; gap:8px; padding:14px 28px 0; }
nav button {
  background:var(--surface); color:var(--muted); border:1px solid var(--border);
  padding:10px 22px; border-radius:10px 10px 0 0; cursor:pointer;
  font:inherit; font-weight:700; font-size:14px; transition:.15s;
}
nav button.active { background:var(--surface2); color:var(--text);
  border-bottom:2px solid var(--accent); }
main { padding:18px 28px 60px; max-width:1280px; margin:0 auto; }
section { display:none; } section.active { display:block; }
.card {
  background:var(--surface); border:1px solid var(--border);
  border-radius:14px; padding:16px 18px; margin-bottom:14px;
}
.btn {
  border:none; border-radius:9px; padding:10px 18px; cursor:pointer;
  font:inherit; font-weight:700; font-size:13.5px; color:#10131a;
  background:var(--surface2); color:var(--text); border:1px solid var(--border);
  transition:.15s;
}
.btn:hover { filter:brightness(1.15); }
.btn:disabled { opacity:.45; cursor:default; }
.btn.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
.btn.ok { background:var(--ok); border-color:var(--ok); }
.btn.bad { background:var(--bad); border-color:var(--bad); }
.row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
.spacer { flex:1; }
.big { font-size:30px; font-weight:800; }
.stats { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:14px; }
.stat { flex:1; min-width:150px; background:var(--surface); border:1px solid var(--border);
  border-radius:14px; padding:14px 18px; }
.stat .num { font-size:28px; font-weight:800; }
.stat .lbl { color:var(--muted); font-size:12.5px; margin-top:2px; }
/* teach */
.teach-grid { display:grid; grid-template-columns:1fr 300px; gap:14px; }
@media (max-width:980px){ .teach-grid { grid-template-columns:1fr; } }
.canvas-wrap { position:relative; background:#0b0d11; border:1px solid var(--border);
  border-radius:14px; overflow:hidden; min-height:420px; display:grid; place-items:center; }
#teach-img { max-width:100%; max-height:66vh; display:block; user-select:none; -webkit-user-drag:none; }
#crop-box { position:absolute; border:2px solid var(--warn); box-shadow:0 0 0 9999px rgba(5,7,10,.55);
  cursor:move; }
#crop-box.taught { border-color:var(--ok); }
#crop-box .h { position:absolute; width:12px; height:12px; background:var(--warn);
  border:2px solid #fff2; border-radius:3px; }
#crop-box .h:nth-child(1){ top:-7px; left:-7px; cursor:nwse-resize; }
#crop-box .h:nth-child(2){ top:-7px; right:-7px; cursor:nesw-resize; }
#crop-box .h:nth-child(3){ bottom:-7px; left:-7px; cursor:nesw-resize; }
#crop-box .h:nth-child(4){ bottom:-7px; right:-7px; cursor:nwse-resize; }
#crop-box.taught .h { background:var(--ok); }
.progressbar { height:6px; background:var(--surface2); border-radius:99px; overflow:hidden; }
.progressbar > div { height:100%; background:var(--accent); border-radius:99px; transition:.2s; }
.type-btn {
  display:block; width:100%; text-align:right; background:var(--surface2);
  border:1px solid var(--border); color:var(--text); border-radius:10px;
  padding:10px 14px; margin-top:8px; cursor:pointer; font:inherit; font-size:14px;
  transition:.15s;
}
.type-btn.sel { border-width:2px; font-weight:800; }
.badge { display:inline-block; padding:3px 10px; border-radius:99px; font-size:11.5px;
  font-weight:700; }
/* pending cards */
.pending-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(440px,1fr));
  gap:14px; }
.pending-card { background:var(--surface); border:1px solid var(--border);
  border-radius:14px; overflow:hidden; }
.pending-card img { width:100%; height:190px; object-fit:contain; background:#0b0d11; }
.pending-card .body { padding:12px 14px; }
.pending-card .name { font-size:13px; font-weight:700; word-break:break-all;
  direction:ltr; unicode-bidi:isolate; text-align:right; }
.pending-card .meta { color:var(--muted); font-size:12px; margin:6px 0 10px; }
.pending-card .actions { display:flex; gap:8px; }
.pending-card .actions .btn { flex:1; padding:8px 0; font-size:12.5px; }
table { width:100%; border-collapse:collapse; }
th, td { padding:10px 12px; text-align:right; border-bottom:1px solid var(--border); font-size:13.5px; }
th { color:var(--muted); font-size:12px; }
tr:hover td { background:var(--surface2); }
#toast {
  position:fixed; bottom:24px; left:50%; transform:translateX(-50%);
  background:var(--surface2); border:1px solid var(--border); color:var(--text);
  padding:12px 22px; border-radius:12px; font-size:14px; opacity:0;
  transition:.25s; pointer-events:none; z-index:99; max-width:80vw;
}
#toast.show { opacity:1; }
.empty { text-align:center; color:var(--muted); padding:40px 0; font-size:14.5px; }
#canvas-wrap { touch-action:none; }
#det-svg { position:absolute; pointer-events:none; }
#det-svg polygon { fill:none; stroke:#4cc2ff; stroke-width:2; stroke-dasharray:6 4;
  vector-effect:non-scaling-stroke; }
#after-canvas { max-width:100%; background:#0b0d11; border:1px solid var(--border);
  border-radius:10px; display:block; margin:8px auto 0; }
.pill { display:inline-block; padding:2px 9px; border-radius:99px; font-size:11.5px;
  font-weight:800; color:#fff; }
.pill.ok { background:var(--ok); } .pill.warn { background:#c9871a; }
.ba { display:grid; grid-template-columns:1fr 1fr; gap:2px; background:#0b0d11; cursor:zoom-in; }
.pane { position:relative; height:230px; }
.pane img, .pane svg { position:absolute; inset:0; width:100%; height:100%; }
.pane img { object-fit:contain; }
.pane svg polygon { fill:none; stroke-width:2.5; vector-effect:non-scaling-stroke; }
.pane .tag { position:absolute; top:6px; right:6px; background:#000a; color:#fff;
  font-size:11px; padding:2px 8px; border-radius:6px; }
.pending-card.review { border-color:#c9871a; box-shadow:0 0 0 1px #c9871a55; }
.pending-card .warns { color:#f2b24d; font-size:12px; margin-bottom:8px; line-height:1.7; }
.chips { display:flex; gap:8px; margin:12px 0 0; }
.chip { background:var(--surface2); border:1px solid var(--border); color:var(--text);
  border-radius:99px; padding:5px 14px; cursor:pointer; font:inherit; font-size:12.5px; }
.chip.sel { background:var(--accent); border-color:var(--accent); color:#fff; }
#modal { position:fixed; inset:0; background:#000d; z-index:120; display:grid;
  place-items:center; padding:20px; }
#modal[hidden] { display:none; }
#modal .box { background:var(--surface); border:1px solid var(--border); border-radius:14px;
  padding:14px; max-width:96vw; max-height:94vh; overflow:auto; }
#modal .big2 { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
#modal .pane { height:min(70vh,640px); background:#0b0d11; border-radius:8px; }
@media (max-width:900px){ #modal .big2 { grid-template-columns:1fr; } .ba { grid-template-columns:1fr; } }
footer { text-align:center; color:var(--muted); font-size:12px; padding:20px; }
</style>
</head>
<body>
<header>
  <div>
    <h1><span class="logo">🗂️</span> مركز تعليم الصور</h1>
    <div class="muted" id="paths"></div>
  </div>
  <div class="muted" id="statusbar" style="text-align:center; font-size:13.5px;"></div>
</header>
<nav>
  <button data-tab="teach" class="active">✏️ التعليم</button>
  <button data-tab="work">⚙️ وضع العمل</button>
  <button data-tab="learned">🧠 ما تعلمه</button>
</nav>
<main>
  <section id="tab-teach" class="active">
    <div class="card">
      <div class="row">
        <span id="t-progress" style="font-weight:800;"></span>
        <div class="progressbar" style="flex:1; min-width:160px;"><div id="t-bar"></div></div>
        <span class="spacer"></span>
        <button class="btn" onclick="loadTeach(true)">تحديث القائمة</button>
      </div>
    </div>
    <div class="teach-grid">
      <div>
        <div class="canvas-wrap" id="canvas-wrap">
          <img id="teach-img" alt="">
          <svg id="det-svg" hidden></svg>
          <div id="crop-box" hidden>
            <div class="h" data-h="tl"></div><div class="h" data-h="tr"></div>
            <div class="h" data-h="bl"></div><div class="h" data-h="br"></div>
          </div>
        </div>
        <div class="card" style="margin-top:14px;">
          <div class="row">
            <span id="t-name" style="font-weight:800;"></span>
            <span class="muted" id="t-meta"></span>
            <span class="spacer"></span>
            <span id="t-conf" class="pill"></span>
            <span id="t-method" class="badge"></span>
          </div>
          <div id="t-model" class="muted" style="margin-top:8px; font-size:12.5px; line-height:1.8;"></div>
          <div id="t-warn" class="muted" style="margin-top:8px; color:#f2b24d; font-size:12.5px; line-height:1.8;"></div>
          <div class="row" style="margin-top:8px;">
            <button class="btn" onclick="resetToSuggestion()">↺ العودة للاقتراح التلقائي</button>
            <label style="cursor:pointer; font-size:13px;"><input type="checkbox" id="t-showdet" checked
              onchange="drawBox()"> أظهر الكشف التلقائي (خط أزرق متقطّع)</label>
          </div>
        </div>
      </div>
      <div>
        <div class="card">
          <div class="muted" style="margin-bottom:4px;">نوع الصورة</div>
          <div id="type-btns"></div>
          <label class="row" style="margin-top:12px; cursor:pointer; font-size:13.5px;">
            <input type="checkbox" id="t-lowq"> جودة منخفضة
          </label>
        </div>
        <div class="card">
          <div class="muted">معاينة ما بعد القص (تتحدّث أثناء السحب)</div>
          <canvas id="after-canvas" width="10" height="10"></canvas>
          <div id="after-info" class="muted" style="text-align:center; margin-top:6px; font-size:12px;"></div>
        </div>
        <div class="card">
          <button class="btn primary" style="width:100%" onclick="teachSave(false)">
            💾 حفظ المثال والتالي (Enter)</button>
          <button class="btn ok" style="width:100%; margin-top:8px;" onclick="teachSave(true)">
            ✂️ حفظ + قصّ النسخة الآن</button>
          <div class="row" style="margin-top:8px;">
            <button class="btn" style="flex:1" onclick="teachNav(1)">← التالي / تخطي</button>
            <button class="btn" style="flex:1" onclick="teachNav(-1)">السابق →</button>
          </div>
          <div class="muted" style="margin-top:10px; font-size:12.5px; line-height:1.8;">
            اسحب على الصورة لرسم مربع القص، أو اسحب من داخله لتحريكه، ومن الزوايا لتغيير حجمه.
          </div>
        </div>
      </div>
    </div>
  </section>

  <section id="tab-work">
    <div class="stats">
      <div class="stat"><div class="num" id="w-pending" style="color:var(--warn)">0</div>
        <div class="lbl">بانتظار اعتمادك</div></div>
      <div class="stat"><div class="num" id="w-approved" style="color:var(--ok)">0</div>
        <div class="lbl">معتمدة</div></div>
      <div class="stat"><div class="num" id="w-rejected" style="color:var(--bad)">0</div>
        <div class="lbl">مرفوضة (نُسخها حُذفت)</div></div>
    </div>
    <div class="card">
      <div class="row">
        <button class="btn primary" onclick="runWork()">▶ تشغيل المعالجة الذاتية</button>
        <button class="btn ok" onclick="approveConfident()">✓ اعتماد الموثوق فقط</button>
        <button class="btn" onclick="approveAll()">اعتماد الكل</button>
        <button class="btn bad" onclick="undoLast()">↩︎ التراجع عن آخر دفعة</button>
        <span class="spacer"></span>
        <button class="btn" onclick="loadPending()">تحديث</button>
      </div>
      <div class="muted" style="margin-top:8px;">
        المعالجة الذاتية تكشف الورقة في كل صورة مهما كانت أبعادها، وكل نتيجة تظهر هنا
        <b>قبل</b> (الإطار = حدود القص) و<b>بعد</b> بانتظار قرارك. الإطار البرتقالي = ثقة منخفضة،
        راجعه. الأصول لا تُلمس أبداً.
      </div>
      <label style="display:block; margin-top:10px; cursor:pointer; font-size:13.5px;">
        <input type="checkbox" id="w-deskew"> تعديل ميل الورقة (تدوير الورقة المائلة لتصبح مستقيمة)
      </label>
      <div class="chips">
        <button class="chip sel" id="f-all" onclick="setFilter('all')">الكل</button>
        <button class="chip" id="f-review" onclick="setFilter('review')">⚠ تحتاج مراجعة</button>
      </div>
      <div id="skipped-box" class="muted" style="margin-top:10px; font-size:12.5px; line-height:1.9;"></div>
    </div>
    <div id="pending-list" class="pending-grid"></div>
  </section>

  <section id="tab-learned">
    <div class="card">
      <div class="row">
        <span style="font-weight:800;">🧠 نموذج تحديد الفاتورة</span>
        <span class="spacer"></span>
        <button class="btn" id="eval-btn" onclick="evaluateModel()">قياس الدقة (اختبار على صور لم يرها)</button>
      </div>
      <div id="model-info" style="margin-top:10px; line-height:1.9;"></div>
      <div id="eval-result" class="muted" style="margin-top:6px; font-size:12.5px; line-height:1.9;"></div>
    </div>
    <div class="card">
      <div class="row">
        <span style="font-weight:800;">المعاملات المتعلمة</span>
        <span class="spacer"></span>
        <button class="btn bad" onclick="resetLearning()">تصفير التعلم</button>
      </div>
      <table id="params-table">
        <thead><tr><th>المعامل</th><th>القيمة</th><th>الشرح</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
    <div class="card">
      <span style="font-weight:800;">أمثلة القصّ المعلَّمة</span>
      <table id="examples-table">
        <thead><tr><th>النوع</th><th>أبعاد الصورة</th><th>عدد الأمثلة</th></tr></thead>
        <tbody></tbody>
      </table>
      <div class="muted" style="margin-top:8px;">
        التعليم لا يرتبط بأبعاد الصورة: يُحفظ كل مثال بنِسَب نسبية ويُقارَن بما كشفه البرنامج،
        فيتعلّم <b>كم تُحبّ أن يكون الهامش حول الورقة</b> ويطبّقه على أي صورة بأي حجم.
        <div id="offsets-info" style="margin-top:8px;"></div>
      </div>
    </div>
  </section>
</main>
<footer>يعمل محلياً بالكامل على جهازك — بلا سحابة ولا نماذج خارجية</footer>
<div id="modal" hidden onclick="this.hidden=true"><div class="box" onclick="event.stopPropagation()">
  <div class="row" style="margin-bottom:8px;"><b id="m-title"></b><span class="spacer"></span>
    <button class="btn" onclick="$('modal').hidden=true">إغلاق ✕</button></div>
  <div id="m-body" class="big2"></div></div></div>
<div id="toast"></div>
<script>
let DETQUAD = null, OPS = [], FILTER = 'all', IMAGES = [], IDX = 0, BOX = null, IMGW = 0, IMGH = 0, METHOD = '', SUGGESTED = null, TYPES = {}, SEL_TYPE = 'other';
const $ = id => document.getElementById(id);

function toast(msg) {
  const t = $('toast'); t.textContent = msg; t.classList.add('show');
  clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove('show'), 2600);
}
async function api(path, body) {
  const res = await fetch(path, body ? {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)} : {});
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data;
}
function status(s) {
  $('statusbar').innerHTML =
    `صور في مجلد العمل: <b>${s.images}</b> &nbsp;•&nbsp; أمثلة معلَّمة: <b>${s.examples}</b> &nbsp;•&nbsp; ` +
    `بانتظار الاعتماد: <b style="color:var(--warn)">${s.pending}</b>`;
  $('w-pending').textContent = s.pending;
  $('w-approved').textContent = s.approved;
  $('w-rejected').textContent = s.rejected;
  if (!$('paths').textContent)
    $('paths').innerHTML = 'مجلد العمل: <span class="ltr">' + s.target_dir + '</span>';
}

/* ---------- tabs ---------- */
document.querySelectorAll('nav button').forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('section').forEach(s => s.classList.remove('active'));
    btn.classList.add('active');
    $('tab-' + btn.dataset.tab).classList.add('active');
    if (btn.dataset.tab === 'work') loadPending();
    if (btn.dataset.tab === 'learned') loadLearned();
  };
});

/* ---------- teach ---------- */
function buildTypeButtons(selected) {
  SEL_TYPE = selected;
  const wrap = $('type-btns'); wrap.innerHTML = '';
  for (const [val, label] of Object.entries(TYPES)) {
    const b = document.createElement('button');
    b.className = 'type-btn' + (val === selected ? ' sel' : '');
    b.style.borderColor = val === selected ? TYPES[val].color : '';
    b.innerHTML = (val === selected ? '● ' : '○ ') + label.ar;
    b.onclick = () => buildTypeButtons(val);
    wrap.appendChild(b);
  }
}
async function loadTeach(refresh) {
  if (refresh || !IMAGES.length) {
    const d = await api('/api/images'); IMAGES = d.images;
  }
  if (!IMAGES.length) {
    $('t-name').textContent = 'لا توجد صور';
    $('t-meta').textContent = 'ضع صوراً في مجلد العمل ثم حدّث القائمة';
    $('teach-img').hidden = true; $('crop-box').hidden = true;
    return;
  }
  $('teach-img').hidden = false;
  IDX = Math.min(IDX, IMAGES.length - 1);
  const img = IMAGES[IDX];
  $('teach-img').onload = drawBox;
  $('teach-img').src = '/img?max=900&path=' + encodeURIComponent(img.path);
  $('t-name').textContent = img.name;
  $('t-progress').textContent = `صورة ${IDX + 1} من ${IMAGES.length}`;
  $('t-bar').style.width = ((IDX + 1) / IMAGES.length * 100) + '%';
  const sug = await api('/api/suggest?path=' + encodeURIComponent(img.path));
  IMGW = sug.w; IMGH = sug.h; METHOD = sug.method;
  SUGGESTED = sug.box; DETQUAD = sug.detect_quad;
  BOX = sug.box ? [...sug.box] : [Math.round(sug.w*.1), Math.round(sug.h*.1),
                                  Math.round(sug.w*.9), Math.round(sug.h*.9)];
  buildTypeButtons(sug.type || 'other');
  const badge = $('t-method'), pill = $('t-conf');
  const names = { learned: 'نموذج متعلَّم ✓', taught: 'قصّ متعلَّم ✓', calibrated: 'كشف + تعلّم ✓', auto: 'كشف تلقائي', prior: 'من أمثلة معلَّمة' };
  badge.textContent = 'الاقتراح: ' + (names[sug.method] || 'يدوي (لم تُكشف ورقة)');
  badge.style.background = (sug.method === 'learned' || sug.method === 'taught' || sug.method === 'calibrated') ? 'var(--ok)' : 'var(--warn)';
  pill.hidden = !sug.box;
  pill.textContent = 'الثقة ' + Math.round(sug.conf * 100) + '%';
  pill.className = 'pill ' + (sug.conf >= 0.70 ? 'ok' : 'warn');
  const msgs = (sug.warning_labels || []).slice();
  if (!sug.box && sug.reason) msgs.push(sug.reason + ' — ارسم المربع بنفسك');
  $('t-warn').textContent = msgs.length ? '⚠ ' + msgs.join(' • ') : '';
  $('t-meta').textContent = sug.w + '×' + sug.h;
  showModel(sug.model);
  drawBox();
}
function resetToSuggestion() {
  if (!SUGGESTED) return toast('لا يوجد اقتراح تلقائي لهذه الصورة');
  BOX = [...SUGGESTED]; METHOD = 'auto'; drawBox();
}
function imgScale() {
  const el = $('teach-img');
  return el.clientWidth / IMGW;
}
function drawBox() {
  const box = $('crop-box'), img = $('teach-img');
  if (!BOX || img.hidden) return;
  box.hidden = false;
  box.classList.toggle('taught', METHOD === 'learned' || METHOD === 'taught' || METHOD === 'calibrated');
  const wrap = $('canvas-wrap').getBoundingClientRect();
  const ir = img.getBoundingClientRect();
  const offX = ir.left - wrap.left, offY = ir.top - wrap.top;
  const s = imgScale();
  box.style.left = (offX + BOX[0]*s) + 'px';
  box.style.top = (offY + BOX[1]*s) + 'px';
  box.style.width = ((BOX[2]-BOX[0])*s) + 'px';
  box.style.height = ((BOX[3]-BOX[1])*s) + 'px';
  const svg = $('det-svg');
  if (DETQUAD && $('t-showdet').checked) {
    svg.hidden = false;
    svg.style.left = offX + 'px'; svg.style.top = offY + 'px';
    svg.style.width = ir.width + 'px'; svg.style.height = ir.height + 'px';
    svg.setAttribute('viewBox', '0 0 ' + IMGW + ' ' + IMGH);
    svg.innerHTML = '<polygon points="' + DETQUAD.map(p => p[0] + ',' + p[1]).join(' ') + '"/>';
  } else svg.hidden = true;
  drawAfter();
}
function drawAfter() {
  const img = $('teach-img'), c = $('after-canvas');
  if (!BOX || !img.naturalWidth) return;
  const k = img.naturalWidth / IMGW;
  const sw = Math.max(1, (BOX[2]-BOX[0]) * k), sh = Math.max(1, (BOX[3]-BOX[1]) * k);
  const sc = Math.min(280 / sw, 260 / sh);
  c.width = Math.max(1, Math.round(sw * sc)); c.height = Math.max(1, Math.round(sh * sc));
  c.getContext('2d').drawImage(img, BOX[0]*k, BOX[1]*k, sw, sh, 0, 0, c.width, c.height);
  const pct = Math.round((BOX[2]-BOX[0]) * (BOX[3]-BOX[1]) / (IMGW * IMGH) * 100);
  $('after-info').textContent = Math.round(BOX[2]-BOX[0]) + '×' + Math.round(BOX[3]-BOX[1]) +
    ' من ' + IMGW + '×' + IMGH + ' (' + pct + '% من الصورة)';
}
function clampBox() {
  BOX[0] = Math.max(0, Math.min(IMGW-2, BOX[0]));
  BOX[1] = Math.max(0, Math.min(IMGH-2, BOX[1]));
  BOX[2] = Math.max(BOX[0]+1, Math.min(IMGW, BOX[2]));
  BOX[3] = Math.max(BOX[1]+1, Math.min(IMGH, BOX[3]));
}
let drag = null;
$('canvas-wrap').addEventListener('pointerdown', e => {
  if (!$('teach-img') || $('teach-img').hidden) return;
  const wrap = $('canvas-wrap').getBoundingClientRect();
  const ir = $('teach-img').getBoundingClientRect();
  const x = (e.clientX - ir.left) / imgScale(), y = (e.clientY - ir.top) / imgScale();
  const inBox = BOX && x >= BOX[0] && x <= BOX[2] && y >= BOX[1] && y <= BOX[3];
  const handle = e.target.dataset && e.target.dataset.h;
  if (handle && BOX) {
    drag = { mode:'resize', h:handle, fix:{x: handle.includes('l') ? BOX[2] : BOX[0],
                                            y: handle.includes('t') ? BOX[3] : BOX[1]} };
  } else if (inBox) {
    drag = { mode:'move', dx:x-BOX[0], dy:y-BOX[1] };
  } else {
    drag = { mode:'draw', sx:x, sy:y };
    BOX = [x, y, x, y];
  }
  METHOD = 'manual'; drawBox();
  e.preventDefault();
});
window.addEventListener('pointermove', e => {
  if (!drag) return;
  const ir = $('teach-img').getBoundingClientRect();
  let x = (e.clientX - ir.left) / imgScale(), y = (e.clientY - ir.top) / imgScale();
  x = Math.max(0, Math.min(IMGW, x)); y = Math.max(0, Math.min(IMGH, y));
  if (drag.mode === 'draw') BOX = [Math.min(drag.sx,x), Math.min(drag.sy,y), Math.max(drag.sx,x), Math.max(drag.sy,y)];
  else if (drag.mode === 'move') {
    const w = BOX[2]-BOX[0], h = BOX[3]-BOX[1];
    let l = Math.max(0, Math.min(IMGW-w, x-drag.dx)), t = Math.max(0, Math.min(IMGH-h, y-drag.dy));
    BOX = [l, t, l+w, t+h];
  } else {
    const fx = drag.fix.x, fy = drag.fix.y;
    BOX = [Math.min(fx,x), Math.min(fy,y), Math.max(fx,x), Math.max(fy,y)];
  }
  drawBox();
});
window.addEventListener('pointerup', () => { if (drag) { drag = null; clampBox(); drawBox(); } });
window.addEventListener('resize', drawBox);
async function teachSave(alsoCrop) {
  if (!BOX) return;
  clampBox();
  const d = await api('/api/teach', { path: IMAGES[IDX].path, box: BOX,
    type: SEL_TYPE, low_quality: $('t-lowq').checked, also_crop: alsoCrop });
  status(d);
  toast(alsoCrop ? 'حُفظ المثال وقُصّت النسخة ✓' : 'حُفظ المثال التعليمي ✓');
  teachNav(1);
}
function teachNav(step) {
  IDX = Math.max(0, Math.min(IMAGES.length-1, IDX + step));
  loadTeach(false);
}
window.addEventListener('keydown', e => {
  if (e.key === 'Escape') $('modal').hidden = true;
  if (!$('tab-teach').classList.contains('active')) return;
  if (e.key === 'Enter') teachSave(false);
  if (e.key === 'ArrowLeft') teachNav(1);
  if (e.key === 'ArrowRight') teachNav(-1);
});

/* ---------- work ---------- */
function isReview(op) {
  const i = op.info || {};
  if (i.conf === null || i.conf === undefined) return false;
  const w = i.warnings || [];
  return i.conf < 0.70 || w.includes('weak_prior') || w.includes('prior_disagree');
}
const WLAB = {
  low_confidence: 'ثقة منخفضة — راجع القص', small_document: 'الورقة صغيرة في الصورة',
  full_frame: 'الورقة تملأ الصورة', touches_edge: 'الورقة قريبة من الحافة / قد تكون مقطوعة',
  tilted: 'الورقة مائلة', weak_prior: 'اعتُمد مثال معلَّم فقط (لم تُكشف ورقة)',
  prior_disagree: 'القص المعلَّم يخالف الكشف التلقائي'
};
function setFilter(f) {
  FILTER = f;
  $('f-all').classList.toggle('sel', f === 'all');
  $('f-review').classList.toggle('sel', f === 'review');
  renderPending();
}
function paneHTML(op, big) {
  const i = op.info || {}, src = i.src || [0, 0];
  const color = isReview(op) ? '#f5aa32' : '#3ddc84';
  const q = i.quad;
  const poly = (q && src[0]) ? '<svg viewBox="0 0 ' + src[0] + ' ' + src[1] +
    '" preserveAspectRatio="xMidYMid meet"><polygon stroke="' + color + '" points="' +
    q.map(p => p[0] + ',' + p[1]).join(' ') + '"/></svg>' : '';
  const mx = big ? 1400 : 420;
  return '<div class="pane"><img src="/img?max=' + mx + '&path=' + encodeURIComponent(op.source_full) +
    '">' + poly + '<span class="tag">قبل</span></div>' +
    '<div class="pane"><img src="/img?max=' + mx + '&path=' + encodeURIComponent(op.destination) +
    '"><span class="tag">بعد</span></div>';
}
function openModal(op) {
  const i = op.info || {};
  $('m-title').innerHTML = op.source + (i.src ? '   <span class="ltr">' + i.src[0] + '×' + i.src[1] + ' → ' + i.out[0] + '×' + i.out[1] + '</span>' : '');
  $('m-body').innerHTML = paneHTML(op, true);
  $('modal').hidden = false;
}
function renderPending() {
  const list = $('pending-list'); list.innerHTML = '';
  let ops = OPS.slice();
  ops.sort((a, b) => (isReview(b) - isReview(a)) ||
    ((a.info.conf === null ? 1 : a.info.conf) - (b.info.conf === null ? 1 : b.info.conf)));
  if (FILTER === 'review') ops = ops.filter(isReview);
  if (!ops.length) {
    list.innerHTML = '<div class="empty card" style="grid-column:1/-1">' +
      (OPS.length ? 'لا توجد نتائج تحتاج مراجعة 👍' :
       'لا توجد عمليات بانتظار الاعتماد — شغّل المعالجة الذاتية أولاً') + '</div>';
    return;
  }
  const methodNames = { learned: 'نموذج متعلَّم ✓', taught: 'قصّ متعلَّم ✓', calibrated: 'كشف + تعلّم ✓', auto: 'كشف تلقائي',
                        prior: 'من أمثلة معلَّمة', legacy: 'قديم' };
  for (const op of ops) {
    const i = op.info || {};
    const type = TYPES[i.type] ? TYPES[i.type].ar : 'غير معروف';
    const rev = isReview(op);
    const conf = (i.conf === null || i.conf === undefined) ? '' :
      '<span class="pill ' + (rev ? 'warn' : 'ok') + '">ثقة ' + Math.round(i.conf * 100) + '%</span> ';
    const dims = i.src ? '<span class="ltr">' + i.src[0] + '×' + i.src[1] + ' → ' + i.out[0] + '×' + i.out[1] + '</span> • ' : '';
    const warns = (i.warnings || []).map(w => WLAB[w] || w);
    const card = document.createElement('div');
    card.className = 'pending-card' + (rev ? ' review' : '');
    card.innerHTML =
      '<div class="ba" title="اضغط للتكبير">' + paneHTML(op, false) + '</div>' +
      '<div class="body"><div class="name">' + op.source + '</div>' +
      '<div class="meta">' + conf + dims + (methodNames[i.method] || i.method || '') + ' • ' + type +
      (i.deskew ? ' • عُدِّل الميل' : '') + '</div>' +
      (warns.length ? '<div class="warns">⚠ ' + warns.join(' • ') + '</div>' : '') +
      '<div class="actions"><button class="btn ok">✓ اعتماد</button>' +
      '<button class="btn bad">✗ رفض</button></div></div>';
    card.querySelector('.ba').onclick = () => openModal(op);
    const [okB, badB] = card.querySelectorAll('button');
    okB.onclick = () => review([op.id], []);
    badB.onclick = () => review([], [op.id]);
    list.appendChild(card);
  }
}
async function loadPending() {
  const d = await api('/api/pending');
  status(d);
  OPS = d.ops || [];
  renderPending();
}
async function review(approve, reject) {
  const d = await api('/api/review', { approve, reject });
  status(d);
  toast(reject.length ? 'رُفضت ' + reject.length + ' وحُذفت نسخها (' + d.removed + ')' : 'اعتُمدت ' + approve.length + ' ✓');
  loadPending();
}
async function approveIds(ids, label) {
  if (!ids.length) return toast('لا يوجد ما يُعتمد');
  if (!confirm('اعتماد ' + ids.length + ' ' + label + '؟')) return;
  const r = await api('/api/review', { approve: ids, reject: [] });
  status(r); toast('اعتُمدت ' + ids.length + ' عملية ✓');
  loadPending();
}
function approveAll() { approveIds(OPS.map(p => p.id), 'عملية معلّقة'); }
function approveConfident() {
  const ids = OPS.filter(p => !isReview(p)).map(p => p.id);
  const left = OPS.length - ids.length;
  approveIds(ids, 'نتيجة موثوقة' + (left ? ' (ستبقى ' + left + ' للمراجعة)' : ''));
}
async function runWork() {
  toast('جارٍ المعالجة الذاتية…');
  const d = await api('/api/work', { deskew: $('w-deskew').checked });
  status(d);
  const sk = d.skipped || [];
  $('skipped-box').innerHTML = sk.length ? '<b>تُخطّيت ' + sk.length + ':</b><br>' +
    sk.slice(0, 20).map(x => '<span class="ltr">' + x[0] + '</span> — ' + x[1]).join('<br>') : '';
  toast('أُنشئت ' + d.created + ' نسخة (' + sk.length + ' تخطّت) — راجع قبل/بعد ثم اعتمد');
  loadPending();
}
async function undoLast() {
  if (!confirm('التراجع عن آخر دفعة؟ تعود الصور المنقولة لمكانها وتُحذف النسخ المقصوصة')) return;
  const d = await api('/api/undo', {});
  status(d);
  toast(`تم التراجع عن ${d.reverted} عملية`);
  loadPending();
}

function modelText(m) {
  if (!m) return '';
  if (m.n < m.needed)
    return 'النموذج: لم يبدأ بعد — علّمه ' + (m.needed - m.n) + ' صور أخرى على الأقل (علّمتَ ' + m.n +
      '). حتى ذلك الحين يُستخدم الكشف التلقائي العام.';
  const acc = (m.cv_iou === null || m.cv_iou === undefined) ? 'الدقة لم تُقَس بعد' :
    'دقته التقديرية ' + Math.round(m.cv_iou * 100) + '%';
  return 'النموذج يتعلّم من ' + m.n + ' صورة معلَّمة — ' + acc +
    '. كل صورة تصحّحها وتحفظها تحسّنه.';
}
function showModel(m) { $('t-model').textContent = modelText(m); }
async function evaluateModel() {
  const btn = $('eval-btn'); btn.disabled = true; btn.textContent = 'جارٍ القياس… (قد يستغرق دقيقة)';
  try {
    const d = await api('/api/evaluate', {});
    if (d.error) { toast(d.error); }
    else {
      $('eval-result').innerHTML = '<b>التطابق المتوسط مع صناديقك على صور لم يرها النموذج: ' +
        Math.round(d.mean_iou * 100) + '%</b> (100% = مطابق تماماً؛ فوق 85% يعني أن القص يطابق ما ترسمه تقريباً)<br>' +
        d.per.map(x => '<span class="ltr">' + x[0] + '</span> ' + Math.round(x[1] * 100) + '%').join(' • ');
      $('model-info').textContent = modelText(d.model);
    }
  } finally { btn.disabled = false; btn.textContent = 'قياس الدقة (اختبار على صور لم يرها)'; }
}

/* ---------- learned ---------- */
async function loadLearned() {
  const d = await api('/api/learned');
  status(d);
  const pb = $('params-table').querySelector('tbody'); pb.innerHTML = '';
  for (const p of d.params) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${p.name}</td>
      <td><b style="color:${p.is_default ? 'var(--muted)' : 'var(--ok)'}">${p.value}</b>
          ${p.is_default ? '<span class="muted"> (افتراضي)</span>' : ''}</td>
      <td class="muted">${p.desc}</td>`;
    pb.appendChild(tr);
  }
  $('model-info').textContent = modelText(d.model);
  const o = d.offsets;
  $('offsets-info').innerHTML = o ? ('الهامش المتعلَّم حول الورقة (% من ضلع الصورة القصير): يسار <b>' + o[0] +
    '</b> • أعلى <b>' + o[1] + '</b> • يمين <b>' + o[2] + '</b> • أسفل <b>' + o[3] + '</b>') :
    '<span class="muted">يحتاج مثالين على الأقل بكشف واثق ليتعلّم الهامش (حتى ذلك الحين يُستخدم هامش القص الافتراضي).</span>';
  const eb = $('examples-table').querySelector('tbody'); eb.innerHTML = '';
  const examples = d.examples_list || [];
  if (!examples.length)
    eb.innerHTML = '<tr><td colspan="3" class="empty">لا شيء بعد — علّم البرنامج من تبويب التعليم</td></tr>';
  for (const ex of examples) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${ex.type}</td><td class="ltr">${ex.size}</td><td>${ex.count}</td>`;
    eb.appendChild(tr);
  }
}
async function resetLearning() {
  if (!confirm('تصفير كل المعاملات المتعلمة؟ (الأمثلة وسجل العمليات لن تُمس)')) return;
  const d = await api('/api/reset', {});
  status(d); toast('صُفّر التعلم');
  loadLearned();
}

/* ---------- boot ---------- */
(async function init() {
  const d = await api('/api/learned');
  status(d);
  TYPES = {
    income: { ar: 'ورقة الدخل', color: 'var(--accent)' },
    invoice: { ar: 'فاتورة', color: 'var(--warn)' },
    receipt: { ar: 'إيصال', color: 'var(--ok)' },
    other: { ar: 'أخرى', color: 'var(--muted)' },
  };
  buildTypeButtons('other');
  await loadTeach(true);
  const hash = location.hash.slice(1);
  if (['teach', 'work', 'learned'].includes(hash))
    document.querySelector(`nav button[data-tab="${hash}"]`).click();
})();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def run():
    """Entry point used by the file toolkit menu: serve on 127.0.0.1 and
    open the browser."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}"
    print(f"\n  Image teaching GUI: {url}  (Ctrl-C to stop)\n")
    threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  [Stopped] GUI server closed.\n")
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
