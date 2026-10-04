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
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
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
                    box, method, type_ = it.learned_crop(conn, img)
                self._json(200, {
                    "box": list(box) if box else None,
                    "method": method,
                    "type": type_ or "other",
                    "w": img.width, "h": img.height,
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
                self._json(200, {
                    "params": params,
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
                        conn, path, img.size, type_, (l, t, r, b),
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
                self._json(200, {"ok": True, "copied": copied,
                                 **_state_snapshot(conn)})
            except Exception as e:
                self._json(500, {"error": str(e)})
            finally:
                conn.close()
        elif route == "/api/work":
            run_id, created, skipped = it.run_autonomous_crop()
            conn = it._open_db()
            try:
                self._json(200, {"run_id": run_id, "created": created,
                                 "skipped": skipped,
                                 **_state_snapshot(conn)})
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


LEARNED_PARAM_INFO = [
    ("similarity_threshold", "عتبة التشابه",
     "أقصى مسافة بصمة يُعد عندها صورتان متشابهتين (المسموح 4–20)"),
    ("crop_margin", "هامش القص",
     "بكسلات تُترك حول المحتوى عند القصّ التلقائي (المسموح 0–120)"),
    ("crop_tolerance", "تسامح الخلفية",
     "كم قد يختلف البكسل عن لون الخلفية قبل عدّه محتوى (المسموح 5–90)"),
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
.pending-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(270px,1fr));
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
            <span id="t-method" class="badge"></span>
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
        <button class="btn ok" onclick="approveAll()">✓ اعتماد كل المعلّق</button>
        <button class="btn bad" onclick="undoLast()">↩︎ التراجع عن آخر دفعة</button>
        <span class="spacer"></span>
        <button class="btn" onclick="loadPending()">تحديث</button>
      </div>
      <div class="muted" style="margin-top:8px;">
        المعالجة الذاتية تقصّ كل الصور بما تعلمه البرنامج، وكل نسخة تظهر هنا بانتظار قرارك —
        الاعتماد يبقيها، والرفض يحذفها. الأصول لا تُلمس أبداً.
      </div>
    </div>
    <div id="pending-list" class="pending-grid"></div>
  </section>

  <section id="tab-learned">
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
        كلما علّمت البرنامج صورة من تبويب التعليم، تُطبَّق نفس نِسَب القصّ تلقائياً على كل
        الصور القادمة بنفس الأبعاد (صور من نفس المصدر عادة).
      </div>
    </div>
  </section>
</main>
<footer>يعمل محلياً بالكامل على جهازك — بلا سحابة ولا نماذج خارجية</footer>
<div id="toast"></div>
<script>
let IMAGES = [], IDX = 0, BOX = null, IMGW = 0, IMGH = 0, METHOD = '', SUGGESTED = null, TYPES = {}, SEL_TYPE = 'other';
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
async function initTypes() {
  const res = await fetch('/api/learned');
  // types come from the backend labels embedded below
}
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
  $('teach-img').src = '/img?max=900&path=' + encodeURIComponent(img.path);
  $('t-name').textContent = img.name;
  $('t-progress').textContent = `صورة ${IDX + 1} من ${IMAGES.length}`;
  $('t-bar').style.width = ((IDX + 1) / IMAGES.length * 100) + '%';
  const sug = await api('/api/suggest?path=' + encodeURIComponent(img.path));
  IMGW = sug.w; IMGH = sug.h; METHOD = sug.method;
  SUGGESTED = sug.box;
  BOX = sug.box ? [...sug.box] : [Math.round(sug.w*.1), Math.round(sug.h*.1),
                                  Math.round(sug.w*.9), Math.round(sug.h*.9)];
  buildTypeButtons(sug.type || 'other');
  const badge = $('t-method');
  if (sug.method === 'taught') {
    badge.textContent = 'الاقتراح: قصّ متعلَّم ✓'; badge.style.background = 'var(--ok)';
  } else {
    badge.textContent = 'الاقتراح: كشف تلقائي'; badge.style.background = 'var(--warn)';
  }
  $('teach-img').onload = drawBox;
  drawBox();
}
function imgScale() {
  const el = $('teach-img');
  return el.clientWidth / IMGW;
}
function drawBox() {
  const box = $('crop-box'), img = $('teach-img');
  if (!BOX || img.hidden) return;
  box.hidden = false;
  box.classList.toggle('taught', METHOD === 'taught');
  const wrap = $('canvas-wrap').getBoundingClientRect();
  const ir = img.getBoundingClientRect();
  const offX = ir.left - wrap.left, offY = ir.top - wrap.top;
  const s = imgScale();
  box.style.left = (offX + BOX[0]*s) + 'px';
  box.style.top = (offY + BOX[1]*s) + 'px';
  box.style.width = ((BOX[2]-BOX[0])*s) + 'px';
  box.style.height = ((BOX[3]-BOX[1])*s) + 'px';
}
function clampBox() {
  BOX[0] = Math.max(0, Math.min(IMGW-2, BOX[0]));
  BOX[1] = Math.max(0, Math.min(IMGH-2, BOX[1]));
  BOX[2] = Math.max(BOX[0]+1, Math.min(IMGW, BOX[2]));
  BOX[3] = Math.max(BOX[1]+1, Math.min(IMGH, BOX[3]));
}
let drag = null;
$('canvas-wrap').addEventListener('mousedown', e => {
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
window.addEventListener('mousemove', e => {
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
window.addEventListener('mouseup', () => { if (drag) { drag = null; clampBox(); drawBox(); } });
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
  if (!$('tab-teach').classList.contains('active')) return;
  if (e.key === 'Enter') teachSave(false);
  if (e.key === 'ArrowLeft') teachNav(1);
  if (e.key === 'ArrowRight') teachNav(-1);
});

/* ---------- work ---------- */
async function loadPending() {
  const d = await api('/api/pending');
  status(d);
  const ops = d.ops || [];
  const list = $('pending-list'); list.innerHTML = '';
  if (!ops.length) {
    list.innerHTML = '<div class="empty card" style="grid-column:1/-1">' +
      'لا توجد عمليات بانتظار الاعتماد — شغّل المعالجة الذاتية أولاً</div>';
    return;
  }
  for (const op of ops) {
    const parts = {};
    (op.detail || '').split(',').forEach(p => {
      const [k, v] = p.trim().split('='); parts[k] = v; });
    const method = parts.method === 'taught' ? 'قصّ متعلَّم ✓' : 'تلقائي';
    const type = TYPES[parts.type] ? TYPES[parts.type].ar : 'غير معروف';
    const card = document.createElement('div');
    card.className = 'pending-card';
    card.innerHTML = `
      <img src="/img?max=340&path=${encodeURIComponent(op.source_full)}">
      <div class="body">
        <div class="name">${op.source}</div>
        <div class="meta">${method} • ${type}</div>
        <div class="actions">
          <button class="btn ok">✓ اعتماد</button>
          <button class="btn bad">✗ رفض</button>
        </div>
      </div>`;
    const [okB, badB] = card.querySelectorAll('button');
    okB.onclick = () => review([op.id], []);
    badB.onclick = () => review([], [op.id]);
    list.appendChild(card);
  }
}
async function review(approve, reject) {
  const d = await api('/api/review', { approve, reject });
  status(d);
  toast(reject.length ? `رُفضت ${reject.length} وحُذفت نسخها (${d.removed})` : `اعتُمدت ${approve.length} ✓`);
  loadPending();
}
async function approveAll() {
  const d = await api('/api/pending');
  const ops = d.ops || [];
  if (!ops.length) return toast('لا يوجد معلّق');
  if (!confirm(`اعتماد ${ops.length} عملية معلّقة؟`)) return;
  const r = await api('/api/review', { approve: ops.map(p => p.id), reject: [] });
  status(r); toast(`اعتُمدت ${d.pending.length} عملية ✓`);
  loadPending();
}
async function runWork() {
  toast('جارٍ المعالجة الذاتية…');
  const d = await api('/api/work', {});
  status(d);
  toast(`أُنشئت ${d.created} نسخة (${d.skipped} تخطّت) — راجعها واعتمدها`);
  loadPending();
}
async function undoLast() {
  if (!confirm('التراجع عن آخر دفعة؟ تعود الصور المنقولة لمكانها وتُحذف النسخ المقصوصة')) return;
  const d = await api('/api/undo', {});
  status(d);
  toast(`تم التراجع عن ${d.reverted} عملية`);
  loadPending();
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
