#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "pillow",
#     "pymupdf",
#     "piexif",
# ]
# ///
"""
Quick Renamer - rename images & PDFs one by one with a big preview,
edit their metadata, use name templates, and jump to the next file automatically.

Run (uv installs the dependencies automatically):
    uv run quick_renamer.py
    uv run quick_renamer.py "C:/my/folder"

Click the (i) button in the app for the template / regex tutorial.

Keys
  Enter            Save metadata (if changed) + rename + go to next file
  Right / Left     Next / previous file (when the name box is empty or not focused)
  Alt+Right/Left   Next / previous file, even while typing a name
  Down / Up        Next / previous page of a PDF  (PageDown/PageUp work too)
  Ctrl+U           Undo last rename
"""
import json
import os
import re
import sys
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk, filedialog, messagebox

import pymupdf as fitz  # PyMuPDF
import piexif
from PIL import Image, ImageTk, ImageOps
from PIL.PngImagePlugin import PngInfo

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif"}
JPG_EXT = {".jpg", ".jpeg"}
SUPPORTED = IMG_EXT | {".pdf"}

FIELDS = [
    ("title", "Title"),
    ("description", "Description / Subject"),
    ("author", "Author"),
    ("keywords", "Keywords"),
    ("copyright", "Copyright"),
    ("comment", "Comment"),
]
PDF_FIELDS = {"title", "description", "author", "keywords"}
PNG_KEYS = {"title": "Title", "description": "Description", "author": "Author",
            "keywords": "Keywords", "copyright": "Copyright", "comment": "Comment"}

BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

KEEP = "(keep current)"
EXT_CHOICES = [".jpg", ".jpeg", ".png", ".webp", ".tif", ".bmp", ".gif", ".pdf"]
DEFAULT_TEMPLATES = ["xx", "xx-09-2026", "xx-{date:%m-%Y}", "{n:03}-xx", "{date}_xx"]
CONFIG_PATH = Path.home() / ".quick_renamer.json"

PALETTES = {
    False: dict(bg="#f0f0f0", fg="#1b1b1b", field="#ffffff", canvas="#d5d8dd", muted="#666666",
                accent="#1a4fa0", select="#bcd4f6", btn="#e3e3e3", btn_hover="#d3d3d3", border="#b5b5b5",
                ok="#1e8449", err="#c0392b", warn="#a06000", code_bg="#eef1f6", code_fg="#a02060"),
    True: dict(bg="#202226", fg="#e6e6e6", field="#2c2f34", canvas="#141517", muted="#9aa0a6",
               accent="#6ea8fe", select="#3b5a8c", btn="#34373d", btn_hover="#42464d", border="#4a4d54",
               ok="#52c07c", err="#ff7b7b", warn="#e3a53a", code_bg="#2f3238", code_fg="#f28fb5"),
}

# One pass over the template: {orig} {n} {date} {mtime} {xx} and a bare "xx"
TPL_RE = re.compile(
    r"\{(?P<tok>orig|n|date|mtime)(?::(?P<fmt>[^{}]*))?\}"
    r"|(?P<xx>\{xx\}|(?<![A-Za-z0-9])xx(?![A-Za-z0-9]))",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------------
# Naming logic (pure function, easy to test)
# ----------------------------------------------------------------------------
def make_plan(path, typed, tpl, ext_raw=KEEP, find="", repl="", icase=False, counter=1):
    """Return (new_path or None, error or None). None/None means 'nothing to rename'."""
    folder = os.path.dirname(path)
    base, cur_ext = os.path.splitext(os.path.basename(path))

    # --- target extension
    ext_raw = (ext_raw or "").strip()
    if not ext_raw or ext_raw == KEEP:
        ext = cur_ext
    else:
        ext = ext_raw if ext_raw.startswith(".") else "." + ext_raw
        if not re.fullmatch(r"\.[A-Za-z0-9]{1,8}", ext):
            return None, f"Invalid extension “{ext_raw}”"

    typed = (typed or "").strip()
    tpl = (tpl or "").strip() or "xx"
    uses_xx = any(m.group("xx") for m in TPL_RE.finditer(tpl))

    if uses_xx and not typed:
        # nothing typed: only an extension change is possible
        if ext == cur_ext:
            return None, None
        new_base = base
    else:
        # {orig} = current name, optionally transformed by the regex
        orig = base
        if find:
            try:
                orig = re.sub(find, repl, base, flags=re.IGNORECASE if icase else 0)
            except (re.error, IndexError) as e:
                return None, f"Regex error: {e}"

        def sub(m):
            if m.group("xx"):
                return typed
            tok, fmt = m.group("tok").lower(), m.group("fmt")
            if tok == "orig":
                return orig
            if tok == "n":
                return format(counter, fmt) if fmt else str(counter)
            if tok == "date":
                return datetime.now().strftime(fmt or "%Y-%m-%d")
            ts = datetime.fromtimestamp(os.path.getmtime(path))
            return ts.strftime(fmt or "%Y-%m-%d")

        try:
            new_base = TPL_RE.sub(sub, tpl)
        except (ValueError, OSError) as e:
            return None, f"Template error: {e}"

    new_base = BAD_CHARS.sub("_", new_base).strip(" .")
    for e in {cur_ext.lower(), ext.lower()}:       # user typed "name.jpg" -> don't double it
        if e and new_base.lower().endswith(e):
            new_base = new_base[: -len(e)].rstrip(" .")
            break
    if not new_base:
        return None, "The new name is empty"

    new_path = os.path.join(folder, new_base + ext)
    if new_path == path:
        return None, None
    return new_path, None


def conflicts(new_path, path):
    """True if new_path is a *different* existing file (case-only renames are not conflicts)."""
    try:
        return os.path.exists(new_path) and not os.path.samefile(new_path, path)
    except OSError:
        return False


def unique_path(p):
    """name.ext -> 'name (1).ext', 'name (2).ext' ... first one that is free."""
    stem, ext = os.path.splitext(p)
    n = 1
    while os.path.exists(f"{stem} ({n}){ext}"):
        n += 1
    return f"{stem} ({n}){ext}"


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
def load_config():
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ----------------------------------------------------------------------------
# Metadata helpers
# ----------------------------------------------------------------------------
def _dec_xp(val):
    if not val:
        return ""
    try:
        return bytes(val).decode("utf-16le").rstrip("\x00")
    except Exception:
        return ""


def _enc_xp(text):
    return tuple((text + "\x00").encode("utf-16le"))


def _dec_ascii(val):
    if not val:
        return ""
    if isinstance(val, bytes):
        return val.decode("utf-8", "replace").rstrip("\x00")
    return str(val)


def read_meta(path):
    """Return (values dict, set of editable keys)."""
    ext = os.path.splitext(path)[1].lower()
    vals = {k: "" for k, _ in FIELDS}
    supported = set()

    if ext == ".pdf":
        doc = fitz.open(path)
        try:
            m = doc.metadata or {}
        finally:
            doc.close()
        vals.update(title=m.get("title") or "", description=m.get("subject") or "",
                    author=m.get("author") or "", keywords=m.get("keywords") or "")
        supported = set(PDF_FIELDS)

    elif ext in JPG_EXT:
        supported = {k for k, _ in FIELDS}
        try:
            z = piexif.load(path).get("0th", {})
        except Exception:
            z = {}
        I = piexif.ImageIFD
        vals["title"] = _dec_xp(z.get(I.XPTitle))
        vals["description"] = _dec_ascii(z.get(I.ImageDescription))
        vals["author"] = _dec_ascii(z.get(I.Artist)) or _dec_xp(z.get(I.XPAuthor))
        vals["keywords"] = _dec_xp(z.get(I.XPKeywords))
        vals["copyright"] = _dec_ascii(z.get(I.Copyright))
        vals["comment"] = _dec_xp(z.get(I.XPComment))

    elif ext == ".png":
        supported = {k for k, _ in FIELDS}
        with Image.open(path) as im:
            text = dict(getattr(im, "text", {}) or {})
        for k, name in PNG_KEYS.items():
            vals[k] = str(text.get(name, ""))

    return vals, supported


def write_meta(path, vals):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        doc = fitz.open(path)
        try:
            meta = dict(doc.metadata or {})
            meta.update(title=vals["title"], subject=vals["description"],
                        author=vals["author"], keywords=vals["keywords"])
            doc.set_metadata(meta)
            try:
                doc.saveIncr()
            except Exception:
                tmp = path + ".tmp"
                doc.save(tmp, garbage=0, deflate=True)
                doc.close()
                os.replace(tmp, path)
        finally:
            if not doc.is_closed:
                doc.close()

    elif ext in JPG_EXT:
        try:
            exif = piexif.load(path)
        except Exception:
            exif = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        z = exif.setdefault("0th", {})
        I = piexif.ImageIFD

        def put(tag, value, xp=False):
            if value:
                z[tag] = _enc_xp(value) if xp else value.encode("utf-8")
            else:
                z.pop(tag, None)

        put(I.XPTitle, vals["title"], xp=True)
        put(I.ImageDescription, vals["description"])
        put(I.Artist, vals["author"])
        put(I.XPAuthor, vals["author"], xp=True)
        put(I.XPKeywords, vals["keywords"], xp=True)
        put(I.Copyright, vals["copyright"])
        put(I.XPComment, vals["comment"], xp=True)
        piexif.insert(piexif.dump(exif), path)

    elif ext == ".png":
        with Image.open(path) as im:
            im.load()
            existing = dict(getattr(im, "text", {}) or {})
            icc = im.info.get("icc_profile")
            info = PngInfo()
            for k, v in existing.items():
                if k not in PNG_KEYS.values():
                    info.add_text(k, str(v))
            for key, name in PNG_KEYS.items():
                if vals[key]:
                    info.add_text(name, vals[key])
            tmp = path + ".tmp"
            kw = {"icc_profile": icc} if icc else {}
            im.save(tmp, "PNG", pnginfo=info, **kw)
        os.replace(tmp, path)


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def human_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


# ----------------------------------------------------------------------------
# Help text (shown by the (i) button)
# ----------------------------------------------------------------------------
HELP = """\
## Templates in 20 seconds
Many files have the same name except one word? Put the shared part in the **Template** and write `xx` where the changing word goes. Then you only type that word in the *New name* box and press Enter.

Example:  Template `xx-09-2026`  +  you type `invoice-ali`  →  `invoice-ali-09-2026.pdf`

## Placeholders you can use in a template
`xx` or `{xx}`  –  what you type in the New name box
`{orig}`  –  the current file name (without extension)
`{n}`  –  a counter: 1, 2, 3 …   `{n:03}` gives 001, 002, 003
`{date}`  –  today's date, e.g. 2026-10-02    `{date:%m-%Y}` gives 10-2026
`{mtime}`  –  the file's last-modified date    `{mtime:%Y-%m}` also works

Date codes: `%Y` year · `%m` month · `%d` day · `%H` hour · `%M` minute
The counter starts at the number in the "Counter {n} next" box and goes up by 1 after each rename.

## Saving templates
Type a template, click **Save**, and it appears in the dropdown next time. **Del** removes the one currently shown. They are stored in `~/.quick_renamer.json`.

## Examples
`xx-09-2026`  →  `report-09-2026.pdf`
`{n:03}-xx`  →  `001-sunset.jpg`, `002-beach.jpg` …
`xx_{date:%Y%m%d}`  →  `receipt_20261002.pdf`
`{orig}-checked`  →  `IMG_0042-checked.jpg`   (no typing needed)

## Regex (advanced, optional)
Click **▸ Regex on current name** to open the Find / Replace boxes. They are applied to the current name first, and the result is what `{orig}` means in your template.

Find `IMG_(\\d+)`  Replace `photo-\\1`  Template `{orig}`   →  IMG_0042 becomes photo-0042
Find `\\s+`  Replace `_`  Template `{orig}`   →  spaces become underscores
Find `^(\\d{4})(\\d{2})(\\d{2})`  Replace `\\1-\\2-\\3`  Template `{orig}`   →  20261002_scan becomes 2026-10-02_scan
Find `[()]`  Replace (empty)   →  removes brackets

Use `\\1`, `\\2` for groups (or `\\g<name>`). Tick **Aa** to ignore upper/lower case. If the regex is wrong, the preview turns red and nothing is renamed.

## Changing the extension
By default the extension is **(keep current)**. Pick one from the list or type your own (e.g. `.jpeg`). If you leave the New name box empty and only change the extension, the file keeps its name and gets the new extension.
⚠ This only renames the file, it does NOT convert the image or PDF. After each file the box goes back to “keep current” unless you tick **remember**.

## Keyboard
`Enter`  –  save metadata, rename, go to the next file
`→` / `←`  –  next / previous file (works when the New name box is empty, or when you click the preview)
`Alt+→` / `Alt+←`  –  next / previous file even while you are typing a name
`↓` / `↑`  –  next / previous page of a PDF (`PageDown` / `PageUp` too)
`Ctrl+U`  –  undo the last rename

## Good to know
• If the template contains `xx` and the New name box is empty, nothing is renamed (Enter just moves on).
• A template without `xx` (like `{orig}-checked`) renames when you press Enter, even with an empty box. Use Skip or Alt+→ to move on without renaming.
• The green line under the box always shows the exact new name before you press Enter.
• If a file with the new name already exists, you are asked: **Keep both** (adds “ (1)”, “ (2)” …), **Overwrite** (replaces the old file, cannot be undone) or **Cancel**. The preview line turns orange as a warning before you even press Enter.
• Characters like / : ? * are replaced with _.
• The 🌙 / ☀ button (top right) switches between dark and light mode; your choice is remembered.
• Ctrl+U or **Undo rename** puts the last name back (metadata changes are not undone).
"""


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
class RenamerApp:
    def __init__(self, root):
        self.root = root
        root.title("Quick Renamer")
        h = min(860, max(600, root.winfo_screenheight() - 90))
        root.geometry(f"1300x{h}")
        root.minsize(980, 560)

        self.cfg = load_config()
        self.templates = list(self.cfg.get("templates") or DEFAULT_TEMPLATES)

        self.files = []
        self.idx = 0
        self.base_img = None
        self.tk_img = None
        self.page = 0
        self.page_count = 1
        self.pdf_page_size = None
        self.orig_meta = {}
        self.supported_meta = set()
        self.undo_stack = []
        self.resize_job = None
        self.field_vars = {}
        self.field_widgets = {}
        self.help_win = None
        self.help_txt = None
        self._last_msg = ""
        self.dark = bool(self.cfg.get("dark", False))
        self.c = PALETTES[self.dark]

        self._build_ui()
        self._bind_keys()
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        self.style = ttk.Style()
        if "clam" in self.style.theme_names():
            self.style.theme_use("clam")

        top = ttk.Frame(self.root, padding=(8, 6))
        top.pack(fill="x")
        ttk.Button(top, text="📂 Open folder…", command=self.choose_folder).pack(side="left")
        self.sub_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Include subfolders", variable=self.sub_var).pack(side="left", padx=10)
        self.theme_btn = ttk.Button(top, text="🌙 Dark", width=9, command=self.toggle_theme)
        self.theme_btn.pack(side="right", padx=(8, 0))
        self.counter = ttk.Label(top, text="No folder loaded")
        self.counter.pack(side="right")

        self.progress = ttk.Progressbar(self.root, mode="determinate")
        self.progress.pack(fill="x", padx=8)

        self.status = ttk.Label(self.root, text="Open a folder to begin.", anchor="w", relief="sunken", padding=(6, 2))
        self.status.pack(fill="x", side="bottom")

        paned = ttk.PanedWindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=8, pady=8)

        # ---- left: preview + info
        left = ttk.Frame(paned)
        self.canvas = tk.Canvas(left, bg=self.c["canvas"], highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_resize)

        pager = ttk.Frame(left)
        pager.pack(fill="x", pady=(4, 0))
        self.prev_page_btn = ttk.Button(pager, text="◀ Page", width=9, command=lambda: self.change_page(-1))
        self.prev_page_btn.pack(side="left")
        self.page_label = ttk.Label(pager, text="", anchor="center")
        self.page_label.pack(side="left", expand=True)
        self.next_page_btn = ttk.Button(pager, text="Page ▶", width=9, command=lambda: self.change_page(1))
        self.next_page_btn.pack(side="right")

        self.info_label = ttk.Label(left, text="", style="Muted.TLabel", justify="left", wraplength=780)
        self.info_label.pack(anchor="w", fill="x", pady=(4, 0))
        paned.add(left, weight=3)

        # ---- right: rename + template + metadata
        right = ttk.Frame(paned, padding=(10, 0))
        paned.add(right, weight=1)

        ttk.Label(right, text="Current name", style="Muted.TLabel").pack(anchor="w")
        self.cur_name = ttk.Label(right, text="—", style="CurName.TLabel", wraplength=380, justify="left")
        self.cur_name.pack(anchor="w", fill="x", pady=(0, 4))

        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(right, textvariable=self.name_var, font=("Segoe UI", 12))
        self.name_entry.pack(fill="x")
        self.preview_label = ttk.Label(right, text="", wraplength=380, justify="left")
        self.preview_label.pack(anchor="w", fill="x", pady=(3, 6))

        # template box
        tf = ttk.LabelFrame(right, text="Template", padding=6)
        tf.pack(fill="x")
        row = ttk.Frame(tf)
        row.pack(fill="x")
        self.tpl_var = tk.StringVar(value=self.cfg.get("last_template", "xx"))
        self.tpl_combo = ttk.Combobox(row, textvariable=self.tpl_var, values=self.templates)
        self.tpl_combo.pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Save", width=5, command=self.save_template).pack(side="left", padx=(4, 0))
        ttk.Button(row, text="Del", width=4, command=self.delete_template).pack(side="left", padx=(2, 0))
        ttk.Button(row, text="ⓘ", width=3, command=self.show_help).pack(side="left", padx=(4, 0))
        ttk.Label(tf, text="xx = what you type   {orig} {n} {date}   — click ⓘ for help",
                  style="Muted.TLabel").pack(anchor="w", pady=(3, 0))

        crow = ttk.Frame(tf)
        crow.pack(fill="x", pady=(4, 0))
        ttk.Label(crow, text="Counter {n} next:").pack(side="left")
        self.counter_var = tk.IntVar(value=1)
        ttk.Spinbox(crow, from_=0, to=999999, textvariable=self.counter_var, width=7).pack(side="left", padx=6)

        self.re_toggle = ttk.Button(tf, text="▸ Regex on current name", command=self.toggle_regex)
        self.re_toggle.pack(anchor="w", pady=(6, 0))
        self.re_box = ttk.Frame(tf)
        self.re_box.columnconfigure(1, weight=1)
        self.re_find = tk.StringVar(value=self.cfg.get("re_find", ""))
        self.re_repl = tk.StringVar(value=self.cfg.get("re_repl", ""))
        self.re_icase = tk.BooleanVar(value=self.cfg.get("re_icase", False))
        ttk.Label(self.re_box, text="Find").grid(row=0, column=0, sticky="w", pady=2)
        self.re_find_entry = ttk.Entry(self.re_box, textvariable=self.re_find)
        self.re_find_entry.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        ttk.Label(self.re_box, text="Replace").grid(row=1, column=0, sticky="w", pady=2)
        self.re_repl_entry = ttk.Entry(self.re_box, textvariable=self.re_repl)
        self.re_repl_entry.grid(row=1, column=1, sticky="ew", padx=(6, 0))
        ttk.Checkbutton(self.re_box, text="Aa  ignore case", variable=self.re_icase).grid(
            row=2, column=1, sticky="w", padx=(6, 0))
        if self.re_find.get():
            self.toggle_regex()

        # extension
        ef = ttk.Frame(right)
        ef.pack(fill="x", pady=(8, 0))
        ttk.Label(ef, text="Extension").pack(side="left")
        self.ext_var = tk.StringVar(value=KEEP)
        self.ext_combo = ttk.Combobox(ef, textvariable=self.ext_var, values=[KEEP] + EXT_CHOICES, width=16)
        self.ext_combo.pack(side="left", padx=6)
        self.ext_sticky = tk.BooleanVar(value=False)
        ttk.Checkbutton(ef, text="remember", variable=self.ext_sticky).pack(side="left")
        self.ext_warn = ttk.Label(right, text="", style="Warn.TLabel", wraplength=380, justify="left")
        self.ext_warn.pack(anchor="w", fill="x")

        # buttons
        ttk.Button(right, text="Save & Next  ⏎", style="Big.TButton", command=self.submit).pack(fill="x", pady=(6, 0))
        nav = ttk.Frame(right)
        nav.pack(fill="x", pady=(6, 0))
        ttk.Button(nav, text="◀ Prev (←)", command=lambda: self.go(-1)).pack(side="left", expand=True, fill="x")
        ttk.Button(nav, text="Skip (→) ▶", command=lambda: self.go(1)).pack(side="left", expand=True, fill="x", padx=4)
        ttk.Button(nav, text="Undo rename", command=self.undo).pack(side="left", expand=True, fill="x")
        ttk.Label(right, text="←  → files   ·   ↑  ↓ PDF pages   ·   Enter save & next\n"
                              "(← → work when the name box is empty; Alt+← → always)",
                  style="Muted.TLabel", justify="left").pack(anchor="w", pady=(4, 0))

        # metadata
        meta = ttk.LabelFrame(right, text="Metadata", padding=8)
        meta.pack(fill="both", expand=True, pady=(10, 0))
        meta.columnconfigure(1, weight=1)
        for r, (key, label) in enumerate(FIELDS):
            ttk.Label(meta, text=label).grid(row=r, column=0, sticky="w", pady=2, padx=(0, 8))
            var = tk.StringVar()
            ent = ttk.Entry(meta, textvariable=var)
            ent.grid(row=r, column=1, sticky="ew", pady=2)
            self.field_vars[key] = var
            self.field_widgets[key] = ent
        self.meta_note = ttk.Label(meta, text="", style="Warn.TLabel", wraplength=340, justify="left")
        self.meta_note.grid(row=len(FIELDS), column=0, columnspan=2, sticky="w", pady=(6, 0))

        # live preview whenever anything that affects the name changes
        for var in (self.name_var, self.tpl_var, self.re_find, self.re_repl,
                    self.re_icase, self.ext_var, self.counter_var):
            var.trace_add("write", lambda *a: self.refresh_preview())

        self.apply_theme()
        self._draw_message("Open a folder to begin")

    def _bind_keys(self):
        r = self.root
        self.name_entry.bind("<Return>", lambda e: self.submit())
        for w in self.field_widgets.values():
            w.bind("<Return>", lambda e: self.submit())
        for w in (self.tpl_combo, self.ext_combo, self.re_find_entry, self.re_repl_entry):
            w.bind("<Return>", lambda e: self.name_entry.focus_set())
        self.tpl_combo.bind("<<ComboboxSelected>>", lambda e: self.name_entry.focus_set())
        self.ext_combo.bind("<<ComboboxSelected>>", lambda e: self.name_entry.focus_set())
        # files: plain arrows (smart: they only steal the key when it can't move a text cursor)
        r.bind("<Right>", lambda e: self._arrow_file(1))
        r.bind("<Left>", lambda e: self._arrow_file(-1))
        r.bind("<Alt-Right>", lambda e: self.go(1))     # always works, even while typing
        r.bind("<Alt-Left>", lambda e: self.go(-1))
        # PDF pages
        r.bind("<Down>", lambda e: self._arrow_page(1))
        r.bind("<Up>", lambda e: self._arrow_page(-1))
        r.bind("<Next>", lambda e: self.change_page(1))    # PageDown
        r.bind("<Prior>", lambda e: self.change_page(-1))  # PageUp
        r.bind("<Control-u>", lambda e: self.undo())

    def _focused(self):
        try:
            return self.root.focus_get()
        except KeyError:        # combobox popdown
            return None

    def _arrow_file(self, d):
        """Left/Right = previous/next file, unless a text field needs the key for its cursor."""
        w = self._focused()
        if isinstance(w, ttk.Entry) and not (w is self.name_entry and not self.name_var.get()):
            return None         # let the entry move its text cursor
        return self.go(d)

    def _arrow_page(self, d):
        """Up/Down = previous/next PDF page (not while a dropdown/spinbox uses the key)."""
        if isinstance(self._focused(), (ttk.Combobox, ttk.Spinbox)):
            return None
        return self.change_page(d)

    # ----------------------------------------------------- templates / help
    def toggle_regex(self):
        if self.re_box.winfo_manager():
            self.re_box.pack_forget()
            self.re_toggle.config(text="▸ Regex on current name")
        else:
            self.re_box.pack(fill="x", pady=(4, 0))
            self.re_toggle.config(text="▾ Regex on current name")

    def save_template(self):
        t = self.tpl_var.get().strip()
        if t and t not in self.templates:
            self.templates.append(t)
            self.tpl_combo.config(values=self.templates)
            self.save_config()
            self.set_status(f"Template saved: {t}")

    def delete_template(self):
        t = self.tpl_var.get().strip()
        if t in self.templates:
            self.templates.remove(t)
            self.tpl_combo.config(values=self.templates)
            self.tpl_var.set("xx")
            self.save_config()
            self.set_status(f"Template deleted: {t}")

    def save_config(self):
        data = {
            "templates": self.templates,
            "last_template": self.tpl_var.get(),
            "re_find": self.re_find.get(),
            "re_repl": self.re_repl.get(),
            "re_icase": self.re_icase.get(),
            "dark": self.dark,
        }
        try:
            CONFIG_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def _on_close(self):
        self.save_config()
        self.root.destroy()

    # ---------------------------------------------------------------- theme
    def toggle_theme(self):
        self.dark = not self.dark
        self.apply_theme()
        self.save_config()

    def apply_theme(self):
        c = self.c = PALETTES[self.dark]
        s, r = self.style, self.root
        r.configure(bg=c["bg"])
        s.configure(".", background=c["bg"], foreground=c["fg"], bordercolor=c["border"],
                    darkcolor=c["bg"], lightcolor=c["bg"], troughcolor=c["field"],
                    focuscolor=c["accent"], selectbackground=c["select"], selectforeground=c["fg"],
                    insertcolor=c["fg"])
        s.configure("TLabelframe", background=c["bg"], bordercolor=c["border"])
        s.configure("TLabelframe.Label", background=c["bg"], foreground=c["fg"])
        s.configure("TButton", background=c["btn"], foreground=c["fg"], bordercolor=c["border"],
                    lightcolor=c["btn"], darkcolor=c["btn"])
        s.map("TButton",
              background=[("disabled", c["bg"]), ("pressed", c["select"]), ("active", c["btn_hover"])],
              foreground=[("disabled", c["muted"])])
        s.configure("Big.TButton", font=("Segoe UI", 11, "bold"), padding=8)
        s.configure("Danger.TButton", foreground=c["err"])
        s.configure("CurName.TLabel", font=("Segoe UI", 12, "bold"), foreground=c["accent"])
        s.configure("Muted.TLabel", foreground=c["muted"])
        s.configure("Warn.TLabel", foreground=c["warn"])
        for name in ("TEntry", "TCombobox", "TSpinbox"):
            s.configure(name, fieldbackground=c["field"], foreground=c["fg"], insertcolor=c["fg"],
                        bordercolor=c["border"], lightcolor=c["field"], darkcolor=c["field"])
        for name in ("TCombobox", "TSpinbox"):
            s.configure(name, background=c["btn"], arrowcolor=c["fg"])
        s.map("TEntry", fieldbackground=[("disabled", c["bg"])], foreground=[("disabled", c["muted"])])
        s.map("TCombobox", fieldbackground=[("disabled", c["bg"]), ("readonly", c["field"])],
              foreground=[("disabled", c["muted"])], background=[("active", c["btn_hover"])])
        s.map("TSpinbox", fieldbackground=[("disabled", c["bg"])], foreground=[("disabled", c["muted"])])
        s.configure("TCheckbutton", background=c["bg"], foreground=c["fg"],
                    indicatorbackground=c["field"], indicatorforeground=c["fg"])
        s.map("TCheckbutton", background=[("active", c["bg"])],
              indicatorbackground=[("pressed", c["btn_hover"]), ("disabled", c["bg"]), ("!disabled", c["field"])])
        s.configure("TScrollbar", background=c["btn"], troughcolor=c["field"], bordercolor=c["border"],
                    arrowcolor=c["fg"], lightcolor=c["btn"], darkcolor=c["btn"])
        s.configure("Horizontal.TProgressbar", background=c["accent"], troughcolor=c["field"],
                    bordercolor=c["border"], lightcolor=c["accent"], darkcolor=c["accent"])
        s.configure("TPanedwindow", background=c["bg"])

        # plain tk widgets + dropdown lists
        self.canvas.configure(bg=c["canvas"])
        for opt, val in (("background", c["field"]), ("foreground", c["fg"]),
                         ("selectBackground", c["select"]), ("selectForeground", c["fg"])):
            r.option_add(f"*TCombobox*Listbox.{opt}", val)
        for cb in (self.tpl_combo, self.ext_combo):
            try:
                pop = r.tk.call("ttk::combobox::PopdownWindow", str(cb))
                r.tk.call(f"{pop}.f.l", "configure", "-background", c["field"], "-foreground", c["fg"],
                          "-selectbackground", c["select"], "-selectforeground", c["fg"])
            except tk.TclError:
                pass
        self.theme_btn.config(text="☀ Light" if self.dark else "🌙 Dark")
        self._style_help()
        if self.base_img is None and self._last_msg:
            self._draw_message(self._last_msg)
        self.refresh_preview()

    def _style_help(self):
        if not (self.help_win and self.help_txt and self.help_win.winfo_exists()):
            return
        c = self.c
        self.help_win.configure(bg=c["bg"])
        t = self.help_txt
        t.configure(bg=c["field"], fg=c["fg"], insertbackground=c["fg"],
                    selectbackground=c["select"], selectforeground=c["fg"])
        t.tag_configure("h", foreground=c["accent"])
        t.tag_configure("code", background=c["code_bg"], foreground=c["code_fg"])

    # ------------------------------------------------- name conflict dialog
    def ask_conflict(self, src, dst):
        """Modal dialog. Returns 'keep' (add a number), 'overwrite' or 'cancel'."""
        win = tk.Toplevel(self.root)
        win.title("File already exists")
        win.configure(bg=self.c["bg"])
        win.transient(self.root)
        win.resizable(False, False)
        choice = {"v": "cancel"}

        def pick(v):
            choice["v"] = v
            win.destroy()

        def details(p):
            st = os.stat(p)
            return f"{human_size(st.st_size)}  •  {time.strftime('%Y-%m-%d %H:%M', time.localtime(st.st_mtime))}"

        frm = ttk.Frame(win, padding=16)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text="A file with this name already exists",
                  font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(frm, text=os.path.basename(dst), style="CurName.TLabel",
                  wraplength=460, justify="left").pack(anchor="w", pady=(6, 8))
        ttk.Label(frm, text=f"Existing file:  {details(dst)}\nThis file:           {details(src)}",
                  style="Muted.TLabel", justify="left").pack(anchor="w")
        ttk.Label(frm, text="What do you want to do?").pack(anchor="w", pady=(14, 6))
        b_keep = ttk.Button(frm, text=f"Keep both  →  {os.path.basename(unique_path(dst))}",
                            style="Big.TButton", command=lambda: pick("keep"))
        b_keep.pack(fill="x")
        ttk.Button(frm, text="Overwrite the existing file  (cannot be undone)", style="Danger.TButton",
                   command=lambda: pick("overwrite")).pack(fill="x", pady=6)
        ttk.Button(frm, text="Cancel – let me change the name", command=lambda: pick("cancel")).pack(fill="x")

        def on_return(_e):
            w = win.focus_get()
            if isinstance(w, ttk.Button):
                w.invoke()
            else:
                pick("keep")
            return "break"

        win.bind("<Return>", on_return)
        win.bind("<Escape>", lambda e: pick("cancel"))
        win.protocol("WM_DELETE_WINDOW", lambda: pick("cancel"))
        win.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - win.winfo_width()) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - win.winfo_height()) // 3
        win.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        b_keep.focus_set()              # the safe choice is the default
        try:
            win.wait_visibility()
            win.grab_set()
        except tk.TclError:
            pass
        self.root.wait_window(win)
        return choice["v"]

    def show_help(self):
        if self.help_win and self.help_win.winfo_exists():
            self.help_win.lift()
            self.help_win.focus_set()
            return
        win = tk.Toplevel(self.root)
        self.help_win = win
        win.title("Templates, regex & extension – how to use")
        win.geometry("720x680")
        win.transient(self.root)
        frm = ttk.Frame(win)
        frm.pack(fill="both", expand=True)
        txt = tk.Text(frm, wrap="word", padx=16, pady=12, font=("Segoe UI", 10), relief="flat", spacing3=3,
                      highlightthickness=0)
        self.help_txt = txt
        sb = ttk.Scrollbar(frm, command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        txt.pack(side="left", fill="both", expand=True)
        txt.tag_configure("h", font=("Segoe UI", 13, "bold"), spacing1=12, spacing3=4)
        txt.tag_configure("code", font=("Consolas", 10))
        txt.tag_configure("b", font=("Segoe UI", 10, "bold"))
        txt.tag_configure("i", font=("Segoe UI", 10, "italic"))

        for line in HELP.splitlines():
            if line.startswith("## "):
                txt.insert("end", line[3:] + "\n", "h")
                continue
            # inline markup: `code`  **bold**  *italic*
            for part in re.split(r"(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)", line):
                if part.startswith("`") and part.endswith("`") and len(part) > 1:
                    txt.insert("end", part[1:-1], "code")
                elif part.startswith("**") and part.endswith("**") and len(part) > 3:
                    txt.insert("end", part[2:-2], "b")
                elif part.startswith("*") and part.endswith("*") and len(part) > 2:
                    txt.insert("end", part[1:-1], "i")
                else:
                    txt.insert("end", part)
            txt.insert("end", "\n")
        txt.configure(state="disabled")
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=6)
        self._style_help()

    # -------------------------------------------------------------- folder
    def choose_folder(self, folder=None):
        folder = folder or filedialog.askdirectory(title="Choose a folder with images / PDFs")
        if not folder:
            return
        files = []
        if self.sub_var.get():
            for dp, _, fn in os.walk(folder):
                files += [os.path.join(dp, f) for f in fn]
        else:
            files = [os.path.join(folder, f) for f in os.listdir(folder)]
        files = [f for f in files if os.path.isfile(f) and os.path.splitext(f)[1].lower() in SUPPORTED]
        files.sort(key=lambda p: natural_key(p))
        if not files:
            messagebox.showinfo("Nothing found", "No images or PDFs in that folder.")
            return
        self.files = files
        self.undo_stack.clear()
        self.load(0)

    # ------------------------------------------------------------- loading
    def load(self, i):
        self.idx = max(0, min(i, len(self.files) - 1))
        path = self.files[self.idx]
        self.page = 0
        self.page_count = 1
        self.pdf_page_size = None

        self._load_preview(path)
        try:
            vals, supported = read_meta(path)
            note = ""
        except Exception as e:
            vals, supported = {k: "" for k, _ in FIELDS}, set()
            note = f"Could not read metadata: {e}"
        self.orig_meta = vals
        self.supported_meta = supported
        for key, _ in FIELDS:
            self.field_vars[key].set(vals[key])
            self.field_widgets[key].state(["!disabled"] if key in supported else ["disabled"])
        ext = os.path.splitext(path)[1].lower()
        if note:
            pass
        elif not supported:
            note = f"Metadata editing isn't supported for {ext} files (rename still works)."
        elif ext == ".pdf":
            note = "PDF stores Title, Subject, Author and Keywords."
        self.meta_note.config(text=note)

        self.cur_name.config(text=os.path.basename(path))
        self.name_var.set("")
        if not self.ext_sticky.get():
            self.ext_var.set(KEEP)
        self._update_info(path)
        self.counter.config(text=f"{self.idx + 1} / {len(self.files)}")
        self.progress.config(maximum=len(self.files), value=self.idx + 1)
        self.refresh_preview()
        self.name_entry.focus_set()

    def _load_preview(self, path):
        self.base_img = None
        try:
            if path.lower().endswith(".pdf"):
                self._render_pdf_page(path)
            else:
                with Image.open(path) as im:
                    im = ImageOps.exif_transpose(im)
                    im.thumbnail((2400, 2400))
                    self.base_img = im.convert("RGBA") if im.mode in ("RGBA", "LA", "P") else im.convert("RGB")
        except Exception as e:
            self.base_img = None
            self._draw_message(f"Cannot preview this file\n{e}")
            return
        self._redraw()

    def _render_pdf_page(self, path):
        doc = fitz.open(path)
        try:
            self.page_count = doc.page_count
            self.page = max(0, min(self.page, self.page_count - 1))
            pg = doc.load_page(self.page)
            r = pg.rect
            zoom = min(3.0, 2000 / max(r.width, r.height, 1))
            pix = pg.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            self.base_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            self.pdf_page_size = (r.width, r.height)
        finally:
            doc.close()

    def change_page(self, d):
        if not self.files:
            return "break"
        path = self.files[self.idx]
        if not path.lower().endswith(".pdf"):
            return "break"
        new = self.page + d
        if 0 <= new < self.page_count:
            self.page = new
            try:
                self._render_pdf_page(path)
                self._redraw()
            except Exception as e:
                self._draw_message(f"Cannot render page\n{e}")
        return "break"

    # ------------------------------------------------------------- drawing
    def _on_resize(self, _e):
        if self.resize_job:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(60, self._redraw)

    def _draw_message(self, text):
        self._last_msg = text
        self.canvas.delete("all")
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.canvas.create_text(w // 2, h // 2, text=text, fill=self.c["muted"], font=("Segoe UI", 14),
                                justify="center", width=max(w - 40, 100))

    def _redraw(self):
        self.resize_job = None
        if self.base_img is None:
            if self._last_msg:
                self._draw_message(self._last_msg)   # keep the message centred when resizing
            return
        w, h = max(self.canvas.winfo_width(), 50), max(self.canvas.winfo_height(), 50)
        img = self.base_img
        scale = min(w / img.width, h / img.height)
        size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
        shown = img.resize(size, Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(shown)
        self.canvas.delete("all")
        self.canvas.create_image(w // 2, h // 2, image=self.tk_img)

        is_pdf = self.files and self.files[self.idx].lower().endswith(".pdf")
        if is_pdf:
            self.page_label.config(text=f"Page {self.page + 1} / {self.page_count}")
            self.prev_page_btn.state(["!disabled"] if self.page > 0 else ["disabled"])
            self.next_page_btn.state(["!disabled"] if self.page < self.page_count - 1 else ["disabled"])
        else:
            self.page_label.config(text="")
            self.prev_page_btn.state(["disabled"])
            self.next_page_btn.state(["disabled"])

    def _update_info(self, path):
        st = os.stat(path)
        ext = os.path.splitext(path)[1].lower()
        parts = [ext[1:].upper(), human_size(st.st_size),
                 time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))]
        if ext == ".pdf":
            parts.append(f"{self.page_count} page(s)")
            if self.pdf_page_size:
                w, h = self.pdf_page_size
                parts.append(f"{w / 72 * 25.4:.0f}×{h / 72 * 25.4:.0f} mm")
        else:
            try:
                with Image.open(path) as im:
                    parts.append(f"{im.width}×{im.height}px")
            except Exception:
                pass
        self.info_label.config(text="  •  ".join(parts) + "\n" + os.path.dirname(path))

    # ---------------------------------------------------------- navigation
    def go(self, d):
        if not self.files:
            return "break"
        new = self.idx + d
        if 0 <= new < len(self.files):
            self.load(new)
        else:
            self.set_status("No more files in that direction.")
        return "break"

    # ------------------------------------------------------- rename / save
    def counter_value(self):
        try:
            return int(self.counter_var.get())
        except (tk.TclError, ValueError):
            return 0

    def plan(self, path):
        return make_plan(path, self.name_var.get(), self.tpl_var.get(), self.ext_var.get(),
                         self.re_find.get(), self.re_repl.get(), self.re_icase.get(),
                         self.counter_value())

    def refresh_preview(self):
        if not self.files:
            return
        path = self.files[self.idx]
        new_path, err = self.plan(path)
        c = self.c
        if err:
            self.preview_label.config(text=f"⚠ {err}", foreground=c["err"])
        elif new_path and conflicts(new_path, path):
            self.preview_label.config(
                text=f"→ {os.path.basename(new_path)}\n⚠ A file with this name already exists — "
                     f"you'll be asked what to do", foreground=c["warn"])
        elif new_path:
            self.preview_label.config(text=f"→ {os.path.basename(new_path)}", foreground=c["ok"])
        elif not self.name_var.get().strip():
            self.preview_label.config(text="Type the new name (Enter on empty = skip)", foreground=c["muted"])
        else:
            self.preview_label.config(text="(name unchanged)", foreground=c["muted"])

        cur_ext = os.path.splitext(path)[1]
        raw = self.ext_var.get().strip()
        if raw and raw != KEEP:
            target = raw if raw.startswith(".") else "." + raw
            if target.lower() != cur_ext.lower():
                self.ext_warn.config(text="⚠ Only renames the file, it does NOT convert it.")
                return
        self.ext_warn.config(text="")

    def current_meta(self):
        return {k: self.field_vars[k].get().strip() for k, _ in FIELDS}

    def meta_dirty(self):
        cur = self.current_meta()
        return any(cur[k] != self.orig_meta.get(k, "").strip() for k in self.supported_meta)

    def submit(self):
        if not self.files:
            return
        path = self.files[self.idx]
        msgs = []

        new_path, err = self.plan(path)
        if err:
            messagebox.showwarning("Check the name", err)
            return

        # 0) a different file already has that name? ask first (nothing has been changed yet)
        overwrite = False
        if new_path and conflicts(new_path, path):
            choice = self.ask_conflict(path, new_path)
            if choice == "cancel":
                self.set_status("Cancelled – nothing was changed. Edit the name and try again.")
                self.name_entry.focus_set()
                return
            if choice == "keep":
                new_path = unique_path(new_path)
                msgs.append("kept both")
            else:
                overwrite = True

        # 1) metadata
        if self.meta_dirty():
            try:
                write_meta(path, self.current_meta())
                msgs.append("metadata saved")
            except Exception as e:
                messagebox.showerror("Metadata error", f"Could not save metadata:\n{e}")
                return

        # 2) rename
        if new_path:
            try:
                if overwrite:
                    os.replace(path, new_path)      # replaces the existing file
                else:
                    os.rename(path, new_path)
            except Exception as e:
                messagebox.showerror("Rename failed", str(e))
                return
            if overwrite:
                # the overwritten file may be in our list: drop it
                target = os.path.normcase(os.path.abspath(new_path))
                for j, f in enumerate(self.files):
                    if j != self.idx and os.path.normcase(os.path.abspath(f)) == target:
                        del self.files[j]
                        if j < self.idx:
                            self.idx -= 1
                        break
                msgs.append("existing file overwritten")
            else:
                self.undo_stack.append((new_path, path))   # an overwrite can't be undone
            self.files[self.idx] = new_path
            msgs.append(f"renamed → {os.path.basename(new_path)}")
            if any((m.group("tok") or "").lower() == "n" for m in TPL_RE.finditer(self.tpl_var.get())):
                self.counter_var.set(self.counter_value() + 1)
            self.save_config()

        self.set_status("; ".join(msgs).capitalize() if msgs else "No changes; moved on.")

        # 3) next
        if self.idx + 1 < len(self.files):
            self.load(self.idx + 1)
        else:
            self.load(self.idx)
            messagebox.showinfo("Done", "That was the last file in the list. 🎉")

    def undo(self):
        if not self.undo_stack:
            self.set_status("Nothing to undo.")
            return "break"
        new_path, old_path = self.undo_stack.pop()
        try:
            os.rename(new_path, old_path)
        except Exception as e:
            messagebox.showerror("Undo failed", str(e))
            return "break"
        if new_path in self.files:
            i = self.files.index(new_path)
            self.files[i] = old_path
            self.load(i)
        self.set_status(f"Restored {os.path.basename(old_path)} (metadata changes are not undone).")
        return "break"

    def set_status(self, text):
        self.status.config(text=text)


def main(folder=None):
    """Launch the renamer GUI. `folder` (or argv[1]) is opened on start —
    the file toolkit menu passes its working folder here."""
    root = tk.Tk()
    app = RenamerApp(root)
    folder = folder or (sys.argv[1] if len(sys.argv) > 1 else None)
    if folder and os.path.isdir(folder):
        root.after(100, lambda: app.choose_folder(folder))
    root.mainloop()


if __name__ == "__main__":
    main()
