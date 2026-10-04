#!/usr/bin/env python3


import os
import io
import re
import shutil
import sqlite3
import configparser
from pathlib import Path

from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from pypdf import PdfReader, PdfWriter

# --------------------------------------------------------------------------
# Shared configuration (file_toolkit.conf)
# --------------------------------------------------------------------------
CONFIG_FILE_NAME = "file_toolkit.conf"
SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_SETTINGS = {
    "folder": ".",
    "database": "files.db",
    "export_output": "export_output",
    "pdf_dir": "pdf",
    "similar_review": "similar_review",
    "cropped_output": "auto_cropped",
}


# Folder names that are themselves the tool's own structure: month
# (08-2026) and day (01-08-2026) folders.
DATE_DIR_NAME_RE = re.compile(r"\d{2}-\d{4}|\d{2}-\d{2}-\d{4}")


def _climb_out_of_date_folders(folder):
    """If `folder` is itself named like a month or day folder, work from
    its parent instead — otherwise launching the app from inside e.g.
    09-2026 would create a second 09-2026 inside it."""
    climbed = folder
    while DATE_DIR_NAME_RE.fullmatch(climbed.name) and climbed.parent != climbed:
        climbed = climbed.parent
    if climbed != folder:
        print(f"[Config] Working folder moved up to '{climbed}' "
              f"(was launched inside a date folder).")
    return climbed


def load_settings():
    """Read settings from a config file if one exists — the current directory
    first (local override), then next to the script — and resolve every path.
    Relative values are anchored at `folder`; anything missing falls back to
    the defaults above."""
    settings = dict(DEFAULT_SETTINGS)
    config_path = None
    for candidate in (Path.cwd() / CONFIG_FILE_NAME, SCRIPT_DIR / CONFIG_FILE_NAME):
        if candidate.is_file():
            parser = configparser.ConfigParser()
            try:
                parser.read(candidate, encoding="utf-8")
            except configparser.Error as e:
                print(f"[Config] Ignoring malformed config file {candidate}: {e}")
                continue
            if parser.has_section("settings"):
                for key, value in parser.items("settings"):
                    if key in settings:
                        settings[key] = value.strip()
            config_path = candidate
            break

    folder = Path(settings["folder"]).expanduser()
    if not folder.is_absolute():
        folder = Path.cwd() / folder
    folder = _climb_out_of_date_folders(folder)

    def resolve(value):
        path = Path(value).expanduser()
        return path if path.is_absolute() else folder / path

    return {
        "folder": folder,
        "database": str(resolve(settings["database"])),
        "export_output": str(resolve(settings["export_output"])),
        "pdf_dir": str(resolve(settings["pdf_dir"])),
        "similar_review": str(resolve(settings["similar_review"])),
        "cropped_output": str(resolve(settings["cropped_output"])),
        "config_path": config_path,
    }


_SETTINGS = load_settings()
TARGET_DIR = _SETTINGS["folder"]          # directory the toolkit operates on
DB_NAME = _SETTINGS["database"]
EXPORT_OUTPUT_DIR = _SETTINGS["export_output"]
PDF_LINK_DIR = _SETTINGS["pdf_dir"]
SIMILAR_REVIEW_DIR = _SETTINGS["similar_review"]
CROPPED_OUTPUT_DIR = _SETTINGS["cropped_output"]
ACTIVE_CONFIG = _SETTINGS["config_path"]

# Basenames of the toolkit's own output folders, never indexed or sorted
OUTPUT_DIR_NAMES = {
    os.path.basename(os.path.normpath(EXPORT_OUTPUT_DIR)),
    os.path.basename(os.path.normpath(PDF_LINK_DIR)),
    os.path.basename(os.path.normpath(SIMILAR_REVIEW_DIR)),
    os.path.basename(os.path.normpath(CROPPED_OUTPUT_DIR)),
}


# ==========================================================================
# 1) Day folder creation  (01-suffix ... 31-suffix)
# ==========================================================================
def create_day_folders():
    month_input = ask("Enter month (1-12): ")
    year_input = ask("Enter year (e.g. 2026): ")
    if month_input is None or year_input is None:
        print("  [Cancelled]\n")
        return

    try:
        month = int(month_input)
        year = int(year_input)
        if not (1 <= month <= 12):
            raise ValueError("Month must be between 1 and 12")
        if year < 1:
            raise ValueError("Year must be a positive number")
    except ValueError as e:
        print(f"  [Cancelled] Invalid month/year: {e}\n")
        return

    month_str = f"{month:02d}"
    year_str = str(year)

    base_dir = TARGET_DIR
    created = 0
    for i in range(1, 32):
        day = f"{i:02d}"                       # pad with leading zero, 2 digits
        folder_name = f"{day}-{month_str}-{year_str}"
        folder_path = base_dir / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        created += 1
        print(f"  Created: {folder_name}")

    print(f"\nDone! Created/verified {created} folders "
          f"(01-{month_str}-{year_str} .. 31-{month_str}-{year_str}).\n")


# ==========================================================================
# 2) Indexing  (index_files.py)
# ==========================================================================
def setup_database(db_path=DB_NAME):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS local_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            folder_name TEXT NOT NULL,
            full_path TEXT NOT NULL UNIQUE,
            extension TEXT NOT NULL
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_filename ON local_files(filename)')
    conn.commit()
    return conn


def scan_and_index(db_path=DB_NAME, target_dir=TARGET_DIR):
    conn = setup_database(db_path)
    cursor = conn.cursor()

    print("Scanning directories... This might take a moment.")

    file_records = []
    valid_extensions = {'.jpeg', '.jpg', '.png', '.pdf'}

    for root, _, files in os.walk(target_dir):
        folder = os.path.basename(root)

        if (folder.startswith('.') or folder in OUTPUT_DIR_NAMES
                or Path(root) == Path(target_dir)):
            continue

        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                full_path = os.path.abspath(os.path.join(root, file))
                file_records.append((file, folder, full_path, ext))

    cursor.executemany('''
        INSERT OR IGNORE INTO local_files (filename, folder_name, full_path, extension)
        VALUES (?, ?, ?, ?)
    ''', file_records)

    conn.commit()
    print(f"Success! Indexed {cursor.rowcount} new files into '{db_path}'.\n")
    conn.close()


# ==========================================================================
# 3) Export  (export_files.py)
# ==========================================================================
def export_files(db_path=DB_NAME, output_dir=EXPORT_OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT folder_name, filename, full_path
        FROM local_files
        WHERE extension = '.jpeg'
          AND (filename = '0.jpeg'
               OR filename GLOB '[0-9][0-9]-[0-9][0-9]-[0-9][0-9][0-9][0-9].jpeg')
        """
    )
    rows = cursor.fetchall()

    for folder_name, filename, full_path in rows:
        new_filename = f"{folder_name}_0"
        destination = os.path.join(output_dir, new_filename)

        try:
            shutil.copy2(full_path, destination)
            print(f"Copied: {new_filename}")
        except Exception as e:
            print(f"Failed to copy {filename}: {e}")

    conn.close()
    print(f"\nDone! All renamed files are inside '{output_dir}'\n")


# ==========================================================================
# 4) Build PDFs from folders  (FilesToPDF.py)
# ==========================================================================
def natural_sort_key(s):
    """Sorts strings containing numbers in human/natural order."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def build_folder_pdfs(db_path=DB_NAME):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
        SELECT folder_name, filename, full_path
        FROM local_files
        WHERE folder_name NOT LIKE 'export_output'
          AND folder_name NOT LIKE '?'
        ORDER BY folder_name
    """
    cursor.execute(query)
    rows = cursor.fetchall()

    VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
    VALID_PDF_EXTENSION = '.pdf'

    all_folder_names = {folder_name for folder_name, _, _ in rows}
    output_pdf_names = {f"{fn.replace('.', '-')}.pdf".lower() for fn in all_folder_names}

    folders = {}
    for folder_name, filename, full_path in rows:
        _, ext = os.path.splitext(filename.lower())

        is_image = ext in VALID_IMAGE_EXTENSIONS
        is_pdf = ext == VALID_PDF_EXTENSION

        if not (is_image or is_pdf):
            continue

        if is_pdf and filename.lower() in output_pdf_names:
            # Skip PDFs that are themselves outputs of a previous run
            continue

        folders.setdefault(folder_name, []).append((filename, full_path, is_pdf))

    for folder, files in folders.items():
        files.sort(key=lambda x: natural_sort_key(x[0]))

        first_file_path = files[0][1]
        folder_dir = os.path.dirname(first_file_path)

        pdf_filename = f"{folder.replace('.', '-')}.pdf"
        pdf_path = os.path.join(folder_dir, pdf_filename)

        print(f"Creating borderless PDF for folder '{folder}' -> {pdf_path}")

        valid_files = [(fp, is_pdf) for _, fp, is_pdf in files if os.path.exists(fp)]

        if not valid_files:
            print(f"  [Skipped] No printable items found for folder {folder}\n")
            continue

        writer = PdfWriter()
        added_any = False

        for full_path, is_pdf in valid_files:
            if is_pdf:
                try:
                    reader = PdfReader(full_path)
                    for page in reader.pages:
                        writer.add_page(page)
                        added_any = True
                except Exception as pdf_err:
                    print(f"  [Error] Failed reading PDF {os.path.basename(full_path)}: {pdf_err}")
            else:
                try:
                    img = ImageReader(full_path)
                    img_w, img_h = img.getSize()

                    buf = io.BytesIO()
                    c = canvas.Canvas(buf, pagesize=(img_w, img_h))
                    c.drawImage(img, 0, 0, width=img_w, height=img_h)
                    c.showPage()
                    c.save()
                    buf.seek(0)

                    img_page = PdfReader(buf).pages[0]
                    writer.add_page(img_page)
                    added_any = True
                except Exception as img_err:
                    print(f"  [Error] Failed drawing image {os.path.basename(full_path)}: {img_err}")

        if not added_any:
            print(f"  [Skipped] No pages successfully added for folder {folder}\n")
            continue

        try:
            with open(pdf_path, "wb") as f:
                writer.write(f)
            print(f"Successfully saved: {pdf_filename}\n")
        except Exception as e:
            print(f"  [Error] Failed compiling PDF for folder {folder}: {e}")

    cursor.close()
    conn.close()


# ==========================================================================
# 5) Hard-link compiled PDFs  (hardLink.py)
# ==========================================================================
def create_pdf_hardlinks(db_path=DB_NAME, target_link_dir=PDF_LINK_DIR):
    pdf_target_dir = os.path.abspath(target_link_dir)

    if not os.path.exists(pdf_target_dir):
        os.makedirs(pdf_target_dir)
        print(f"Created target directory: {pdf_target_dir}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
        SELECT DISTINCT folder_name, full_path
        FROM local_files
        WHERE folder_name NOT LIKE 'export_output'
          AND folder_name NOT LIKE '?'
    """
    cursor.execute(query)
    rows = cursor.fetchall()

    folder_directories = {}
    for folder_name, full_path in rows:
        if folder_name not in folder_directories:
            folder_directories[folder_name] = os.path.dirname(full_path)

    cursor.close()
    conn.close()

    print(f"\nScanning for compiled PDFs to link into '{target_link_dir}/'...")
    print("-" * 60)

    linked_count = 0
    for folder, folder_dir in folder_directories.items():
        pdf_filename = f"{folder.replace('.', '-')}.pdf"
        source_pdf_path = os.path.join(folder_dir, pdf_filename)
        destination_link_path = os.path.join(pdf_target_dir, pdf_filename)

        if os.path.exists(source_pdf_path):
            if os.path.exists(destination_link_path):
                os.remove(destination_link_path)

            try:
                os.link(source_pdf_path, destination_link_path)
                print(f" Linked: {pdf_filename} -> {target_link_dir}/")
                linked_count += 1
            except Exception as e:
                print(f" [Error] Failed to create link for {pdf_filename}: {e}")
        else:
            print(f" [Missing] Could not find compiled source PDF at: {source_pdf_path}")

    print("-" * 60)
    print(f"Done! Successfully created {linked_count} hard links inside '{pdf_target_dir}'.\n")


# ==========================================================================
# 7) Sort files into month folders (by date in filename)
# ==========================================================================
# Matches a DD-MM-YYYY style date anywhere in a filename. Separators may be
# '-', '.' or '_'; the lookarounds reject matches inside longer digit runs
# (so '2026-08-01' or '01-08-12026' are never misread).
DATE_IN_NAME_RE = re.compile(r"(?<!\d)(\d{2})[-._](\d{2})[-._](\d{4})(?!\d)")


def extract_date_parts(filename):
    """Return (day, month, year) from the first date in `filename`, or None."""
    match = DATE_IN_NAME_RE.search(filename)
    if match is None:
        return None
    day, month, year = (int(part) for part in match.groups())
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return None
    return day, month, year


def unique_destination(directory, filename):
    """Return a non-existing path in `directory` for `filename`, appending
    ' (1)', ' (2)', ... before the extension on collisions."""
    stem, ext = os.path.splitext(filename)
    candidate = directory / filename
    counter = 1
    while candidate.exists():
        candidate = directory / f"{stem} ({counter}){ext}"
        counter += 1
    return candidate


def plan_month_moves(target_dir, recursive, group_by_day=False):
    """Collect (source, destination_folder) pairs for every file whose name
    contains a date. With `group_by_day` the destination is a day subfolder
    named after the file's date inside the month folder, otherwise the month
    folder itself. Files already inside their own destination, hidden files
    and the toolkit's own output folders are left alone."""
    plan = []

    def consider(path):
        if path.name.startswith("."):
            return
        parsed = extract_date_parts(path.name)
        if parsed is None:
            return
        day, month, year = parsed
        folder_name = f"{month:02d}-{year:04d}"
        month_dir = target_dir / folder_name
        if group_by_day:
            dest_dir = month_dir / f"{day:02d}-{month:02d}-{year}"
        else:
            dest_dir = month_dir
        if path.parent == dest_dir:
            return                      # already in its destination folder
        plan.append((path, dest_dir))

    if recursive:
        for root, dirs, files in os.walk(target_dir):
            dirs[:] = [d for d in dirs
                       if not d.startswith(".") and d not in OUTPUT_DIR_NAMES]
            for name in files:
                consider(Path(root) / name)
    else:
        for entry in target_dir.iterdir():
            if entry.is_file():
                consider(entry)
    return plan


def sort_files_into_month_folders(target_dir=TARGET_DIR):
    def display(path):
        try:
            return str(path.relative_to(target_dir))
        except ValueError:
            return str(path)

    print("Moves every file whose name contains a DD-MM-YYYY date into a")
    print("month folder named MM-YYYY (e.g. 01-08-2026.jpg -> 08-2026/).")
    answer = ask("Scan subfolders too? (Y/n): ")
    if answer is None:
        print("  [Cancelled]\n")
        return
    recursive = answer.strip().lower() not in ("n", "no")
    answer = ask(
        "Put each file inside a same-named day folder within the month\n"
        "folder, e.g. 08-2026/01-08-2026/01-08-2026.jpg? (y/N): "
    )
    if answer is None:
        print("  [Cancelled]\n")
        return
    group_by_day = answer.strip().lower() in ("y", "yes")

    plan = plan_month_moves(target_dir, recursive, group_by_day)
    if not plan:
        print("No files with a DD-MM-YYYY date in their name were found.\n")
        return

    plan.sort(key=lambda item: str(item[0]))
    print(f"\nFound {len(plan)} file(s) to move:")
    for source, dest_dir in plan:
        print(f"  {display(source)}  ->  {display(dest_dir / source.name)}")

    confirm = ask(f"\nProceed with moving {len(plan)} file(s)? (Y/n): ")
    if confirm is None or confirm.strip().lower() in ("n", "no"):
        print("  [Cancelled] Nothing was moved.\n")
        return

    moved = 0
    for source, dest_dir in plan:
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            destination = unique_destination(dest_dir, source.name)
            shutil.move(str(source), str(destination))
            moved += 1
            print(f"  Moved: {display(source)} -> {display(destination)}")
        except Exception as e:
            print(f"  [Error] Failed to move {display(source)}: {e}")

    print(f"\nDone! Moved {moved} of {len(plan)} file(s) into month folders.\n")


# ==========================================================================
# 8) One-click full pipeline: Index -> Build PDFs -> Hard link
# ==========================================================================
def run_full_pipeline():
    print("\n=== Running full pipeline: Index -> Build PDFs -> Hard Link ===\n")
    scan_and_index()
    build_folder_pdfs()
    create_pdf_hardlinks()
    print("=== Full pipeline complete! ===\n")


# ==========================================================================
# Menu
# ==========================================================================
def ask(prompt):
    """input() that returns None on EOF/interrupt instead of crashing
    (so piped runs and Ctrl-D end cleanly)."""
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        print()
        return None


MENU = """
==================== File Toolkit ====================
 1) Create day folders (01-MM-YYYY .. 31-MM-YYYY)
 2) Index files into database
 3) Export files (0.jpeg and DD-MM-YYYY.jpeg)
 4) Build PDFs from indexed folders
 5) Hard-link compiled PDFs into ./pdf
 6) One click: Index -> Build PDFs -> Hard Link
 7) Sort files into month folders (by date in filename)
 8) Group similar images (perceptual hash, offline)
 9) Auto-crop document photos into ./auto_cropped
10) Crop feedback: teach the cropper what worked
11) Show learned settings / reset learning
12) Undo last image move/crop batch
13) Image teaching GUI (opens in your browser)
14) Quick Renamer (web GUI: rename + metadata)
 0) Exit
========================================================
"""


def _image_tools():
    """Import the image module lazily so a missing optional package never
    breaks the rest of the menu."""
    try:
        import image_tools
    except ImportError as e:
        print(f"  [Error] Image features need extra packages: {e}")
        print("  Install them with: uv sync\n")
        return None
    return image_tools


def main():
    config_note = str(ACTIVE_CONFIG) if ACTIVE_CONFIG else "defaults (no config file found)"
    print(f"Config: {config_note}")
    print(f"Working folder: {TARGET_DIR}\n")
    while True:
        print(MENU)
        choice = ask("Select an option: ")
        if choice is None:
            print("Goodbye!")
            break
        choice = choice.strip()

        if choice == "1":
            create_day_folders()
        elif choice == "2":
            scan_and_index()
        elif choice == "3":
            export_files()
        elif choice == "4":
            build_folder_pdfs()
        elif choice == "5":
            create_pdf_hardlinks()
        elif choice == "6":
            run_full_pipeline()
        elif choice == "7":
            sort_files_into_month_folders()
        elif choice == "8":
            tools = _image_tools()
            if tools:
                tools.group_similar_images()
        elif choice == "9":
            tools = _image_tools()
            if tools:
                tools.auto_crop_images()
        elif choice == "10":
            tools = _image_tools()
            if tools:
                tools.crop_feedback()
        elif choice == "11":
            tools = _image_tools()
            if tools:
                tools.learning_status()
        elif choice == "12":
            tools = _image_tools()
            if tools:
                tools.undo_last_operation()
        elif choice == "13":
            if _image_tools():
                try:
                    import image_gui
                except ImportError as e:
                    print(f"  [Error] The GUI needs tkinter (python3-tk): {e}\n")
                    continue
                image_gui.run()
        elif choice == "14":
            try:
                import renamer_web
            except ImportError as e:
                print(f"  [Error] The renamer needs its packages: {e}\n")
                continue
            renamer_web.run()
        elif choice == "0":
            print("Goodbye!")
            break
        else:
            print("Invalid choice, please try again.\n")


if __name__ == "__main__":
    main()
