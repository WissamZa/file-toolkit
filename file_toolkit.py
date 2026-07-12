#!/usr/bin/env python3


import os
import io
import re
import shutil
import sqlite3
from pathlib import Path

from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from pypdf import PdfReader, PdfWriter

# --------------------------------------------------------------------------
# Shared configuration
# --------------------------------------------------------------------------
DB_NAME = "files.db"
TARGET_DIR = Path(".")            # directory scanned by the indexer
EXPORT_OUTPUT_DIR = "./export_output"
PDF_LINK_DIR = "pdf"


# ==========================================================================
# 1) Day folder creation  (01-suffix ... 31-suffix)
# ==========================================================================
def create_day_folders():
    month_input = input("Enter month (1-12): ").strip()
    year_input = input("Enter year (e.g. 2026): ").strip()

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

        if folder.startswith('.') or root == '.':
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
        "SELECT folder_name, filename, full_path FROM local_files WHERE filename='0.jpeg'"
    )
    rows = cursor.fetchall()

    for folder_name, filename, full_path in rows:
        new_filename = f"{folder_name}_{filename}"
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
    root_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_target_dir = os.path.join(root_dir, target_link_dir)

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
# 6) One-click full pipeline: Index -> Build PDFs -> Hard link
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
MENU = """
==================== File Toolkit ====================
 1) Create day folders (01-MM-YYYY .. 31-MM-YYYY)
 2) Index files into database
 3) Export files (copies every 0.jpeg)
 4) Build PDFs from indexed folders
 5) Hard-link compiled PDFs into ./pdf
 6) One click: Index -> Build PDFs -> Hard Link
 0) Exit
========================================================
"""


def main():
    while True:
        print(MENU)
        choice = input("Select an option: ").strip()

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
        elif choice == "0":
            print("Goodbye!")
            break
        else:
            print("Invalid choice, please try again.\n")


if __name__ == "__main__":
    main()
