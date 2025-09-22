import os
import re
import sqlite3
import pdfplumber
from tqdm import tqdm

# === CONFIGURATION ===
STATEMENTS_DIR = "statements"
WORKING_PAPERS_DIR = "working_papers"
DB_PATH = "positions.db"
DEBUG_MODE = True  # 设置为 True 会重建表（清空数据）


# === STEP 1: INITIALIZE DATABASE ===
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if DEBUG_MODE:
        print("DEBUG_MODE is ON: Dropping existing table...")
        cursor.execute("DROP TABLE IF EXISTS npt_positions")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS npt_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            year INTEGER,
            country TEXT,
            doc_type TEXT,
            source_file TEXT,
            page_number INTEGER,
            paragraph_id INTEGER,
            full_text TEXT,
            issues TEXT,
            position_summary TEXT,
            confidence_score REAL,
            sentence_count INTEGER,
            sentences TEXT,
            issue_sentence_map TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_file, page_number, paragraph_id)
        )
    """)
    conn.commit()
    return conn


# === STEP 2: EXTRACT YEAR, COUNTRY, TYPE FROM FILENAME ===
def parse_filename(filename, folder_name):
    match = re.match(r"(\d{4})_(\w+)_", filename)
    if not match:
        return None
    year = int(match.group(1))
    country = match.group(2)
    doc_type = "statement" if folder_name == "statements" else "working_paper"
    return year, country, doc_type


# === STEP 3: SMART PARAGRAPH SPLITTER ===
def split_paragraphs(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    paragraphs = []
    current_para = []

    for i, line in enumerate(lines):
        if i == 0:
            current_para.append(line)
            continue

        prev_line = lines[i - 1]

        # 启发式判断新段落
        is_new_para = (
            prev_line.endswith(('.', '。', '!', '?')) and line[0].isupper()
        )
        is_numbered = re.match(r"^(\d+[\.\)]|[-•—])\s*", line)

        if is_new_para or is_numbered:
            if current_para:
                paragraphs.append(' '.join(current_para))
                current_para = [line]
            else:
                current_para.append(line)
        else:
            current_para.append(line)

    if current_para:
        paragraphs.append(' '.join(current_para))

    return paragraphs


# === STEP 4: PROCESS SINGLE PDF ===
def process_pdf(filepath, folder_name, conn):
    filename = os.path.basename(filepath)
    parsed = parse_filename(filename, folder_name)
    if not parsed:
        print(f"⛔️ Skipped malformed filename: {filename}")
        return

    year, country, doc_type = parsed
    cursor = conn.cursor()

    with pdfplumber.open(filepath) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text:
                continue

            paragraphs = split_paragraphs(text)

            for para_id, paragraph in enumerate(paragraphs, start=1):
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO npt_positions 
                        (year, country, doc_type, source_file, page_number, paragraph_id, full_text)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (year, country, doc_type, filename, page_num, para_id, paragraph))
                except Exception as e:
                    print(f"❗ Error inserting: {filename} p{page_num} ¶{para_id}: {e}")
    
    conn.commit()


# === STEP 5: MAIN EXECUTION ===
def main():
    conn = init_db()
    
    for folder in [STATEMENTS_DIR, WORKING_PAPERS_DIR]:
        print(f"\n📂 Processing folder: {folder}")
        folder_path = os.path.join(".", folder)
        files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

        for file in tqdm(files, desc=f"Processing {folder}"):
            filepath = os.path.join(folder_path, file)
            process_pdf(filepath, folder, conn)
    
    conn.close()
    print("\nExtraction complete. Data saved to positions.db")


if __name__ == "__main__":
    main()
