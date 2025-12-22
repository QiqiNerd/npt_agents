#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dotenv import load_dotenv
load_dotenv()
import os
import re
import json
import time
import sqlite3
from typing import Optional, Tuple, List
from tqdm import tqdm
from openai import OpenAI

from constants import ISSUES_LIST

from config import (
    DB_PATH,
    MODEL_CLASSIFY,
    MAX_RETRIES,
    BASE_SLEEP,
    BATCH_SIZE,
    MAX_ROWS,
    FAIL_LOG,
)
# ============================================================


PROMPT_TEMPLATE = """You are an expert on the Nuclear Non-Proliferation Treaty (NPT).

Task:
1) Identify which of the 36 NPT issues this paragraph addresses (multi-label allowed).
2) Summarize the country's position on those issue(s) in 1–3 sentences.
3) Provide a confidence score between 0.0 and 1.0.

Issues list:
{issues_list}

Paragraph:
\"\"\"{paragraph}\"\"\"

Return strictly valid JSON (no prose, no markdown). Use this schema:
{{
  "issues": ["...", "..."],
  "position_summary": "...",
  "confidence_score": 0.85
}}
"""

def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("❌ OPENAI_API_KEY environment variable is not set.")
    return OpenAI(api_key=api_key)

def build_prompt(paragraph_text: str) -> str:
    return PROMPT_TEMPLATE.format(
        issues_list=json.dumps(ISSUES_LIST, indent=2, ensure_ascii=False),
        paragraph=paragraph_text
    ).strip()

def call_gpt(client: OpenAI, prompt: str) -> Optional[str]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.responses.create(
                model=MODEL_CLASSIFY,
                input=prompt,
            )
            return resp.output_text
        except Exception as e:
            wait = BASE_SLEEP * attempt
            print(f"⚠️ GPT call failed (attempt {attempt}/{MAX_RETRIES}): {e}. Sleeping {wait:.1f}s")
            time.sleep(wait)
    return None

def extract_json_block(text: str) -> Optional[str]:
    if not text:
        return None
    try:
        json.loads(text)
        return text
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1].strip()
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass
    return None

def parse_model_output(raw: str) -> Optional[Tuple[List[str], str, float]]:
    js = extract_json_block(raw)
    if not js:
        return None
    try:
        data = json.loads(js)
        issues = data.get("issues", [])
        if issues is None:
            issues = []
        if not isinstance(issues, list):
            issues = [issues] if isinstance(issues, str) else []
        summary = (data.get("position_summary") or "").strip()
        score = data.get("confidence_score", 0.0)
        try:
            score = float(score)
        except Exception:
            score = 0.0
        issues = [i for i in issues if i in ISSUES_LIST]
        return issues, summary, score
    except Exception:
        return None

def fetch_next_batch(conn: sqlite3.Connection, limit: int):
    cur = conn.cursor()
    cur.execute(
        "SELECT id, full_text FROM npt_positions WHERE issues IS NULL ORDER BY id LIMIT ?",
        (limit,)
    )
    return cur.fetchall()

def update_row(conn: sqlite3.Connection, row_id: int, issues: List[str], summary: str, score: float):
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE npt_positions
        SET issues = ?, position_summary = ?, confidence_score = ?
        WHERE id = ?
        """,
        (json.dumps(issues, ensure_ascii=False), summary, score, row_id)
    )
    conn.commit()

def log_failure(row_id: int, text: str, raw_output: Optional[str], reason: str):
    rec = {
        "id": row_id,
        "reason": reason,
        "full_text": text,
        "raw_output": raw_output
    }
    with open(FAIL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def classify_loop():
    client = get_client()
    conn = sqlite3.connect(DB_PATH)
    processed_total = 0

    while True:
        if MAX_ROWS is not None and processed_total >= MAX_ROWS:
            break
        this_limit = BATCH_SIZE if MAX_ROWS is None else min(BATCH_SIZE, MAX_ROWS - processed_total)
        batch = fetch_next_batch(conn, this_limit)
        if not batch:
            break
        for row_id, text in tqdm(batch, desc=f"Processed {processed_total} so far"):
            prompt = build_prompt(text)
            raw = call_gpt(client, prompt)
            if raw is None:
                log_failure(row_id, text, raw_output=None, reason="model_call_failed")
                continue
            parsed = parse_model_output(raw)
            if not parsed:
                log_failure(row_id, text, raw_output=raw, reason="json_parse_failed")
                continue
            issues, summary, score = parsed
            try:
                update_row(conn, row_id, issues, summary, score)
            except Exception as e:
                log_failure(row_id, text, raw_output=raw, reason=f"db_update_failed: {e}")
                continue
            processed_total += 1
        time.sleep(0.5)

    conn.close()
    print(f"✅ Done. Total processed this run: {processed_total}")

if __name__ == "__main__":
    classify_loop()
