#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
export_buckets_python_only.py

Pure Python process (no views created, no SQL written):
1) Read all data from the `npt_positions` table in `positions.db` into memory.
2) Parse the `issues` (JSON field), filter out empty issues, and categorize them based on confidence thresholds.
3) Flatten the data (multi-label → multiple rows), and calculate simple weights.
4) Export the data as `{country}/{issue}.json` (primarily using Core data, merging with Supplement data when necessary).
"""

import os
import re
import json
import sqlite3
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

from constants import ISSUES_LIST

# =============== Config ===============

from config import (
    DB_PATH,
    FULL_DIR,
    INCLUDE_SUPPLEMENT,
    CORE_THRESHOLD,
    SUPPLEMENT_LOWER,
)

# --------- Utility functions ---------
def normalize_country(raw_country: str) -> str:
    """
    Address redundancy in the 'country' field, e.g., 'China_Statements', 'USA_WP' → 'China', 'USA'
    Rules:
    - Remove suffixes such as _Statements, _Statement, _WP, _WorkingPapers, _Working_Papers, etc.
    - Only retain the country name at the beginning.
    """
    if not raw_country:
        return raw_country
    # Common suffixes
    suffixes = [
        "_Statements", "_Statement",
        "_WP", "_WorkingPapers", "_Working_Papers", "_WorkingPaper", "_Working_Paper"
    ]
    country = raw_country
    for sfx in suffixes:
        if country.endswith(sfx):
            country = country[: -len(sfx)]
            break
    # One last clarification: If there's still an underscore, take the content before the underscore.
    if "_" in country:
        country = country.split("_")[0]
    return country

def safe_json_loads(s: Optional[str]) -> Optional[Any]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    # The `issues` field should be a JSON array; parse it as safely as possible.
    try:
        return json.loads(s)
    except Exception:
        return None

def safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def canonical_issues(lst) -> List[str]:
    """
    Only retain items that are within the list of 36 standard topics to prevent scope creep.
    """
    if not isinstance(lst, list):
        return []
    return [i for i in lst if i in ISSUES_LIST]

def weight_from_conf(conf: Optional[float], k: int) -> Optional[float]:
    """
    Simple weighting: confidence / (# of issues in that paragraph)
    """
    if conf is None or conf < 0 or k <= 0:
        return None
    return conf / float(k)

def safe_issue_filename(issue: str) -> str:
    """
    The topic title is converted into a safe filename.
    """
    s = issue.replace(" ", "_")
    s = s.replace("/", "-")
    s = re.sub(r"[^A-Za-z0-9_\-\.]", "_", s)
    return s

# --------- core process ---------
def load_all_rows(db_path: str) -> List[Dict[str, Any]]:
    """
    The entire table data is read from SQLite into memory (read-only), and all subsequent processing is done in Python.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT
            id, year, country, doc_type, source_file, page_number, paragraph_id,
            full_text, issues, position_summary, confidence_score
        FROM npt_positions
        ORDER BY id
    """)
    rows = cur.fetchall()
    conn.close()

    cols = ["id","year","country","doc_type","source_file","page_number","paragraph_id",
            "full_text","issues","position_summary","confidence_score"]
    out = []
    for r in rows:
        item = dict(zip(cols, r))
        out.append(item)
    return out

def explode_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Flattening: One paragraph → Multiple issue lines
    Applying filtering logic:
    - Empty/invalid issues → Discard
    - confidence_score ≥ CORE_THRESHOLD → Belongs to core
    - When INCLUDE_SUPPLEMENT=True, 0.60–0.70 are also included in the supplement
    Returns a flattened list (one issue per line), with fields including:
    id, year, country_norm, doc_type, source_file, page_number, paragraph_id,
    issue, position_summary, full_text, confidence_score, weight, tier
    """
    exploded = []
    for row in rows:
        raw_country = row.get("country")
        country_norm = normalize_country(raw_country)
        doc_type = (row.get("doc_type") or "").strip()  # 'statement' / 'working_paper'
        issues_raw = row.get("issues")
        issues = safe_json_loads(issues_raw)
        issues = canonical_issues(issues)  # Strictly align the 36 items in the list.
        conf = safe_float(row.get("confidence_score"))

        # Filter: Empty issues or None
        if not issues:
            continue

        # Categorized into tiers
        tier = None
        if conf is not None and conf >= CORE_THRESHOLD:
            tier = "core"
        elif INCLUDE_SUPPLEMENT and conf is not None and (SUPPLEMENT_LOWER <= conf < CORE_THRESHOLD):
            tier = "supplement"
        else:
            # Not included in the export.
            continue

        k = len(issues)
        w = weight_from_conf(conf, k)

        for issue in issues:
            exploded.append({
                "id": row.get("id"),
                "year": row.get("year"),
                "country_norm": country_norm,
                "doc_type": doc_type,
                "source_file": row.get("source_file"),
                "page_number": row.get("page_number"),
                "paragraph_id": row.get("paragraph_id"),
                "issue": issue,
                "position_summary": row.get("position_summary") or "",
                "full_text": row.get("full_text") or "",
                "confidence_score": conf,
                "weight": w,
                "tier": tier
            })
    return exploded

def aggregate_buckets(exploded: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Aggregated as country → issue → entries[]
    Entries are sorted internally by tier (core first, then supplement), and then by confidence/weight.
    """
    buckets = defaultdict(lambda: defaultdict(list))
    for rec in exploded:
        c = rec["country_norm"]
        issue = rec["issue"]
        entry = {
            "id": rec["id"],
            "year": rec["year"],
            "doc_type": rec["doc_type"],
            "source_file": rec["source_file"],
            "page": rec["page_number"],
            "paragraph_id": rec["paragraph_id"],
            "position_summary": rec["position_summary"],
            "quote": rec["full_text"],
            "confidence": rec["confidence_score"],
            "weight": rec["weight"],
            "tier": rec["tier"]
        }
        buckets[c][issue].append(entry)

    # Sorting order: core items first, followed by supplementary items; within the same tier
    # sort in descending order by confidence/weight/year.
    def sort_key(e):
        tier_rank = 0 if e["tier"] == "core" else 1
        conf = e["confidence"] if e["confidence"] is not None else -1
        w = e["weight"] if e["weight"] is not None else -1
        year = e["year"] if e["year"] is not None else -1
        return (tier_rank, -conf, -w, -year, e["id"])

    for c in buckets:
        for issue in buckets[c]:
            buckets[c][issue].sort(key=sort_key)

    return buckets

def export_buckets(buckets: Dict[str, Dict[str, List[Dict[str, Any]]]], out_dir: str):
    """
    Writing to {out_dir}/{country}/{issue}.json
    """
    ts = datetime.utcnow().isoformat() + "Z"
    os.makedirs(out_dir, exist_ok=True)
    total_files = 0
    total_entries = 0

    for country, issues_map in buckets.items():
        country_dir = os.path.join(out_dir, country)
        os.makedirs(country_dir, exist_ok=True)
        for issue, entries in issues_map.items():
            data = {
                "country": country,
                "issue": issue,
                "updated_at": ts,
                "entries": entries,
                # Reserved field; can be used later to generate key point summaries using LLM (Large Language Model).
                "topline_summary": None 
            }
            fname = f"{safe_issue_filename(issue)}.json"
            path = os.path.join(country_dir, fname)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            total_files += 1
            total_entries += len(entries)

    # Generate a summary report.
    report = {
        "exported_files": total_files,
        "total_entries": total_entries,
        "core_threshold": CORE_THRESHOLD,
        "include_supplement": INCLUDE_SUPPLEMENT,
        "supplement_lower": SUPPLEMENT_LOWER if INCLUDE_SUPPLEMENT else None,
        "generated_at": ts
    }
    with open(os.path.join(out_dir, "_export_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

def main():
    # 1) Read the entire table.
    rows = load_all_rows(DB_PATH)

    # 2) Flattening + Filtering (all performed in Python memory)
    exploded = explode_rows(rows)

    # 3) Aggregate by country and topic
    buckets = aggregate_buckets(exploded)

    # 4) Export JSON collection
    export_buckets(buckets, FULL_DIR)

    # Print brief statistics (can be deleted)
    countries = sorted(buckets.keys())
    print(f"   Export done to: {FULL_DIR}")
    print(f"   Countries: {countries}")
    total = sum(len(issues) for issues in buckets.values())
    print(f"   Country-Issue files: {total}")

if __name__ == "__main__":
    main()
