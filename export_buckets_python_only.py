#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
export_buckets_python_only.py

纯 Python 流程（不建视图、不写 SQL）：
1) 从 positions.db 读取 npt_positions 全表数据到内存
2) 解析 issues（JSON 字段），过滤空 issues、按置信度阈值分层
3) 展平（multi-label → 多行），计算简单权重
4) 以 {country}/{issue}.json 方式导出（Core 为主，必要时可合并 Supplement）
"""

import os
import re
import json
import sqlite3
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

from constants import ISSUES_LIST

# =============== 配置 ===============

from config import (
    DB_PATH,
    FULL_DIR,
    INCLUDE_SUPPLEMENT,
    CORE_THRESHOLD,
    SUPPLEMENT_LOWER,
)

# --------- 工具函数 ---------
def normalize_country(raw_country: str) -> str:
    """
    处理 country 字段冗余，如 'China_Statements', 'USA_WP' → 'China', 'USA'
    规则：
      - 去掉末尾的 _Statements / _Statement / _WP / _WorkingPapers / _Working_Papers 等
      - 只保留前面的国家名
    """
    if not raw_country:
        return raw_country
    # 常见后缀
    suffixes = [
        "_Statements", "_Statement",
        "_WP", "_WorkingPapers", "_Working_Papers", "_WorkingPaper", "_Working_Paper"
    ]
    country = raw_country
    for sfx in suffixes:
        if country.endswith(sfx):
            country = country[: -len(sfx)]
            break
    # 再兜底一次：如果还有下划线，取下划线前内容
    if "_" in country:
        country = country.split("_")[0]
    return country

def safe_json_loads(s: Optional[str]) -> Optional[Any]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    # issues 字段应为 JSON array；尽量安全解析
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
    只保留在 36 个标准议题列表内的条目，避免漂移。
    """
    if not isinstance(lst, list):
        return []
    return [i for i in lst if i in ISSUES_LIST]

def weight_from_conf(conf: Optional[float], k: int) -> Optional[float]:
    """
    简单权重：confidence / (#issues in that paragraph)
    """
    if conf is None or conf < 0 or k <= 0:
        return None
    return conf / float(k)

def safe_issue_filename(issue: str) -> str:
    """
    讲议题名变成安全文件名。
    """
    s = issue.replace(" ", "_")
    s = s.replace("/", "-")
    s = re.sub(r"[^A-Za-z0-9_\-\.]", "_", s)
    return s

# --------- 核心流程 ---------
def load_all_rows(db_path: str) -> List[Dict[str, Any]]:
    """
    从 SQLite 读取整表数据到内存（只读），后续全在 Python 中处理。
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
    展平：一条段落 → 多个 issue 行
    应用过滤逻辑：
      - issues 为空 / 非法 → 丢弃
      - confidence_score ≥ CORE_THRESHOLD → 属于 core
      - INCLUDE_SUPPLEMENT=True 时，0.60–0.70 也纳入 supplement
    返回扁平列表（每行一个 issue），字段包含：
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
        issues = canonical_issues(issues)  # 严格对齐 36 列表
        conf = safe_float(row.get("confidence_score"))

        # 过滤：空 issues 或 None
        if not issues:
            continue

        # 分类到 tier
        tier = None
        if conf is not None and conf >= CORE_THRESHOLD:
            tier = "core"
        elif INCLUDE_SUPPLEMENT and conf is not None and (SUPPLEMENT_LOWER <= conf < CORE_THRESHOLD):
            tier = "supplement"
        else:
            # 不纳入导出
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
    聚合为 country → issue → entries[]
    entries 内部按 tier 优先（core 再 supplement）、再按 confidence/weight 排序
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

    # 排序：core 优先，其次 supplement；同 tier 内按 confidence/weight/年份倒序
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
    写出为 {out_dir}/{country}/{issue}.json
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
                "topline_summary": None  # 预留字段，后续可用 LLM 生成要点摘要
            }
            fname = f"{safe_issue_filename(issue)}.json"
            path = os.path.join(country_dir, fname)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            total_files += 1
            total_entries += len(entries)

    # 产出一个汇总报告
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
    # 1) 读取全表
    rows = load_all_rows(DB_PATH)

    # 2) 展平 + 过滤（全部在 Python 内存中完成）
    exploded = explode_rows(rows)

    # 3) 按国家×议题聚合
    buckets = aggregate_buckets(exploded)

    # 4) 导出 JSON 合集
    export_buckets(buckets, FULL_DIR)

    # 打印简要统计（可删）
    countries = sorted(buckets.keys())
    print(f"   Export done to: {FULL_DIR}")
    print(f"   Countries: {countries}")
    total = sum(len(issues) for issues in buckets.values())
    print(f"   Country-Issue files: {total}")

if __name__ == "__main__":
    main()
