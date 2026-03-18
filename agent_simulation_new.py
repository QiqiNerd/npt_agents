#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
agent_simulation.py (minimal, runnable)

Implements:
1) DelegationHead reduces repetitiveness + adds explicit issue linkage sentence.
2) User control issues per turn (1–5) + user-selected Round 1 issues.
3) Recency weighting (newer year favored) in evidence ranking.
4) Simulation metadata (session year / meeting type / scenario name) shown in outputs and included in prompt.

Requirements:
  pip install openai numpy
Env:
  export OPENAI_API_KEY="..."
Data:
  outputs/agent_buckets/{Country}/{Issue}.json
  outputs/agent_buckets_lite/{Country}/{Issue}.lite.json  (recommended)
  outputs/agent_buckets_emb/{Country}/{Issue}.embeddings.npz (optional)

NOTE: This script is designed to be copied and run as-is.
"""

import os
import re
import json
import time
import math
import random
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from functools import lru_cache
from openai import OpenAI

from constants import ISSUES_LIST

from dotenv import load_dotenv
load_dotenv()

from config import *  # centralized config (with optional local overrides)

# ======================== OpenAI Client =========================

def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("❌ OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)

def call_llm(prompt: str, model: str = MODEL_TEXT, temperature: float = TEMP_TEXT) -> str:
    client = get_client()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.responses.create(
                model=model,
                input=prompt,
                temperature=temperature,
            )
            return resp.output_text
        except Exception as e:
            wait = BASE_SLEEP * attempt
            print(f"⚠️ LLM call failed (attempt {attempt}/{MAX_RETRIES}): {e}. sleeping {wait:.1f}s")
            time.sleep(wait)
    return ""

# ========================== Helpers =============================

def _safe_issue_filename(issue: str) -> str:
    s = issue.replace(" ", "_").replace("/", "-")
    return re.sub(r"[^A-Za-z0-9_\-\.]", "_", s)

@lru_cache(maxsize=8192)
def _load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def json_extract_block(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        cand = text[start:end+1]
        try:
            return json.loads(cand)
        except Exception:
            return None
    return None

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def recency_score(entry_year: Optional[int], sim_year: int = SIM_SESSION_YEAR) -> float:
    """
    Simple, explainable recency: same year => 1.0, one year older => 0.85, etc.
    """
    if entry_year is None:
        return 0.5  # neutral fallback
    delta = sim_year - int(entry_year)
    # If entry is from the future relative to sim_year, treat as 1.0
    if delta <= 0:
        return 1.0
    score = 1.0 - RECENCY_DECAY * delta
    return clamp(score, 0.0, 1.0)

# ========================== Router ==============================

def route_issues_llm(prev_round_text: str, topk: int) -> List[str]:
    prompt = f"""
You are an expert on issues surrounding the Treaty on the Nonproliferation of Nuclear Weapons (NPT). 
Your job is to classify diplomatic interventions by the issues they address, 
including interventions at meetings of the Preparatory Committee (PrepCom) and Review Conferences (RevCon) within the NPT review process.

You hear the following interventions from one or multiple delegates, in one or multiple rounds of deliberations (verbatim, may include multiple countries):
\"\"\"{prev_round_text}\"\"\"

Your job is to select up to {topk} issues from the list that are most relevant to these previous rounds of deliberations.

Issues list:
{json.dumps(ISSUES_LIST, ensure_ascii=False, indent=2)}

Return strictly valid JSON:
{{
  "issues": ["...", "..."],
  "scores": {{"Issue A": 0.82, "Issue B": 0.73}}
}}
"""
    out = call_llm(prompt)
    data = json_extract_block(out) or {}
    issues = data.get("issues") or []
    issues = [i for i in issues if i in ISSUES_LIST]
    if not issues:
        issues = random.sample(ISSUES_LIST, k=topk)
    return issues[:topk]

def route_issues_simple(prev_round_text: str, topk: int) -> List[str]:
    return random.sample(ISSUES_LIST, k=topk)

def route_issues(prev_round_text: str, topk: int) -> List[str]:
    if USE_LLM_ROUTER and prev_round_text.strip():
        return route_issues_llm(prev_round_text, topk)
    # If round 1 with empty context, router fallback:
    return route_issues_simple(prev_round_text, topk)

# ===================== Evidence Selection =======================

def _bag_of_words(text: str):
    toks = re.findall(r"[A-Za-z0-9\-]+", (text or "").lower())
    freqs = {}
    for t in toks:
        freqs[t] = freqs.get(t, 0) + 1
    return freqs

def _kw_score(query: str, doc: str) -> float:
    q = _bag_of_words(query)
    d = _bag_of_words(doc)
    score = 0.0
    for w, c in q.items():
        if w in d:
            score += math.log(1 + c) * math.log(1 + d[w])
    return score

def _rank_full_by_keywords(country: str, issue: str, context: str, top_k: int) -> List[Dict[str, Any]]:
    path = os.path.join(FULL_DIR, country, _safe_issue_filename(issue) + ".json")
    full = _load_json(path) or {}
    entries = full.get("entries", [])

    scored = []
    for e in entries:
        txt = " ".join([e.get("position_summary",""), e.get("quote",""), e.get("source_file","")]).strip()
        base = _kw_score(context, txt)

        conf = float(e.get("confidence") or 0.0)
        w = float(e.get("weight") or 0.0)
        y = e.get("year", None)
        r = recency_score(y)

        # Boss suggestion #3: recency matters
        score = base + (W_CONF * conf) + (W_RECENCY * r) + (W_WEIGHT * w)
        scored.append((score, e))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:top_k]]

def _embed_query(text: str) -> np.ndarray:
    client = get_client()
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b)/(na*nb))

def _rank_full_by_semantic(country: str, issue: str, query: str, top_k: int) -> List[Dict[str, Any]]:
    emb_path = os.path.join(EMB_DIR, country, _safe_issue_filename(issue) + ".embeddings.npz")
    if not os.path.exists(emb_path):
        return _rank_full_by_keywords(country, issue, query, top_k)

    npz = np.load(emb_path, allow_pickle=True)
    M = npz["embeddings"]
    meta = list(npz["meta"])
    q = _embed_query(query)

    sims = [ _cosine(q, M[i]) for i in range(M.shape[0]) ]
    order = np.argsort(sims)[::-1]

    full = _load_json(os.path.join(FULL_DIR, country, _safe_issue_filename(issue) + ".json")) or {"entries":[]}
    src = {(e["source_file"], e["page"], e["paragraph_id"]): e for e in full.get("entries", [])}

    ranked = []
    for idx in order[: top_k * 3]:
        m = meta[idx]

        # compatible for different formats：
        # 1) m is already dict
        # 2) m is numpy object scalar need .item()
        if hasattr(m, "item") and not isinstance(m, dict):
            try:
                m = m.item()
            except Exception:
                pass

        if not isinstance(m, dict):
            continue

    key = (m.get("source_file"), m.get("page"), m.get("paragraph_id"))
    if key in src:
        ranked.append(src[key])

    # apply recency re-ranking lightly to avoid purely semantic but old evidence
    ranked2 = []
    for e in ranked:
        conf = float(e.get("confidence") or 0.0)
        w = float(e.get("weight") or 0.0)
        r = recency_score(e.get("year"))
        score = (W_CONF * conf) + (W_RECENCY * r) + (W_WEIGHT * w)
        ranked2.append((score, e))
    ranked2.sort(key=lambda x: x[0], reverse=True)

    return [e for _, e in ranked2[:top_k]]

def select_evidence(country: str, issue: str, round_context: str, k: int) -> List[Dict[str, Any]]:
    # 1) Try lite
    lite_path = os.path.join(LITE_DIR, country, _safe_issue_filename(issue) + ".lite.json")
    lite = _load_json(lite_path)
    entries = []
    if lite:
        entries = lite.get("entries", [])[:k]

    # If no context or lite is enough:
    if not round_context.strip() or len(entries) >= k:
        return entries[:k]

    # 2) If context exists, pull top-k from full and merge (dedupe)
    ranked = _rank_full_by_semantic(country, issue, round_context, top_k=k) if USE_SEMANTIC_EVIDENCE \
             else _rank_full_by_keywords(country, issue, round_context, top_k=k)

    seen = {(e.get("source_file"), e.get("page"), e.get("paragraph_id")) for e in entries}
    for e in ranked:
        key = (e.get("source_file"), e.get("page"), e.get("paragraph_id"))
        if key not in seen:
            entries.append(e)
            seen.add(key)
        if len(entries) >= k:
            break
    return entries[:k]

def select_evidence_full_bucket(country: str, issue: str, max_entries: int, max_chars: int) -> List[Dict[str, Any]]:
    """
    Experimental mode:
    Skip retrieval ranking and pass a larger slice of the full country-issue bucket
    directly to the issue expert, with hard caps on entry count and total chars.
    """
    path = os.path.join(FULL_DIR, country, _safe_issue_filename(issue) + ".json")
    full = _load_json(path) or {}
    entries = full.get("entries", [])

    if not entries:
        return []

    selected = []
    total_chars = 0

    for e in entries[:max_entries]:
        quote = e.get("quote", "") or ""
        # lightweight estimate of prompt load from this entry
        entry_chars = len(quote) + len(e.get("source_file", "") or "") + 100

        if selected and (total_chars + entry_chars) > max_chars:
            break

        selected.append(e)
        total_chars += entry_chars

    return selected

# ======================== Agents ===============================

def issue_expert(country: str, issue: str, round_ctx: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Output JSON: {issue, text, citations[]}
    """
    ev_compact = []
    for e in evidence:
        ev_compact.append({
            "year": e.get("year"),
            "doc_type": e.get("doc_type"),
            "source": e.get("source_file"),
            "page": e.get("page"),
            "para": e.get("paragraph_id"),
            # "summary": e.get("position_summary",""),
            "quote": e.get("quote","")
        })

    prompt = f"""

        System: 
        You are a delegate representing {country} during the Nuclear Nonproliferation Treaty (NPT) review process, focused ONLY on the issue "{issue}".

        You know your country's position on this issue from previous statements, working papers, and other position documents from your government. All excerpts from these position documents that are relevant to this issue have been provided below.

        You are now representing your country as a member of your country's delegation at the simulation: {SIM_SESSION_YEAR} {SIM_MEETING_TYPE}.

        User: 
        You have heard other delegations make statements to the following effect during deliberations up to this point:
        {round_ctx}

        Country- and issue-specific position statements:
        {json.dumps(ev_compact, ensure_ascii=False)}

        Based on the context of these previous deliberations, you shall suggest a statement that accomplishes the following:
        1) Respond ONLY on "{issue}". Do NOT repeat generic opening lines.Lay out your own country’s position on this particular issue using language that is consistent with prior statements, working papers, and other position documents from your own country provided above. You shall prioritize more recent position documents but reference older documents if the position has not significantly changed.
        2) Your response must aAddress something concrete arguments advanced by other delegations during previous deliberations as provided abovefrom last round (agree/rebut/clarify/propose), and, where necessary, draw logical linkages between your issue and other related issues under discussion during previous deliberations.
        3) Cite 1–3 sources as (source_file p.X ¶Y). Do not fabricate citations.
        4) Keep it between {ISSUE_EXPERT_WORDS[0]}–{ISSUE_EXPERT_WORDS[1]} words.

        Return strictly valid JSON:
        {{
        "issue": "{issue}",
        "text": "...",
        "citations": [{{"source":"2024_USA_Statements_1.pdf","page":3,"para":2}}]
        }}
        """
    out = call_llm(prompt)
    data = json_extract_block(out) or {}
    if not isinstance(data.get("citations"), list):
        data["citations"] = []
    if "issue" not in data:
        data["issue"] = issue
    if "text" not in data:
        data["text"] = ""
    return data

def delegation_head(country: str, round_ctx: str, drafts: List[Dict[str, Any]], selected_issues: List[str], round_num: int) -> Dict[str, Any]:
    """
    Boss suggestion #1:
    - avoid repetitiveness: opening line should NOT preview all three issues
    - add explicit linkage sentence connecting issues
    """
    prompt = f"""
System: You are the head of the diplomatic delegation representing {country} at a meeting to review the Treaty on the Nonproliferation of Nuclear Weapons (NPT), 
specifically the {SIM_SCENARIO_NAME} | {SIM_SESSION_YEAR} {SIM_MEETING_TYPE} | Round {round_num}

User:
You have heard other delegations make statements to the following effect during deliberations up to this point:
{round_ctx}

You have a number of issue experts on your delegation. 
Each issue expert has drafted language you could use to address one specific issue in the statement you are about to prepare and deliver. 
You are provided with the draft(s) below:
{json.dumps(drafts, ensure_ascii=False)}

You shall now use the language in the provided draft(s) to address the following issues that they were intended to address (do not add new issues):
{json.dumps(selected_issues, ensure_ascii=False)}

Constraints (IMPORTANT):
- Acknowledge: At the top, acknowledge having heard the previous speakers’ points.
- Respond: If you are in rounds >=2, minimize the mere repetition of position statements and instead reflect upon your country’s positions on various issues, thinking of ways to respond to previous speakers convincingly and at a higher level. You should remain faithful to, and reiterate when appropriate, your own country’s positions as provided to you by your issue expert(s).
- Link: You should make explicit the linkages that exist between the selected issues (e.g., cause–effect, tradeoff, sequencing). This is not a summary; it should explain how issues intersect. Your issue expert(s) may have suggested linkages. Incorporate them appropriately.
- Reiterate: In the context of these responses and linkages, reiterate your countries’ positions at a high level. Once again, you should remain faithful to your own country’s positions as provided to you by your issue expert.
- Closing: At the end, based on all of the above, propose a reasonable basis for consensus that is highly consistent with your own country’s positions but that takes into account the foregoing deliberations, suggesting a reasonable next step/way forward.
- Total length: {DELEGATION_WORDS[0]}–{DELEGATION_WORDS[1]} words.
- Citations: merge & deduplicate; cite as (source_file p.X ¶Y). Do not invent citations.

Return strictly valid JSON:
{{
  "statement": "...",
  "citations": [{{"source":"...", "page":3, "para":2}}],
  "selected_issues": ["...", "..."]
}}
"""
    out = call_llm(prompt)
    data = json_extract_block(out) or {}
    if "statement" not in data:
        data["statement"] = ""
    if not isinstance(data.get("citations"), list):
        data["citations"] = []
    if not isinstance(data.get("selected_issues"), list):
        data["selected_issues"] = selected_issues

    # Mechanically prepend "Chair, " for the 2025 PrepCom demo
    statement = (data.get("statement") or "").lstrip()
    if str(SIM_MEETING_TYPE) == "PrepCom" and str(SIM_SESSION_YEAR) == "2025":
        if statement and not statement.startswith("Chair."):
            statement = "Chair. " + statement
    data["statement"] = statement
    return data

# ======================== Orchestrator =========================

def make_statement_header(country: str, round_num: int) -> str:
    # Boss suggestion #4: make the simulation metadata explicit to the reader
    # return f"[{SIM_SESSION_YEAR} {SIM_MEETING_TYPE} | {SIM_LOCATION}]"
    return f"[{SIM_SESSION_YEAR} {SIM_MEETING_TYPE}]"

def simulate_rounds_return_dict():
    os.makedirs(LOG_DIR, exist_ok=True)

    transcripts: List[List[Dict[str, Any]]] = []
    prev_round_text = ""  # used for routing + evidence relevance

    for t in range(1, MAX_ROUNDS + 1):
        print(f"\n===== ROUND {t} =====")
        round_records: List[Dict[str, Any]] = []

        for country in COUNTRIES:
            # Boss suggestion #2: user can set round-1 issues
            if t == 1 and ROUND1_ISSUES:
                selected_issues = [i for i in ROUND1_ISSUES if i in ISSUES_LIST][:ISSUES_PER_TURN]
                if not selected_issues:
                    selected_issues = route_issues(prev_round_text, topk=ISSUES_PER_TURN)
            else:
                selected_issues = route_issues(prev_round_text, topk=ISSUES_PER_TURN)

            drafts = []
            for issue in selected_issues:
                if EVIDENCE_MODE == "bucket_full":
                    ev = select_evidence_full_bucket(
                        country,
                        issue,
                        max_entries=FULL_BUCKET_MAX_ENTRIES,
                        max_chars=FULL_BUCKET_MAX_CHARS,
                    )
                else:
                    ev = select_evidence(country, issue, prev_round_text, k=EVIDENCE_TOPK)

                if not ev:
                    continue

                drafts.append(issue_expert(country, issue, prev_round_text, ev))

            if not drafts:
                continue
                # drafts = [{"issue": "General", "text": "No directly relevant evidence available this round.", "citations": []}]

            final = delegation_head(country, prev_round_text, drafts, selected_issues, round_num=t)

            header = make_statement_header(country, t)
            statement_with_header = header + "\n" + final.get("statement", "")

            record = {
                "round": t,
                "country": country,
                "simulation_header": header,
                "selected_issues": final.get("selected_issues", selected_issues),
                "drafts": drafts,
                "statement": statement_with_header,
                "citations": final.get("citations", [])
            }
            round_records.append(record)
            print(f" - {country} issues: {record['selected_issues']}")

        # update prev_round_text (used for round t+1 routing)
        prev_round_text = "\n\n".join([r["statement"] for r in round_records])
        transcripts.append(round_records)

    out = {
        "scenario": {
            "name": SIM_SCENARIO_NAME,
            "session_year": SIM_SESSION_YEAR,
            "meeting_type": SIM_MEETING_TYPE,
            "location": SIM_LOCATION
        },
        "config": {
            "countries": COUNTRIES,
            "max_rounds": MAX_ROUNDS,
            "issues_per_turn": ISSUES_PER_TURN,
            "round1_issues": ROUND1_ISSUES,
            "evidence_topk": EVIDENCE_TOPK,
            "use_llm_router": USE_LLM_ROUTER,
            "use_semantic_evidence": USE_SEMANTIC_EVIDENCE,
            "recency_decay": RECENCY_DECAY,
            "weights": {"conf": W_CONF, "recency": W_RECENCY, "weight": W_WEIGHT},
            "evidence_mode": EVIDENCE_MODE,
            "full_bucket_max_entries": FULL_BUCKET_MAX_ENTRIES if EVIDENCE_MODE == "bucket_full" else None,
            "full_bucket_max_chars": FULL_BUCKET_MAX_CHARS if EVIDENCE_MODE == "bucket_full" else None
        },
        "transcripts": transcripts,
        "generated_at": datetime.utcnow().isoformat() + "Z"
    }

    return out
    # out_path = os.path.join(LOG_DIR, f"npt_sim_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
    # with open(out_path, "w", encoding="utf-8") as f:
    #     json.dump(out, f, ensure_ascii=False, indent=2)

    # print(f"\n✅ Simulation saved to {out_path}")

def simulate_rounds():
    """
    CLI / local usage:
    generate simulation and save to outputs/simulation_logs/
    """
    out = simulate_rounds_return_dict()

    os.makedirs("outputs/simulation_logs", exist_ok=True)
    path = os.path.join(
        "outputs/simulation_logs",
        f"npt_sim_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Simulation saved to {path}")

def run_simulation(config: dict) -> dict:
    """
    This is the function Streamlit will call.
    config example:
    {
        "countries": ["USA", "China"],
        "max_rounds": 2,
        "issues_per_turn": 3,
        "round1_issues": [...]
    }
    """
    global COUNTRIES, MAX_ROUNDS, ISSUES_PER_TURN, ROUND1_ISSUES
    global EVIDENCE_MODE, FULL_BUCKET_MAX_ENTRIES, FULL_BUCKET_MAX_CHARS

    if "evidence_mode" in config:
        EVIDENCE_MODE = config["evidence_mode"]
    if "full_bucket_max_entries" in config:
        FULL_BUCKET_MAX_ENTRIES = int(config["full_bucket_max_entries"])
    if "full_bucket_max_chars" in config:
        FULL_BUCKET_MAX_CHARS = int(config["full_bucket_max_chars"])

    if "countries" in config:
        COUNTRIES = config["countries"]
    if "max_rounds" in config:
        MAX_ROUNDS = int(config["max_rounds"])
    if "issues_per_turn" in config:
        ISSUES_PER_TURN = int(config["issues_per_turn"])
    if "round1_issues" in config:
        ROUND1_ISSUES = config["round1_issues"]

    return simulate_rounds_return_dict()


# ============================== MAIN ============================

if __name__ == "__main__":
    # Basic safety check for ISSUES_PER_TURN
    if not (1 <= ISSUES_PER_TURN <= 5):
        raise ValueError("ISSUES_PER_TURN must be between 1 and 5.")
    simulate_rounds()
