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

# ============================ CONFIG ============================

# --- Simulation metadata (Boss suggestion #4) ---
SIM_SCENARIO_NAME = "NPT Review Process Agent Simulation"
SIM_SESSION_YEAR = 2025                 # anchor year for recency + "what meeting is this"
SIM_MEETING_TYPE = "PrepCom"            # "PrepCom" or "RevCon" (or any label you want)
SIM_LOCATION = "Geneva (simulated)"     # optional display string

# --- Countries / rounds ---
COUNTRIES = ["USA"]
MAX_ROUNDS = 1

# --- User control (Boss suggestion #2) ---
ISSUES_PER_TURN = 1   # user chooses 1..5
ROUND1_ISSUES = None  # e.g. ["Fulfillment of Article VI disarmament obligations", "Risk reduction and confidence-building measures"]
# If set, round 1 will use these issues (up to ISSUES_PER_TURN). If None, round 1 uses router.

# --- Evidence / retrieval ---
EVIDENCE_TOPK = 3
FULL_DIR = "outputs/agent_buckets"
LITE_DIR = "outputs/agent_buckets_lite"
EMB_DIR = "outputs/agent_buckets_emb"

# --- Router strategy ---
USE_LLM_ROUTER = True       # if False: random fallback
USE_SEMANTIC_EVIDENCE = True
EMBEDDING_MODEL = "text-embedding-3-large"

# --- LLM for text generation ---
MODEL_TEXT = "gpt-5.2"
TEMP_TEXT = 0.2
MAX_RETRIES = 4
BASE_SLEEP = 2.0

# --- Output length constraints (soft) ---
ISSUE_EXPERT_WORDS = (110, 150)
DELEGATION_WORDS = (180, 230)

# --- Logging ---
LOG_DIR = "outputs/simulation_logs"

# --- Evidence scoring weights (Boss suggestion #3) ---
# total doesn't need to sum to 1, but clearer if it does
W_CONF = 0.55
W_RECENCY = 0.30
W_WEIGHT = 0.15

# Recency decay per year (simple linear decay)
RECENCY_DECAY = 0.15  # 1 year older => -0.15

# ===============================================================

ISSUES_LIST = [
    "Treaty on the Prohibition of Nuclear Weapons",
    "Quantitative and qualitative expansion of nuclear arsenals",
    "Humanitarian consequences of nuclear weapon use",
    "Transparency and accountability of nuclear arsenals and doctrines",
    "Role and significance of nuclear weapons in military and security concepts, doctrines and policies",
    "Arms control agreements",
    "Disarmament verification",
    "Reduced role and operational readiness of nuclear weapons",
    "Fulfillment of Article VI disarmament obligations",
    "Security assurances",
    "No first use",
    "Risk reduction and confidence-building measures",
    "Comprehensive Nuclear-Test-Ban Treaty",
    "Moratorium on nuclear testing",
    "The Fissile Material Cutoff Treaty",
    "Moratorium on fissile material production",
    "Legacy of nuclear weapons, their use and testing",
    "Gender",
    "Emerging and disruptive technologies",
    "Nonproliferation and disarmament education",
    "Middle East Weapons of Mass Destruction Free Zone and Israel",
    "Universality of the Treaty on the Non-Proliferation of Nuclear Weapons",
    "Nuclear-Weapon-Free Zones",
    "International Atomic Energy Agency safeguards",
    "Export controls",
    "Regional proliferation challenges including the the Democratic People's Republic of Korea, Iran and Joint Comprehensive Plan of Action",
    "Nuclear threats",
    "Attacks on nuclear facilities",
    "Peaceful uses of nuclear technology",
    "Nuclear safety",
    "Nuclear security",
    "Strengthening the Treaty on the Non-Proliferation of Nuclear Weapons review process",
    "Discouraging the Treaty on the Non-Proliferation of Nuclear Weapons withdrawal",
    "Naval propulsion",
    "Nuclear sharing and extended deterrence",
    "Ukraine"
]

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
You classify NPT discussion topics for routing.

Simulation context: {SIM_SESSION_YEAR} {SIM_MEETING_TYPE} (scenario: {SIM_SCENARIO_NAME}).

Select up to {topk} most relevant issues from the list, given the previous round interventions.

Issues list:
{json.dumps(ISSUES_LIST, ensure_ascii=False, indent=2)}

Previous round interventions (verbatim, may include multiple countries):
\"\"\"{prev_round_text}\"\"\"

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
    for idx in order[: top_k*3]:
        m = meta[idx].item()
        key = (m["source_file"], m["page"], m["paragraph_id"])
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
            "summary": e.get("position_summary",""),
            "quote": e.get("quote","")
        })

    prompt = f"""
System: You are the {country} NPT delegate focused ONLY on the issue "{issue}".
You must reflect the country's established positions using the evidence pack and cite sources.

Simulation: {SIM_SCENARIO_NAME} | {SIM_SESSION_YEAR} {SIM_MEETING_TYPE} | Location: {SIM_LOCATION}

User:
Last round context (other countries' highlights):
{round_ctx}

Evidence pack (ranked):
{json.dumps(ev_compact, ensure_ascii=False)}

Instructions:
1) Respond ONLY on "{issue}". Do NOT repeat generic opening lines.
2) Your response must address something concrete from last round (agree/rebut/clarify/propose).
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
System: You are the head of the {country} NPT delegation. Produce ONE unified intervention.

Simulation: {SIM_SCENARIO_NAME} | {SIM_SESSION_YEAR} {SIM_MEETING_TYPE} | Round {round_num}

User:
Other delegations' last round highlights:
{round_ctx}

Issue expert drafts (each only covers its issue):
{json.dumps(drafts, ensure_ascii=False)}

You must address these issues (do not add new ones):
{json.dumps(selected_issues, ensure_ascii=False)}

Constraints (IMPORTANT):
- Reduce repetition. In rounds >=2, do NOT write a first paragraph that previews all issue-specific content.
- Opening: 1 sentence ONLY, referencing prior speakers and explicitly situating this as a simulated intervention at the {SIM_SESSION_YEAR} {SIM_MEETING_TYPE}. No detailed issue claims here.
- Linkage: 1 sentence that CONNECTS the selected issues (cause–effect, tradeoff, sequencing). This is not a summary; it should explain how issues intersect.
- Body: 1 short paragraph per issue (2–4 sentences each). Do NOT repeat the opening or linkage content verbatim.
- Closing: 1 sentence proposing next step / way forward.
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
    return data

# ======================== Orchestrator =========================

def make_statement_header(country: str, round_num: int) -> str:
    # Boss suggestion #4: make the simulation metadata explicit to the reader
    return f"[{SIM_SCENARIO_NAME} | {SIM_SESSION_YEAR} {SIM_MEETING_TYPE} | {SIM_LOCATION} | Round {round_num} | {country}]"

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
                ev = select_evidence(country, issue, prev_round_text, k=EVIDENCE_TOPK)
                if not ev:
                    continue
                drafts.append(issue_expert(country, issue, prev_round_text, ev))

            if not drafts:
                drafts = [{"issue": "General", "text": "No directly relevant evidence available this round.", "citations": []}]

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
            "weights": {"conf": W_CONF, "recency": W_RECENCY, "weight": W_WEIGHT}
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
