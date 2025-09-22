#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
agent_simulation.py

多国 NPT 代理模拟（单文件版本）：
- Orchestrator：多国多轮回合
- DelegationHead：抽取议题、调度专家、合成发言
- IssueExpert：按议题+证据生成回应并引证
- Evidence Selector：先 lite，必要时 full 里 Top-K
- 可选语义检索（如已构建 embeddings 索引）

依赖：
  pip install openai
环境变量：
  export OPENAI_API_KEY

文件依赖（建议先准备）：
  outputs/agent_buckets/{Country}/{Issue}.json
  outputs/agent_buckets_lite/{Country}/{Issue}.lite.json   (可选，推荐)
  outputs/agent_buckets_emb/{Country}/{Issue}.embeddings.npz (可选，语义检索)
"""

import os
import re
import json
import time
import math
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from functools import lru_cache
from openai import OpenAI

# ============================ 配置 ============================
# 会期设置
COUNTRIES = ["USA", "China", "Russia"]       # 你可以扩展
MAX_ROUNDS = 3                               # 回合数
ROUTER_TOPK = 3                              # 每轮选择的议题数（最多3）
EVIDENCE_TOPK = 3                            # 每议题取证据条数（每位专家）

# 证据目录
FULL_DIR = "outputs/agent_buckets"
LITE_DIR = "outputs/agent_buckets_lite"      # 若无则只用 FULL
EMB_DIR  = "outputs/agent_buckets_emb"       # 可选语义索引目录

# 模型与调用
MODEL_TEXT = "gpt-5"
TEMP_TEXT  = 0.2
MAX_RETRIES = 4
BASE_SLEEP  = 2.0

# Router 选择策略
USE_LLM_ROUTER = True        # True: 用 LLM 多标签路由；False: 简易随机/关键词示例
USE_SEMANTIC_EVIDENCE = False # True: 用 embeddings 语义检索（需先构建索引）
EMBEDDING_MODEL = "text-embedding-3-large"

# 成本控制（只是提示词限制，非硬限制）
ISSUE_EXPERT_WORDS = (120, 150)   # 议题专家输出字数范围
DELEGATION_WORDS   = (180, 220)   # 团长合并输出字数范围

# 输出
LOG_DIR = "outputs/simulation_logs"
# ============================================================

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

# ======================== OpenAI 客户端 =======================
def get_client() -> OpenAI:
    api_key = api_key=os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("❌ OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)

def call_llm(prompt: str, model: str = MODEL_TEXT, temperature: float = TEMP_TEXT) -> str:
    client = get_client()
    for attempt in range(1, MAX_RETRIES+1):
        try:
            resp = client.responses.create(
                model=model,
                input=prompt,
                # temperature=temperature
            )
            return resp.output_text
        except Exception as e:
            wait = BASE_SLEEP * attempt
            print(f"⚠️ LLM call failed (attempt {attempt}/{MAX_RETRIES}): {e}. sleep {wait:.1f}s")
            time.sleep(wait)
    return ""

# ========================= 工具函数 ===========================
def _safe_issue_filename(issue: str) -> str:
    s = issue.replace(" ", "_").replace("/", "-")
    return re.sub(r"[^A-Za-z0-9_\-\.]", "_", s)

@lru_cache(maxsize=4096)
def _load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def json_extract_block(text: str) -> Optional[dict]:
    if not text:
        return None
    # 尽量解析为 JSON
    try:
        return json.loads(text)
    except Exception:
        pass
    # 退而求其次：截取第一个 {...}
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None

# ========================= 议题路由 ===========================
def route_issues_from_context_llm(prev_round_summary: str, topk: int = ROUTER_TOPK) -> List[str]:
    prompt = f"""
You classify NPT discussion topics.

Select up to {topk} most relevant issues from the list given the previous round summary.

Issues list:
{json.dumps(ISSUES_LIST, ensure_ascii=False, indent=2)}

Previous round summary:
\"\"\"{prev_round_summary}\"\"\"

Return strictly valid JSON:
{{
  "issues": ["...", "..."],
  "scores": {{"Issue A": 0.82, "Issue B": 0.73}}
}}
"""
    out = call_llm(prompt)
    data = json_extract_block(out) or {}
    issues = data.get("issues") or []
    # 兜底
    if not issues:
        issues = random.sample(ISSUES_LIST, k=min(topk, len(ISSUES_LIST)))
    return issues[:topk]

def route_issues_from_context_simple(prev_round_summary: str, topk: int = ROUTER_TOPK) -> List[str]:
    # 简易随机（可换关键词 → 议题映射）
    return random.sample(ISSUES_LIST, k=min(topk, len(ISSUES_LIST)))

def route_issues(prev_round_summary: str, topk: int = ROUTER_TOPK) -> List[str]:
    if USE_LLM_ROUTER:
        return route_issues_from_context_llm(prev_round_summary, topk=topk)
    return route_issues_from_context_simple(prev_round_summary, topk=topk)

# ========================= 证据检索 ===========================
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

def _rank_full_by_keywords(country: str, issue: str, context: str, top_k: int):
    path = os.path.join(FULL_DIR, country, _safe_issue_filename(issue) + ".json")
    full = _load_json(path)
    if not full:
        return []
    entries = full.get("entries", [])
    scored = []
    for e in entries:
        txt = " ".join([
            e.get("position_summary", ""), e.get("quote", ""), e.get("source_file", "")
        ])
        base = _kw_score(context, txt)
        conf = e.get("confidence") or 0.0
        w = e.get("weight") or 0.0
        score = base + 0.2*conf + 0.1*w
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

def _rank_full_by_semantic(country: str, issue: str, query: str, top_k: int):
    emb_path = os.path.join(EMB_DIR, country, _safe_issue_filename(issue) + ".embeddings.npz")
    if not os.path.exists(emb_path):
        return _rank_full_by_keywords(country, issue, query, top_k)
    npz = np.load(emb_path, allow_pickle=True)
    M = npz["embeddings"]       # (N, D)
    meta = list(npz["meta"])    # list of dicts
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
    return ranked[:top_k]

def select_evidence(country: str, issue: str, round_context: str, k: int = EVIDENCE_TOPK) -> List[Dict[str, Any]]:
    # 1) 先尝试 lite
    lite_path = os.path.join(LITE_DIR, country, _safe_issue_filename(issue) + ".lite.json")
    lite = _load_json(lite_path)
    entries = []
    if lite:
        entries = lite.get("entries", [])[:k]

    # 2) 如果上下文为空或 lite 足够，就返回
    if not round_context or len(entries) >= k:
        return entries[:k]

    # 3) 上下文非空 → 从 full 做相关性 Top-K（关键词或语义）
    need = k - len(entries)
    ranked = _rank_full_by_semantic(country, issue, round_context, top_k=k) if USE_SEMANTIC_EVIDENCE \
             else _rank_full_by_keywords(country, issue, round_context, top_k=k)
    # 去重
    seen = {(e["source_file"], e["page"], e["paragraph_id"]) for e in entries}
    for e in ranked:
        key = (e["source_file"], e["page"], e["paragraph_id"])
        if key not in seen:
            entries.append(e)
            seen.add(key)
        if len(entries) >= k:
            break
    return entries[:k]

# ======================== Issue Expert =======================
def issue_expert(country: str, issue: str, round_ctx: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    ev_compact = [
        {
            "source": e.get("source_file"),
            "page": e.get("page"),
            "para": e.get("paragraph_id"),
            "summary": e.get("position_summary",""),
            "quote": e.get("quote",""),
            "confidence": e.get("confidence"),
            "tier": e.get("tier")
        } for e in evidence
    ]
    prompt = f"""
System: You are the {country} NPT delegate focused on the issue: "{issue}".
You must produce concise, diplomatic language reflecting the country's established positions. Use the provided evidence and cite sources.

User:
Round context (other countries' main points):
{round_ctx}

Your evidence pack (ranked, max {len(evidence)}):
{json.dumps(ev_compact, ensure_ascii=False)}

Instructions:
1) Respond ONLY on "{issue}". If the context touches other issues, briefly signpost but stay on-topic.
2) Synthesize a clear stance that responds to the last round (agree/rebut/clarify/propose).
3) Include 1–3 short citations using (source_file p.X ¶Y). Do not fabricate citations.
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
    # 兜底
    if "issue" not in data:
        data["issue"] = issue
    if "text" not in data:
        data["text"] = ""
    if "citations" not in data or not isinstance(data["citations"], list):
        data["citations"] = []
    return data

# ======================= Delegation Head ======================
def delegation_head(country: str, round_ctx: str, drafts: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompt = f"""
System: You are the head of the {country} NPT delegation. Produce one unified country statement.

User:
Round context (other countries highlights):
{round_ctx}

Drafts from your issue experts:
{json.dumps(drafts, ensure_ascii=False)}

Constraints:
- Keep the country's established positions consistent.
- Merge overlapping content; remove redundancy.
- Structure:
  (1) Opening sentence referencing prior speakers.
  (2) 2–3 short paragraphs, one per selected issue.
  (3) Closing line indicating next steps or a proposal.
- Keep it between {DELEGATION_WORDS[0]}–{DELEGATION_WORDS[1]} words total.
- Merge and deduplicate citations. Use format: (source_file p.X ¶Y)

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
    if "citations" not in data or not isinstance(data["citations"], list):
        data["citations"] = []
    if "selected_issues" not in data or not isinstance(data["selected_issues"], list):
        data["selected_issues"] = [d.get("issue","") for d in drafts]
    return data

# ========================= Orchestrator =======================
def simulate_rounds():
    os.makedirs(LOG_DIR, exist_ok=True)
    transcripts: List[List[Dict[str, Any]]] = []
    prev_round_text = ""  # 上一轮全部发言拼接（供 Router 使用）

    for t in range(1, MAX_ROUNDS + 1):
        round_records: List[Dict[str, Any]] = []
        print(f"\n===== ROUND {t} =====")
        for country in COUNTRIES:
            # 1) 路由：从上一轮文本中选出最多 3 个议题
            selected_issues = route_issues(prev_round_text, topk=ROUTER_TOPK)
            # 2) 逐议题取证据并由专家生成草案
            drafts = []
            for issue in selected_issues:
                ev = select_evidence(country, issue, prev_round_text, k=EVIDENCE_TOPK)
                if not ev:
                    # 该议题无证据，跳过或换下一个议题
                    continue
                draft = issue_expert(country, issue, prev_round_text, ev)
                drafts.append(draft)
            if not drafts:
                drafts = [{"issue": "General", "text": "No directly relevant evidence available this round.", "citations": []}]

            # 3) 团长合并为国家发言
            final = delegation_head(country, prev_round_text, drafts)
            record = {
                "round": t,
                "country": country,
                "selected_issues": final.get("selected_issues", [d.get("issue","") for d in drafts]),
                "statement": final.get("statement", ""),
                "citations": final.get("citations", []),
                "drafts": drafts
            }
            round_records.append(record)
            print(f" - {country} issues: {record['selected_issues']}")

        # 4) 汇总本轮文本，供下一轮路由参考
        prev_round_text = "\n\n".join([f"{r['country']}: {r['statement']}" for r in round_records])
        transcripts.append(round_records)

        # （可选）简单停止条件：若上一轮与本轮议题集合高度重合、且文本变化很小，可提前停止
        # TODO：可加入“共识度评分/变化率阈值”

    # 保存对话日志
    out = {
        "countries": COUNTRIES,
        "max_rounds": MAX_ROUNDS,
        "use_llm_router": USE_LLM_ROUTER,
        "use_semantic_evidence": USE_SEMANTIC_EVIDENCE,
        "transcripts": transcripts,
        "generated_at": datetime.utcnow().isoformat() + "Z"
    }
    out_path = os.path.join(LOG_DIR, f"npt_sim_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Simulation saved to {out_path}")

# ============================ 入口 ============================
if __name__ == "__main__":
    simulate_rounds()
    pass
