# streamlit_app.py
import os
import json
import glob
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st

from constants import ISSUES_LIST

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="NPT Agent Simulation Viewer",
    layout="wide",
)

# -----------------------------
# Helpers
# -----------------------------
def load_json_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_latest_log(log_dir: str = "outputs/simulation_logs") -> Optional[str]:
    paths = sorted(glob.glob(os.path.join(log_dir, "npt_sim_*.json")))
    return paths[-1] if paths else None

def normalize_transcripts(data: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    """
    Expects schema like:
    {
      "transcripts": [
        [ {round,country,selected_issues,statement,citations,...}, ... ],  # round 1
        [ ... ],  # round 2
      ],
      "scenario": {...},
      "config": {...}
    }
    """
    transcripts = data.get("transcripts")
    if not transcripts or not isinstance(transcripts, list):
        return []
    return transcripts

def collect_countries(transcripts: List[List[Dict[str, Any]]]) -> List[str]:
    s = set()
    for rnd in transcripts:
        for rec in rnd:
            c = rec.get("country")
            if c:
                s.add(c)
    return sorted(list(s))

def find_record(transcripts: List[List[Dict[str, Any]]], round_num: int, country: str) -> Optional[Dict[str, Any]]:
    if round_num < 1 or round_num > len(transcripts):
        return None
    for rec in transcripts[round_num - 1]:
        if rec.get("country") == country:
            return rec
    return None

def render_meta(data: Dict[str, Any], transcripts: List[List[Dict[str, Any]]]) -> None:
    scenario = data.get("scenario", {}) or {}
    config = data.get("config", {}) or {}

    name = scenario.get("name", "NPT Agent Simulation")
    session_year = scenario.get("session_year", config.get("SIM_SESSION_YEAR", "—"))
    meeting_type = scenario.get("meeting_type", config.get("SIM_MEETING_TYPE", "—"))
    location = scenario.get("location", "—")

    countries = config.get("countries") or collect_countries(transcripts)
    rounds = config.get("max_rounds") or len(transcripts)
    issues_per_turn = config.get("issues_per_turn", "—")

    st.title("NPT Agent Simulation Viewer")
    st.caption("Meeting-minutes style presentation: statements are organized by round and delegation, with supporting evidence expandable on demand.")


    
    c1, c2, c3 = st.columns(3)
    # c1.metric("Scenario", name)
    c1.metric("Session", f"{session_year} {meeting_type}")

    #c3.metric("Location", location)
    c2.metric("Countries", ", ".join(countries) if countries else "—")
    c3.metric("Rounds", str(rounds))


    with st.expander("Show configuration details"):
        st.json({"scenario": scenario, "config": config}, expanded=False)

def render_statement_card(rec: Dict[str, Any]) -> None:
    """
    Display a clean "statement card":
    - Issues
    - Statement text
    - Evidence expander with citations
    - (Optional) show drafts in a separate expander (disabled by default)
    """
    selected_issues = rec.get("selected_issues", []) or []
    statement = rec.get("statement", "") or ""
    citations = rec.get("citations", []) or []

    # Header
    st.markdown("#### Statement")
    if selected_issues:
        st.markdown("**Issues addressed:** " + " | ".join(selected_issues))
    else:
        st.markdown("**Issues addressed:** —")

    # Body
    # We keep it readable: treat as plain text but preserve line breaks
    st.markdown(
        statement.replace("\n", "  \n"),
        unsafe_allow_html=False
    )

    # Evidence
    with st.expander("📚 Evidence & citations (click to expand)", expanded=False):
        if not citations:
            st.info("No citations were returned in this record.")
        else:
            for i, c in enumerate(citations, start=1):
                src = c.get("source", "—")
                page = c.get("page", "—")
                para = c.get("para", "—")
                st.markdown(f"{i}. **{src}**, p.{page} ¶{para}")

    # Optional: show drafts (technical)
    with st.expander("🧪 Technical: issue-expert drafts (optional)", expanded=False):
        drafts = rec.get("drafts", [])
        if not drafts:
            st.write("No drafts available.")
        else:
            st.json(drafts, expanded=False)

def render_round_issue_summary(transcripts: List[List[Dict[str, Any]]], round_num: int) -> None:
    """
    Lightweight summary: which issues appeared this round, and who covered them.
    """
    if round_num < 1 or round_num > len(transcripts):
        return
    issue_map: Dict[str, List[str]] = {}
    for rec in transcripts[round_num - 1]:
        c = rec.get("country", "—")
        for issue in rec.get("selected_issues", []) or []:
            issue_map.setdefault(issue, []).append(c)

    if not issue_map:
        st.write("No issues found in this round.")
        return

    st.markdown("### Round snapshot")
    st.caption("Distribution of issues and delegations covered in the current round (automatically aggregated from JSON; no LLM involvement).")

    # Sort issues by how many countries addressed them
    items = sorted(issue_map.items(), key=lambda x: len(x[1]), reverse=True)
    for issue, countries in items:
        st.markdown(f"- **{issue}** — {', '.join(sorted(set(countries)))}")

# -----------------------------
# Sidebar controls (single source of truth)
# -----------------------------
st.sidebar.header("Data source")

mode = st.sidebar.radio(
    "Choose input source",
    ["Run simulation", "Upload a log JSON", "Load latest log from local folder"],
    index=0,
)

# Keep last result in session_state
if "simulation_data" not in st.session_state:
    st.session_state["simulation_data"] = None

data: Optional[Dict[str, Any]] = None

if mode == "Run simulation":
    st.sidebar.caption("This will call the LLM in real time and generate a new simulation result.")
    countries = st.sidebar.multiselect(
        "Countries",
        options=["USA", "China", "Russia"],
        default=["USA", "China", "Russia"]
    )
    max_rounds = st.sidebar.slider("Number of rounds", 1, 6, 3)
    issues_per_turn = st.sidebar.slider("Issues per turn", 1, 5, 3)


    evidence_mode = st.sidebar.radio(
    "Evidence mode",
    ["selected", "bucket_full"],
    index=0,
    help="selected = current retrieval-first mode; bucket_full = pass a larger slice of the full issue bucket directly to the LLM"
)
    
    full_bucket_max_entries = 8
    full_bucket_max_chars = 12000

    if evidence_mode == "bucket_full":
        full_bucket_max_entries = st.sidebar.slider(
            "Full bucket max entries", 1, 20, 8
        )
        full_bucket_max_chars = st.sidebar.slider(
            "Full bucket max chars", 2000, 30000, 12000, step=1000
        )

    round1_issues = st.sidebar.multiselect(
        "Round 1 issues (optional)",
        options=ISSUES_LIST,
        default=[]
    )

    if st.sidebar.button("▶ Run simulation"):
        with st.spinner("Running simulation (LLM calls in progress)..."):
            import agent_simulation_new  # your simulation module

            config = {
                "countries": countries,
                "max_rounds": max_rounds,
                "issues_per_turn": issues_per_turn,
                "round1_issues": round1_issues or None,
                "evidence_mode": evidence_mode,
                "full_bucket_max_entries": full_bucket_max_entries,
                "full_bucket_max_chars": full_bucket_max_chars,
            }

            result = agent_simulation_new.run_simulation(config)
            st.session_state["simulation_data"] = result
            st.success("Simulation completed.")
    # Use the latest simulation result if available
    data = st.session_state["simulation_data"]

elif mode == "Upload a log JSON":
    uploaded = st.sidebar.file_uploader("Upload npt_sim_*.json", type=["json"])
    if uploaded is not None:
        try:
            data = json.loads(uploaded.read().decode("utf-8"))
            st.session_state["simulation_data"] = None  # clear run result to avoid confusion
        except Exception as e:
            st.sidebar.error(f"Failed to parse JSON: {e}")

else:  # Load latest log
    log_dir = st.sidebar.text_input("Log folder", value="outputs/simulation_logs")
    latest = get_latest_log(log_dir)
    if latest:
        st.sidebar.success(f"Found latest: {os.path.basename(latest)}")
        data = load_json_file(latest)
        st.session_state["simulation_data"] = None
    else:
        st.sidebar.warning("No log file found. Run simulation first or upload a log JSON.")

# -----------------------------
# Main rendering
# -----------------------------
if not data:
    if mode == "Run simulation":
        st.info("Click ▶ Run simulation to generate a new result.")
    else:
        st.info("Please load a simulation log JSON (upload or load latest).")
    st.stop()


transcripts = normalize_transcripts(data)
if not transcripts:
    st.error("No transcripts found in this JSON. Please check your log file schema.")
    st.stop()

render_meta(data, transcripts)

# Round selector
max_rounds = len(transcripts)
round_num = st.radio("Select round", options=list(range(1, max_rounds + 1)), horizontal=True)

# Round snapshot summary
render_round_issue_summary(transcripts, round_num)

st.divider()

# Country tabs
countries = collect_countries(transcripts)
if not countries:
    st.error("No countries found in transcripts.")
    st.stop()

tab_labels = [f"{c}" for c in countries]
tabs = st.tabs(tab_labels)

for i, country in enumerate(countries):
    with tabs[i]:
        rec = find_record(transcripts, round_num, country)
        if not rec:
            st.warning(f"No record for {country} in round {round_num}.")
        else:
            # Show a clean card
            render_statement_card(rec)

# Download current JSON
st.divider()
st.download_button(
    "Download this simulation log JSON",
    data=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
    file_name=f"npt_sim_view_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
    mime="application/json",
)


