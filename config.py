# config.py

# ============================ EXTRACT CONFIG ============================

# Directories for PDF ingestion
STATEMENTS_DIR = "statements"
WORKING_PAPERS_DIR = "working_papers"

# Database
DB_PATH = "positions.db"

# Safety switch:
# extract_paragraphs_to_db.py should NEVER drop tables unless explicitly changed
EXTRACT_DEBUG_MODE = False

# ============================ CLASSIFY CONFIG ============================

# Classify uses the SAME DB by default
DB_PATH = "positions.db"

# Model for classification
MODEL_CLASSIFY = "gpt-5.2"

# Retry/backoff (shared semantics with simulation)
MAX_RETRIES = 4
BASE_SLEEP = 2.0

# Batch controls
BATCH_SIZE = 50
MAX_ROWS = 500  # None means process until done

# Failure log path
FAIL_LOG = "classification_failures.jsonl"

# ============================ EXPORT CONFIG ============================
# Export behavior
INCLUDE_SUPPLEMENT = False              # whether to include 0.60-0.70 paragraphs
CORE_THRESHOLD = 0.70
SUPPLEMENT_LOWER = 0.60

# ============================ LITE BUCKETS CONFIG ============================
LITE_TOP_N = 10  # keep top N entries per issue

# ============================ EMBEDDINGS CONFIG ============================
EMB_DIR = "outputs/agent_buckets_emb"
EMBEDDING_MODEL = "text-embedding-3-large"



# ============================ SIMULATION CONFIG ============================

# --- Simulation metadata ---
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


# ============================ LOCAL OVERRIDES ============================
# Optional: create config_local.py to override any UPPERCASE variables.
# Example in config_local.py:
#   COUNTRIES = ["USA", "RUS"]
#   MAX_ROUNDS = 3
#   USE_LLM_ROUTER = False

try:
    import config_local  # type: ignore
    for k in dir(config_local):
        if k.isupper():
            globals()[k] = getattr(config_local, k)
except Exception:
    pass
