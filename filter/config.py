"""
config.py — all paths and global settings.
"""
import os

# ── Data root (override via env: DATA_ROOT=/path/to/data) ─
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data")))

# ── PrimeKG knowledge-graph files (override via env: PRIMEKG_DIR=...) ─
PRIMEKG_DIR     = os.environ.get("PRIMEKG_DIR", os.path.join(DATA_ROOT, "primekg"))
CLINGRAPH_NODES = os.path.join(PRIMEKG_DIR, "ClinGraph_nodes.csv")
CLINGRAPH_EDGES = os.path.join(PRIMEKG_DIR, "ClinGraph_edges.csv")
PRIME_NODES     = os.path.join(PRIMEKG_DIR, "primekg_nodes_full.csv")
PRIME_EDGES     = os.path.join(PRIMEKG_DIR, "kg.csv")

# ── OpenAI ───────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GPT_MODEL      = "gpt-4o-mini"

# ── Filter input (override via env: INFERENCE_XLSX=...) ──────
SFT_DIR        = os.path.join(DATA_ROOT, "sft_data")
INFERENCE_XLSX = os.environ.get("INFERENCE_XLSX",
                                os.path.join(DATA_ROOT, "inference.xlsx"))

# ── Filter output (auto-creates a subdir named after the input file) ─
# Override via env: SFT_OUTPUT_BASE=/path/to/filter_output
SFT_OUTPUT_BASE = os.environ.get("SFT_OUTPUT_BASE",
                                 os.path.join(os.path.dirname(DATA_ROOT), "filter_output"))
_INPUT_STEM     = os.path.splitext(os.path.basename(INFERENCE_XLSX))[0]
SFT_OUTPUT_DIR  = os.environ.get("SFT_OUTPUT_DIR",
                                 os.path.join(SFT_OUTPUT_BASE, _INPUT_STEM))
os.makedirs(SFT_OUTPUT_DIR, exist_ok=True)

# ── Filter parameters ────────────────────────────────────────
# Step 2 (RAC filter): if a turn's mean hop > RAC_MAX_HOP, truncate the previous turn
RAC_MAX_HOP = 3

# Cases whose remaining turns < MIN_KEEP_TURNS after truncation are dropped entirely
MIN_KEEP_TURNS = 2
