"""
prepare_data.py

Assemble the GPT-distilled and branch trajectories into the training-data pool.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Override via env: MEDACTION_BASE=/path/to/root, DATA_DIR=..., OUT_DIR=...
BASE = os.environ.get("MEDACTION_BASE",
                      os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
DATA = os.environ.get("DATA_DIR", os.path.join(BASE, "data"))
OUT  = os.environ.get("OUT_DIR",  os.path.join(BASE, "ablation", "data"))

# Source files
GPT_DISTILL = os.path.join(DATA, "gpt", "gpt_distilled.jsonl")
BC_BRANCH   = os.path.join(DATA, "bc", "bc_branch_full.jsonl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_jsonl(path):
    """Read JSONL file, return list of lines (strings)."""
    with open(path, encoding="utf-8") as f:
        return [line for line in f if line.strip()]


def write_jsonl(path, lines):
    """Write list of line strings to JSONL."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line if line.endswith("\n") else line + "\n")
    print(f"  Wrote {len(lines)} records → {os.path.basename(path)}")


def concat_jsonl(paths):
    """Concatenate multiple JSONL files into a list of lines."""
    lines = []
    for p in paths:
        lines.extend(read_jsonl(p))
    return lines


# ---------------------------------------------------------------------------
# Assemble the GPT + branch training pool
# ---------------------------------------------------------------------------

SOURCES = [GPT_DISTILL, BC_BRANCH]


def main():
    os.makedirs(OUT, exist_ok=True)

    out_path = os.path.join(OUT, "train_gpt_branch.jsonl")
    if os.path.exists(out_path):
        with open(out_path) as f:
            n = sum(1 for _ in f)
        print(f"  SKIP {os.path.basename(out_path)} ({n} records, exists)")
        return

    lines = concat_jsonl(SOURCES)
    write_jsonl(out_path, lines)


if __name__ == "__main__":
    main()

