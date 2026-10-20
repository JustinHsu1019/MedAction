"""
filter.py — two-step SFT-data filter (DTC truncation + RAC filtering).

Step 1 — DTC tail truncation:
  Scan from the last turn backward; find the first turn whose dtc_hop <= turn 1's.
  Keep that turn and everything before it; drop everything after.
  If no later turn matches or beats turn 1, drop the entire case.

Step 2 — per-turn RAC filter:
  For the turns kept by step 1, check each turn's RAC hop (delta DDx vs previous
  turn's ordered tests). If a turn has rac_hop > RAC_MAX_HOP, remove the *previous*
  turn (the one that did the test ordering). Turn 1 has no RAC and is skipped.
  No iteration / no re-compute.

Final check:
  If fewer than MIN_KEEP_TURNS remain after both steps, drop the case.

Usage:
  python filter.py                          # full run (includes encoding)
  python filter.py --refilter               # re-apply thresholds only (needs a prior full run)
  python filter.py --rac-max-hop 5          # override config's RAC_MAX_HOP
  python filter.py --min-keep 3             # override config's MIN_KEEP_TURNS

Output (under sft_output/{input_stem}/):
  filter_accepted.jsonl  — {"case_id": "...", "keep_turns": [1, 2, 4], "total_turns": 6}
  filter_rejected.jsonl  — {"case_id": "...", "reason": "..."}
  case_turn_stats.jsonl  — per-turn metrics (used by --refilter and analysis)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import config
from encoder import MedCPTEncoder, load_nodes_with_faiss
from graph import robust_load_graph
from metrics import (
    compute_dtc_cached,
    compute_delta_ddx,
    compute_rac_cached,
    parse_ddx_list,
    resolve_cached,
    safe_hop,
)


TURN_STATS_PATH = os.path.join(config.SFT_OUTPUT_DIR, "case_turn_stats.jsonl")


# ══════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════

def load_cases():
    df = pd.read_excel(config.INFERENCE_XLSX, sheet_name="Turn Details")
    sheet_names = pd.ExcelFile(config.INFERENCE_XLSX).sheet_names
    gt_sheet = "Case Results" if "Case Results" in sheet_names else "Case Summary"
    df_gt = pd.read_excel(config.INFERENCE_XLSX, sheet_name=gt_sheet)
    gt_map = dict(zip(df_gt["case_id"], df_gt["gt_diagnosis"]))

    cases = []
    for case_id, grp in df.groupby("case_id", sort=False):
        grp = grp.sort_values("turn")
        final_diagnosis = gt_map.get(case_id, "")

        turns = []
        for _, row in grp.iterrows():
            tests = []
            if pd.notna(row["tests_parsed"]) and str(row["tests_parsed"]).strip():
                tests = [t.strip() for t in str(row["tests_parsed"]).split(",")
                         if t.strip()]

            raw_resp = str(row["raw_response"]) if pd.notna(row.get("raw_response")) else ""

            turns.append({
                "turn":          int(row["turn"]),
                "curr_diag":     str(row["top1_ddx"]) if pd.notna(row["top1_ddx"]) else "",
                "tests_ordered": tests,
                "raw_response":  raw_resp,
            })

        cases.append({
            "case_id":         case_id,
            "final_diagnosis": final_diagnosis,
            "turns":           turns,
        })

    print(f"total {len(cases)} cases")
    return cases


# ══════════════════════════════════════════════════════════════
#  Batch encode
# ══════════════════════════════════════════════════════════════

def batch_encode_all(cases, encoder):
    all_texts = set()
    for case in cases:
        all_texts.add(case["final_diagnosis"])
        for turn in case["turns"]:
            curr_diag = turn.get("curr_diag", "")
            if curr_diag:
                all_texts.add(curr_diag)
            for act in turn.get("tests_ordered", []):
                all_texts.add(str(act))
            # disease names inside the DDx list also need encoding
            ddx_list = parse_ddx_list(turn.get("raw_response", ""))
            if ddx_list:
                for d in ddx_list:
                    all_texts.add(d)

    all_texts = [t for t in all_texts if t and t.strip()]
    print(f"{len(all_texts)} unique texts to encode")

    all_vecs = encoder.encode(all_texts, batch_size=256)
    text2vec = {t: all_vecs[i] for i, t in enumerate(all_texts)}
    print(f"batch encode done, {len(text2vec)} entries")
    return text2vec


# ══════════════════════════════════════════════════════════════
#  Per-turn metric computation (DTC + RAC)
# ══════════════════════════════════════════════════════════════

def compute_all_turn_stats(cases, text2vec,
                           cg_nodes, cg_index, cg_id_col, cg_name_col, CG_NET,
                           pk_nodes, pk_index, pk_id_col, pk_name_col, PK_NET):
    """
    Returns all_turn_stats, a dict mapping case_id to a list of per-turn dicts:
      {
        "turn": 1,
        "curr_diag": "...",
        "dtc_hop": int or None,
        "rac_hop": float or None,   # None for turn 1 or parse failure
        "n_delta": int,
        "n_valid": int,
      }
    """
    all_turn_stats = {}

    for i, case in enumerate(cases):
        case_id  = case["case_id"]
        final_dx = case["final_diagnosis"]
        turns    = case["turns"]

        if i % 200 == 0:
            print(f"  [{i}/{len(cases)}] computing turn stats...")

        per_turn = []
        prev_ddx_list = None
        prev_actions  = None

        for t_idx, turn in enumerate(turns):
            curr_diag = turn.get("curr_diag", "") or None
            actions   = turn.get("tests_ordered", [])
            raw_resp  = turn.get("raw_response", "")

            # ── DTC ──────────────────────────────────────────
            dtc_hop = None
            if curr_diag:
                dtc = compute_dtc_cached(
                    [curr_diag], final_dx,
                    pk_nodes, pk_index, pk_id_col, pk_name_col,
                    PK_NET, text2vec,
                )
                dtc_hop = dtc["dtc_hop"]

            # ── Parse DDx list ───────────────────────────────
            curr_ddx_list = parse_ddx_list(raw_resp)

            # ── RAC: delta DDx vs previous turn's actions ──
            rac_hop  = None
            n_delta  = 0
            n_valid  = 0

            if t_idx > 0 and prev_ddx_list is not None and curr_ddx_list is not None and prev_actions:
                delta = compute_delta_ddx(
                    prev_ddx_list, curr_ddx_list,
                    cg_nodes, cg_index, cg_id_col, cg_name_col, text2vec,
                )
                if delta:
                    rac = compute_rac_cached(
                        delta, prev_actions,
                        cg_nodes, cg_index, cg_id_col, cg_name_col,
                        CG_NET, text2vec,
                    )
                    rac_hop = rac["rac_hop"]
                    n_delta = rac["n_delta"]
                    n_valid = rac["n_valid"]

            per_turn.append({
                "turn":      turn["turn"],
                "curr_diag": curr_diag,
                "dtc_hop":   dtc_hop,
                "rac_hop":   rac_hop,
                "n_delta":   n_delta,
                "n_valid":   n_valid,
            })

            prev_ddx_list = curr_ddx_list
            prev_actions  = actions

        all_turn_stats[case_id] = per_turn

    return all_turn_stats


# ══════════════════════════════════════════════════════════════
#  Step 1: DTC tail truncation
# ══════════════════════════════════════════════════════════════

def step1_dtc_truncate(per_turn):
    """
    Scan from the last turn backward; find the first turn with dtc_hop <= turn 1's.
    Keep that turn and everything before it; drop everything after.

    Returns:
      keep_turn_numbers: list of int (turn numbers kept), or None (drop entire case)
      reason: str or None (reason for drop)
    """
    if not per_turn:
        return None, "no_turns"

    # Find turn 1's dtc_hop
    dtc_hop_1 = per_turn[0].get("dtc_hop")
    if dtc_hop_1 is None:
        # Turn 1 has no valid dtc, so there is no baseline to compare against; drop the case
        return None, "turn1_no_dtc"

    # Only 1 turn, so we cannot judge convergence; drop
    if len(per_turn) == 1:
        return None, "only_1_turn"

    # Scan backward from the last turn (skip index 0 — comparing to self is meaningless)
    n = len(per_turn)
    for i in range(n - 1, 0, -1):
        h = per_turn[i].get("dtc_hop")
        if h is not None and h <= dtc_hop_1:
            # Found it: keep indices 0..i (inclusive)
            keep = [t["turn"] for t in per_turn[:i + 1]]
            return keep, None

    # No subsequent turn reaches or beats turn 1
    return None, f"no_convergence (all dtc_hop > turn1 dtc_hop={dtc_hop_1})"


# ══════════════════════════════════════════════════════════════
#  Step 2: per-turn RAC filter
# ══════════════════════════════════════════════════════════════

def step2_rac_prune(per_turn, keep_turns, rac_max_hop):
    """
    For each turn in keep_turns, check rac_hop.
    If rac_hop > rac_max_hop, remove the *previous* turn (the test-ordering turn).

    per_turn:      original full per-turn list
    keep_turns:    turn numbers kept by step 1
    rac_max_hop:   threshold

    Returns: (pruned keep_turns list, set of removed turn numbers)
    """
    keep_set = set(keep_turns)

    # Map turn_number to its per_turn entry
    turn_map = {t["turn"]: t for t in per_turn}

    # Iterate in turn order
    sorted_keeps = sorted(keep_turns)
    turns_to_remove = set()

    for idx, turn_num in enumerate(sorted_keeps):
        t = turn_map.get(turn_num)
        if t is None:
            continue

        rac_hop = t.get("rac_hop")
        if rac_hop is None:
            continue  # turn 1 or parse failure; skip

        if rac_hop > rac_max_hop:
            # Remove the previous turn (the one that ordered the tests)
            if idx > 0:
                prev_turn_num = sorted_keeps[idx - 1]
                turns_to_remove.add(prev_turn_num)

    pruned = [t for t in keep_turns if t not in turns_to_remove]
    return pruned, turns_to_remove


# ══════════════════════════════════════════════════════════════
#  Main filter pipeline
# ══════════════════════════════════════════════════════════════

def run_filter(cases, all_turn_stats, rac_max_hop, min_keep_turns):
    """
    Two-step filter.

    Returns:
      accepted: list of {"case_id", "keep_turns": [int], "total_turns": int, ...}
      rejected: list of {"case_id", "reason", "total_turns": int}
    """
    accepted = []
    rejected = []

    for case in cases:
        case_id     = case["case_id"]
        total_turns = len(case["turns"])
        per_turn    = all_turn_stats.get(case_id, [])

        # ── Step 1: DTC tail truncation ──────────────────────
        keep_turns, reason = step1_dtc_truncate(per_turn)

        if keep_turns is None:
            rejected.append({
                "case_id":     case_id,
                "reason":      f"dtc: {reason}",
                "total_turns": total_turns,
            })
            continue

        dtc_truncated = len(keep_turns) < total_turns

        # ── Step 2: per-turn RAC filter ──────────────────────
        pruned_turns, removed = step2_rac_prune(per_turn, keep_turns, rac_max_hop)

        # ── Final check: enough turns remain? ────────────────
        if len(pruned_turns) < min_keep_turns:
            parts = []
            if not pruned_turns:
                parts.append("all turns pruned")
            else:
                parts.append(f"only {len(pruned_turns)} turn(s) remaining")
            if dtc_truncated:
                parts.append(f"dtc truncated {total_turns}->{len(keep_turns)}")
            if removed:
                parts.append(f"rac removed turns {sorted(removed)}")
            rejected.append({
                "case_id":     case_id,
                "reason":      f"too_few_turns: {', '.join(parts)}",
                "total_turns": total_turns,
            })
            continue

        entry = {
            "case_id":     case_id,
            "keep_turns":  pruned_turns,
            "total_turns": total_turns,
        }
        if removed:
            entry["rac_removed_turns"] = sorted(removed)
        if dtc_truncated:
            entry["dtc_truncated_from"] = max(keep_turns) + 1

        accepted.append(entry)

    return accepted, rejected


# ══════════════════════════════════════════════════════════════
#  Save
# ══════════════════════════════════════════════════════════════

def save_turn_stats(all_turn_stats):
    os.makedirs(config.SFT_OUTPUT_DIR, exist_ok=True)
    with open(TURN_STATS_PATH, "w") as f:
        for case_id, turns in all_turn_stats.items():
            f.write(json.dumps({"case_id": case_id, "turns": turns}, ensure_ascii=False) + "\n")
    print(f"Per-turn metrics saved to: {TURN_STATS_PATH}")


def load_turn_stats():
    turn_stats = {}
    with open(TURN_STATS_PATH) as f:
        for line in f:
            obj = json.loads(line.strip())
            turn_stats[obj["case_id"]] = obj["turns"]
    print(f"Loaded {len(turn_stats)} cases from {TURN_STATS_PATH}")
    return turn_stats


def save_results(accepted, rejected, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    path_accepted = os.path.join(output_dir, "filter_accepted.jsonl")
    path_rejected = os.path.join(output_dir, "filter_rejected.jsonl")

    with open(path_accepted, "w") as f:
        for item in accepted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(path_rejected, "w") as f:
        for item in rejected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    total = len(accepted) + len(rejected)

    # ── Classification stats ──────────────────────────────────
    # accepted breakdown
    n_full       = sum(1 for a in accepted
                       if len(a["keep_turns"]) == a["total_turns"])
    n_dtc_only   = sum(1 for a in accepted
                       if len(a["keep_turns"]) < a["total_turns"]
                       and "rac_removed_turns" not in a)
    n_rac_only   = sum(1 for a in accepted
                       if len(a["keep_turns"]) < a["total_turns"]
                       and "rac_removed_turns" in a
                       and "dtc_truncated_from" not in a)
    n_both       = sum(1 for a in accepted
                       if "rac_removed_turns" in a
                       and "dtc_truncated_from" in a)

    # rejected breakdown
    from collections import Counter
    reject_reasons = Counter()
    for item in rejected:
        r = item.get("reason", "unknown")
        if r.startswith("dtc:"):
            reject_reasons["dtc"] += 1
        elif r.startswith("too_few_turns:"):
            reject_reasons["too_few_turns"] += 1
        else:
            reject_reasons["other"] += 1

    # ── Report ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Filter results")
    print(f"{'='*60}")

    print(f"\n  Input: {total} cases")

    print(f"\n  ── Step 1: DTC tail truncation ──")
    n_dtc_rejected = reject_reasons.get("dtc", 0)
    n_dtc_passed   = total - n_dtc_rejected
    n_dtc_trimmed_accepted = sum(1 for a in accepted if "dtc_truncated_from" in a)
    print(f"    passed:    {n_dtc_passed:>5}  ({n_dtc_passed/total:.1%})")
    print(f"      of which truncated: {n_dtc_trimmed_accepted}")
    print(f"    dropped:   {n_dtc_rejected:>5}  ({n_dtc_rejected/total:.1%})")

    print(f"\n  ── Step 2: per-turn RAC filter ──")
    n_rac_pruned = sum(1 for a in accepted if "rac_removed_turns" in a)
    n_rac_turns_removed = sum(len(a["rac_removed_turns"]) for a in accepted if "rac_removed_turns" in a)
    print(f"    cases with turns removed:  {n_rac_pruned}")
    print(f"    total turns removed:       {n_rac_turns_removed}")

    n_too_few = reject_reasons.get("too_few_turns", 0)
    if n_too_few > 0:
        print(f"    dropped due to too few turns:  {n_too_few}")

    print(f"\n  ── Final ──")
    print(f"    full keep:                {n_full:>5}  ({n_full/total:.1%})  (no truncate, no turns removed)")
    print(f"    DTC truncated only:       {n_dtc_only:>5}  ({n_dtc_only/total:.1%})")
    print(f"    RAC pruned only:          {n_rac_only:>5}  ({n_rac_only/total:.1%})")
    print(f"    DTC + RAC both:           {n_both:>5}  ({n_both/total:.1%})")
    print(f"    dropped:                  {len(rejected):>5}  ({len(rejected)/total:.1%})")
    print(f"    ────────────────────────")
    print(f"    total usable:             {len(accepted):>5}  ({len(accepted)/total:.1%})")
    print(f"{'='*60}")

    # partial-keep stats
    partial = [a for a in accepted if len(a["keep_turns"]) < a["total_turns"]]
    if partial:
        orig = [a["total_turns"] for a in partial]
        kept = [len(a["keep_turns"]) for a in partial]
        print(f"\n  Partial-keep stats ({len(partial)} cases):")
        print(f"    original turns: mean={np.mean(orig):.1f}, min={min(orig)}, max={max(orig)}")
        print(f"    kept turns:     mean={np.mean(kept):.1f}, min={min(kept)}, max={max(kept)}")

    # rejection breakdown
    if rejected:
        print(f"\n  Rejection breakdown ({len(rejected)} cases):")
        for reason, count in reject_reasons.most_common():
            print(f"    {reason}: {count}")

    print(f"\n  Files:")
    print(f"    accepted -> {path_accepted}")
    print(f"    rejected -> {path_rejected}")


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-step SFT-data filter (DTC truncation + RAC filtering)")
    parser.add_argument("--refilter", action="store_true",
                        help="re-apply thresholds only, do not recompute metrics (requires a prior full filter run)")
    parser.add_argument("--rac-max-hop", type=float, default=config.RAC_MAX_HOP,
                        help=f"RAC threshold (default {config.RAC_MAX_HOP})")
    parser.add_argument("--min-keep", type=int, default=config.MIN_KEEP_TURNS,
                        help=f"minimum turns to keep (default {config.MIN_KEEP_TURNS})")
    parser.add_argument("--output-dir", type=str, default=config.SFT_OUTPUT_DIR,
                        help="output directory")
    args = parser.parse_args()

    print(f"Input:  {config.INFERENCE_XLSX}")
    print(f"Output: {args.output_dir}")
    print(f"RAC_MAX_HOP    = {args.rac_max_hop}")
    print(f"MIN_KEEP_TURNS = {args.min_keep}")

    cases = load_cases()

    # ── Per-turn metrics ─────────────────────────────────────
    if args.refilter:
        all_turn_stats = load_turn_stats()
    else:
        encoder = MedCPTEncoder()
        cg_nodes, cg_index, cg_id_col, cg_name_col = load_nodes_with_faiss(
            config.CLINGRAPH_NODES, encoder)
        pk_nodes, pk_index, pk_id_col, pk_name_col = load_nodes_with_faiss(
            config.PRIME_NODES, encoder)
        CG_NET = robust_load_graph(config.CLINGRAPH_EDGES)
        PK_NET = robust_load_graph(config.PRIME_EDGES)

        text2vec = batch_encode_all(cases, encoder)

        all_turn_stats = compute_all_turn_stats(
            cases, text2vec,
            cg_nodes, cg_index, cg_id_col, cg_name_col, CG_NET,
            pk_nodes, pk_index, pk_id_col, pk_name_col, PK_NET,
        )

        save_turn_stats(all_turn_stats)

    # ── Run filter ───────────────────────────────────────────
    accepted, rejected = run_filter(cases, all_turn_stats, args.rac_max_hop, args.min_keep)

    # ── Save ─────────────────────────────────────────────────
    save_results(accepted, rejected, args.output_dir)
