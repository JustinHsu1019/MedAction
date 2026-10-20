"""
analysis.py — Inference-result analysis (DTC / RAC two metrics).

Pure analysis, no filtering. Reads one inference xlsx and runs both metrics.

Metrics:
  DTC — Disease Trajectory Consistency (current diagnosis ↔ GT, PrimeKG hop)
  RAC — Reasoning-Action Consistency (previous tests ↔ DDx change, ClinGraph hop)

All metrics use raw hops; smaller is better.

Output:
  {input_stem}_analysis/
    ├── analysis_report.xlsx    — multi-sheet Excel (raw + metrics + summary)
    ├── fig_dtc_by_turn.png     — DTC hop across turns
    ├── fig_rac_by_turn.png     — RAC hop across turns
    └── fig_distributions.png   — case-level histograms

Usage:
  python analysis.py                                    # uses config.INFERENCE_XLSX
  python analysis.py --input /path/to/inference.xlsx    # explicit input
  python analysis.py --input inference.xlsx --cpu       # force CPU
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from encoder import MedCPTEncoder, load_nodes_with_faiss
from graph import robust_load_graph
from metrics import (compute_dtc_cached, compute_delta_ddx,
                     compute_rac_cached, parse_ddx_list, resolve_cached)
from filter import step1_dtc_truncate, step2_rac_prune


# ══════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════

def load_cases(xlsx_path):
    df_turns = pd.read_excel(xlsx_path, sheet_name="Turn Details")

    # Handle different sheet names for ground truth
    sheet_names = pd.ExcelFile(xlsx_path).sheet_names
    gt_sheet = "Case Results" if "Case Results" in sheet_names else "Case Summary"
    df_cases = pd.read_excel(xlsx_path, sheet_name=gt_sheet)
    gt_map = dict(zip(df_cases["case_id"], df_cases["gt_diagnosis"]))

    # Handle different column names
    tests_col = "tests_parsed" if "tests_parsed" in df_turns.columns else "parsed_tests"
    diag_col = "top1_ddx" if "top1_ddx" in df_turns.columns else "conclusion"
    has_tests = tests_col in df_turns.columns
    has_raw = "raw_response" in df_turns.columns
    if not has_raw:
        print("[warn] raw_response column missing; RAC metric will be skipped")

    cases = []
    rac_skip_cases = set()

    for case_id, grp in df_turns.groupby("case_id", sort=False):
        grp = grp.sort_values("turn")
        final_diagnosis = gt_map.get(case_id, "")

        turns = []
        ddx_parse_ok = True

        for _, row in grp.iterrows():
            tests = []
            if has_tests and pd.notna(row.get(tests_col)) and str(row[tests_col]).strip():
                tests = [t.strip() for t in str(row[tests_col]).split(",")
                         if t.strip()]

            ddx_list = None
            if has_raw and pd.notna(row.get("raw_response")):
                ddx_list = parse_ddx_list(row["raw_response"])
                if ddx_list is None:
                    ddx_parse_ok = False

            turns.append({
                "turn":          int(row["turn"]),
                "curr_diag":     str(row[diag_col]) if pd.notna(row.get(diag_col)) else "",
                "tests_ordered": tests,
                "ddx_list":      ddx_list,
            })

        if not ddx_parse_ok:
            rac_skip_cases.add(case_id)

        cases.append({
            "case_id":         case_id,
            "final_diagnosis": final_diagnosis,
            "turns":           turns,
        })

    print(f"loaded {len(cases)} cases (from {os.path.basename(xlsx_path)})")
    if rac_skip_cases:
        print(f"[warn] DDx parse failed for {len(rac_skip_cases)} cases; RAC skipped for them")

    return cases, df_turns, df_cases, rac_skip_cases


# ══════════════════════════════════════════════════════════════
#  Batch encode (includes DDx list text)
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
            ddx_list = turn.get("ddx_list")
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
#  Compute all metrics (no filtering)
# ══════════════════════════════════════════════════════════════

def compute_all_metrics(cases, text2vec, rac_skip_cases,
                        cg_nodes, cg_index, cg_id_col, cg_name_col, CG_NET,
                        pk_nodes, pk_index, pk_id_col, pk_name_col, PK_NET):
    case_records = []
    turn_records = []

    for i, case in enumerate(cases):
        case_id  = case["case_id"]
        final_dx = case["final_diagnosis"]
        turns    = case["turns"]
        n_turns  = len(turns)
        do_rac   = case_id not in rac_skip_cases

        if i % 200 == 0:
            print(f"  [{i}/{len(cases)}] computing metrics...")

        rac_hops = []
        prev_actions  = None
        prev_ddx_list = None

        for t_idx, turn in enumerate(turns):
            actions   = turn.get("tests_ordered", [])
            curr_diag = turn.get("curr_diag", "") or None
            curr_ddx  = turn.get("ddx_list")

            # ── DTC ──────────────────────────────────────────
            dtc_hop = None
            if curr_diag:
                dtc = compute_dtc_cached([curr_diag], final_dx,
                                          pk_nodes, pk_index, pk_id_col, pk_name_col,
                                          PK_NET, text2vec)
                dtc_hop = dtc["dtc_hop"]

            # ── RAC (delta DDx vs previous turn's actions) ───
            rac_hop, rac_n_delta, rac_n_valid = None, 0, 0
            if do_rac and t_idx > 0 and prev_ddx_list and curr_ddx and prev_actions:
                delta = compute_delta_ddx(
                    prev_ddx_list, curr_ddx,
                    cg_nodes, cg_index, cg_id_col, cg_name_col, text2vec)
                if delta:
                    rac = compute_rac_cached(
                        delta, prev_actions,
                        cg_nodes, cg_index, cg_id_col, cg_name_col,
                        CG_NET, text2vec)
                    rac_hop     = rac["rac_hop"]
                    rac_n_delta = rac["n_delta"]
                    rac_n_valid = rac["n_valid"]
                    if rac_hop is not None:
                        rac_hops.append(rac_hop)

            turn_records.append({
                "case_id":      case_id,
                "turn":         t_idx + 1,
                "curr_diag":    curr_diag,
                "n_actions":    len(actions),
                "dtc_hop":      dtc_hop,
                "rac_hop":      round(rac_hop, 4) if rac_hop is not None else None,
                "rac_n_delta":  rac_n_delta,
                "rac_n_valid":  rac_n_valid,
            })

            prev_actions  = actions
            prev_ddx_list = curr_ddx

        # ── Case-level aggregates ────────────────────────────
        last_diag = turns[-1].get("curr_diag", "") or final_dx
        final_dtc = compute_dtc_cached([last_diag], final_dx,
                                        pk_nodes, pk_index, pk_id_col, pk_name_col,
                                        PK_NET, text2vec)

        avg_rac = float(np.mean(rac_hops)) if rac_hops else None

        case_records.append({
            "case_id":        case_id,
            "gt_diagnosis":   final_dx,
            "n_turns":        n_turns,
            "final_dtc_hop":  final_dtc["dtc_hop"],
            "gt_hit":         1 if final_dtc["dtc_hop"] is not None and final_dtc["dtc_hop"] == 0 else 0,
            "avg_rac_hop":    round(avg_rac, 4) if avg_rac is not None else None,
            "rac_available":  1 if do_rac else 0,
        })

    df_case = pd.DataFrame(case_records)
    df_turn = pd.DataFrame(turn_records)
    return df_case, df_turn


# ══════════════════════════════════════════════════════════════
#  Summary stats
# ══════════════════════════════════════════════════════════════

def build_summary(df_case, df_turn):
    rows = []

    # ── Case-level ────────────────────────────────────────────
    rows.append({"section": "Case-Level", "metric": "total_cases", "value": len(df_case)})
    rows.append({"section": "Case-Level", "metric": "gt_hit_count", "value": int(df_case["gt_hit"].sum())})
    rows.append({"section": "Case-Level", "metric": "gt_hit_rate", "value": round(df_case["gt_hit"].mean(), 4)})
    rows.append({"section": "Case-Level", "metric": "rac_available_cases",
                 "value": int(df_case["rac_available"].sum())})

    for col in ["final_dtc_hop", "avg_rac_hop", "n_turns"]:
        s = df_case[col].dropna()
        if len(s) == 0:
            continue
        rows.append({"section": "Case-Level", "metric": f"{col}_mean", "value": round(s.mean(), 4)})
        rows.append({"section": "Case-Level", "metric": f"{col}_std",  "value": round(s.std(), 4)})
        rows.append({"section": "Case-Level", "metric": f"{col}_median", "value": round(s.median(), 4)})
        rows.append({"section": "Case-Level", "metric": f"{col}_min",  "value": round(s.min(), 4)})
        rows.append({"section": "Case-Level", "metric": f"{col}_max",  "value": round(s.max(), 4)})

    # ── By-turn ───────────────────────────────────────────────
    for col in ["dtc_hop", "rac_hop"]:
        by_turn = df_turn.groupby("turn")[col].agg(["mean", "std", "count"])
        for turn_idx in sorted(by_turn.index):
            m, s, n = by_turn.loc[turn_idx]
            if pd.notna(m):
                rows.append({
                    "section": f"By-Turn ({col})",
                    "metric":  f"turn_{turn_idx}_mean",
                    "value":   round(float(m), 4),
                })
                rows.append({
                    "section": f"By-Turn ({col})",
                    "metric":  f"turn_{turn_idx}_std",
                    "value":   round(float(s), 4) if pd.notna(s) else None,
                })
                rows.append({
                    "section": f"By-Turn ({col})",
                    "metric":  f"turn_{turn_idx}_n",
                    "value":   int(n),
                })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
#  Filter stats (dry run — count only, don't actually filter)
# ══════════════════════════════════════════════════════════════

def compute_filter_stats(df_case, df_turn, rac_max_hop=3, min_keep_turns=2):
    """
    Simulate the two-step filter from filter.py on analysis results.
    Returns a dict of stats and a DataFrame of per-case filter results.
    """
    total_cases = len(df_case)
    total_turns = len(df_turn)

    dtc_rejected = 0
    dtc_truncated_cases = 0
    dtc_truncated_turns = 0
    rac_pruned_cases = 0
    rac_pruned_turns = 0
    final_rejected = 0
    fully_kept = 0

    case_filter_results = []

    for case_id, grp in df_turn.groupby("case_id", sort=False):
        grp = grp.sort_values("turn")
        per_turn = []
        for _, row in grp.iterrows():
            per_turn.append({
                "turn":     int(row["turn"]),
                "curr_diag": row.get("curr_diag", ""),
                "dtc_hop":  row["dtc_hop"] if pd.notna(row.get("dtc_hop")) else None,
                "rac_hop":  row["rac_hop"] if pd.notna(row.get("rac_hop")) else None,
            })

        n_total = len(per_turn)

        # Step 1: DTC truncation
        keep_turns, reason = step1_dtc_truncate(per_turn)
        if keep_turns is None:
            dtc_rejected += 1
            final_rejected += 1
            case_filter_results.append({
                "case_id": case_id, "total_turns": n_total,
                "status": "rejected", "reason": f"dtc: {reason}",
                "kept_turns": 0,
            })
            continue

        dtc_trimmed = n_total - len(keep_turns)
        if dtc_trimmed > 0:
            dtc_truncated_cases += 1
            dtc_truncated_turns += dtc_trimmed

        # Step 2: RAC pruning
        pruned_turns, removed = step2_rac_prune(per_turn, keep_turns, rac_max_hop)
        if removed:
            rac_pruned_cases += 1
            rac_pruned_turns += len(removed)

        # Final check
        if len(pruned_turns) < min_keep_turns:
            final_rejected += 1
            case_filter_results.append({
                "case_id": case_id, "total_turns": n_total,
                "status": "rejected", "reason": "too_few_turns",
                "kept_turns": len(pruned_turns),
            })
        else:
            if len(pruned_turns) == n_total:
                fully_kept += 1
            case_filter_results.append({
                "case_id": case_id, "total_turns": n_total,
                "status": "accepted",
                "reason": "",
                "kept_turns": len(pruned_turns),
            })

    accepted = total_cases - final_rejected
    accepted_turns = sum(r["kept_turns"] for r in case_filter_results)
    stats = {
        "total_cases":          total_cases,
        "total_turns":          total_turns,
        "dtc_rejected_cases":   dtc_rejected,
        "dtc_rejected_pct":     round(dtc_rejected / total_cases * 100, 1) if total_cases else 0,
        "dtc_truncated_cases":  dtc_truncated_cases,
        "dtc_truncated_turns":  dtc_truncated_turns,
        "rac_pruned_cases":     rac_pruned_cases,
        "rac_pruned_turns":     rac_pruned_turns,
        "final_rejected":       final_rejected,
        "final_rejected_pct":   round(final_rejected / total_cases * 100, 1) if total_cases else 0,
        "accepted":             accepted,
        "accepted_pct":         round(accepted / total_cases * 100, 1) if total_cases else 0,
        "fully_kept":           fully_kept,
        "fully_kept_pct":       round(fully_kept / total_cases * 100, 1) if total_cases else 0,
        "accepted_turns":       accepted_turns,
        "accepted_turns_pct":   round(accepted_turns / total_turns * 100, 1) if total_turns else 0,
    }
    return stats, pd.DataFrame(case_filter_results)


def print_filter_stats(stats):
    total = stats["total_cases"]
    print(f"\n{'─'*60}")
    print(f"  Filter Simulation (dry-run, RAC_MAX_HOP=3, MIN_KEEP=2)")
    print(f"{'─'*60}")
    print(f"  Total cases: {total}")
    print(f"  DTC rejected:  {stats['dtc_rejected_cases']:>5}  ({stats['dtc_rejected_pct']}%)")
    print(f"  DTC truncated (partial):  {stats['dtc_truncated_cases']:>5}  ({stats['dtc_truncated_turns']} turns removed)")
    print(f"  RAC pruned: {stats['rac_pruned_cases']:>5}  ({stats['rac_pruned_turns']} turns removed)")
    print(f"  Final rejected: {stats['final_rejected']:>5}  ({stats['final_rejected_pct']}%)")
    print(f"  Accepted: {stats['accepted']:>5}  ({stats['accepted_pct']}%)")
    print(f"  Fully kept (no changes): {stats['fully_kept']:>5}  ({stats['fully_kept_pct']}%)")


# ══════════════════════════════════════════════════════════════
#  Plots
# ══════════════════════════════════════════════════════════════

def _turn_agg(df_turn, col):
    agg = df_turn.groupby("turn")[col].agg(["mean", "std", "count"])
    agg = agg[agg["count"] >= 5]
    return agg


def _plot_single_turn(df_turn, col, ylabel, title, color, marker, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    agg = _turn_agg(df_turn, col)
    if len(agg) == 0:
        ax.set_title(f"{title} (no data)")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    ax.errorbar(agg.index, agg["mean"], yerr=agg["std"], marker=marker, capsize=4,
                linewidth=2, color=color)
    ax.set_xlabel("Turn")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(agg.index)
    ax.grid(True, alpha=0.3)
    for i, row in agg.iterrows():
        ax.annotate(f'n={int(row["count"])}', (i, row["mean"]),
                    textcoords="offset points", xytext=(0, 12), fontsize=8, ha="center", color="gray")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {os.path.basename(out_path)}")


def plot_dtc_by_turn(df_turn, out_path, filter_stats=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    agg = df_turn.groupby("turn")["dtc_hop"].agg(["mean", "std", "count"])
    agg = agg[agg["count"] >= 1]
    if len(agg) == 0:
        ax.set_title("DTC Hop by Turn (no data)")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    ax.plot(agg.index, agg["mean"], marker="o", linewidth=2, color="tab:green")
    ax.set_xlabel("Turn")
    ax.set_ylabel("DTC Hop (distance to GT)")
    ax.set_title("DTC Hop by Turn (mean ± std)")
    ax.set_xticks(agg.index)
    ax.grid(True, alpha=0.3)
    for i, row in agg.iterrows():
        ax.annotate(f'n={int(row["count"])}', (i, row["mean"]),
                    textcoords="offset points", xytext=(0, 12), fontsize=8, ha="center", color="gray")

    if filter_stats:
        info_lines = [
            f"Total cases: {filter_stats['total_cases']}",
            f"DTC rejected: {filter_stats['dtc_rejected_cases']} ({filter_stats['dtc_rejected_pct']}%)",
            f"DTC truncated: {filter_stats['dtc_truncated_cases']} cases, {filter_stats['dtc_truncated_turns']} turns",
            f"RAC pruned: {filter_stats['rac_pruned_cases']} cases, {filter_stats['rac_pruned_turns']} turns",
            f"Final accepted: {filter_stats['accepted']} ({filter_stats['accepted_pct']}%)",
        ]
        info_text = "\n".join(info_lines)
        ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {os.path.basename(out_path)}")


def plot_rac_by_turn(df_turn, out_path):
    _plot_single_turn(df_turn, "rac_hop", "RAC Hop (action to DDx delta distance)",
                      "RAC Hop by Turn (mean ± std)", "tab:blue", "s", out_path)


def plot_distributions(df_case, out_path):
    metrics = [
        ("final_dtc_hop", "Final DTC Hop",  "tab:green"),
        ("avg_rac_hop",   "Avg RAC Hop",    "tab:blue"),
        ("n_turns",       "Number of Turns", "tab:gray"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, (col, title, color) in enumerate(metrics):
        ax = axes[idx]
        data = df_case[col].dropna()
        if len(data) == 0:
            ax.set_title(f"{title} (no data)")
            continue
        ax.hist(data, bins=30, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(data.mean(), color="black", linestyle="--", linewidth=1.5,
                   label=f"mean={data.mean():.3f}")
        ax.axvline(data.median(), color="gray", linestyle=":", linewidth=1.5,
                   label=f"median={data.median():.3f}")
        ax.set_title(title)
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Case-Level Metric Distributions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {os.path.basename(out_path)}")


# ══════════════════════════════════════════════════════════════
#  Excel output
# ══════════════════════════════════════════════════════════════

def save_xlsx(df_case, df_turn, df_summary, df_turns_raw, df_cases_raw, out_path):
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_case.to_excel(writer, sheet_name="Case Metrics", index=False)
        df_turn.to_excel(writer, sheet_name="Turn Metrics", index=False)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        df_turns_raw.to_excel(writer, sheet_name="Raw Turn Details", index=False)
        df_cases_raw.to_excel(writer, sheet_name="Raw Case Results", index=False)

    print(f"  saved: {os.path.basename(out_path)}")


# ══════════════════════════════════════════════════════════════
#  Console report
# ══════════════════════════════════════════════════════════════

def print_report(df_case, df_turn):
    print(f"\n{'═'*60}")
    print(f"  Analysis report")
    print(f"{'═'*60}")

    print(f"\n  Total cases:     {len(df_case)}")
    print(f"  GT hits:         {int(df_case['gt_hit'].sum())}  ({df_case['gt_hit'].mean():.1%})")
    rac_avail = int(df_case["rac_available"].sum())
    print(f"  RAC available:   {rac_avail}  ({rac_avail/len(df_case):.1%})")

    print(f"\n{'─'*60}")
    print(f"  Case-level metric distribution")
    print(f"{'─'*60}")
    cols = ["final_dtc_hop", "avg_rac_hop", "n_turns"]
    existing = [c for c in cols if c in df_case.columns]
    print(df_case[existing].describe().round(4))

    print(f"\n{'─'*60}")
    print(f"  By-turn mean metrics")
    print(f"{'─'*60}")
    for col in ["dtc_hop", "rac_hop"]:
        if col not in df_turn.columns:
            continue
        agg = df_turn.groupby("turn")[col].agg(["mean", "std", "count"])
        agg.columns = [f"{col}_mean", f"{col}_std", f"{col}_n"]
        agg = agg.round(4)
        valid = agg[agg[f"{col}_n"] > 0]
        if len(valid) > 0:
            print(f"\n  {col}:")
            print(valid.to_string())

    print(f"\n{'═'*60}\n")


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference-result two-metric analysis (DTC / RAC)")
    parser.add_argument("--input", type=str, default=config.INFERENCE_XLSX,
                        help="input xlsx path (default: config.INFERENCE_XLSX)")
    parser.add_argument("--cpu", action="store_true",
                        help="force CPU for encoding")
    parser.add_argument("--max-turns", type=int, default=None,
                        help="limit max turns shown in plots (e.g., 3, 4, 5)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="custom output directory (default: {input_stem}_analysis)")
    parser.add_argument("--dtc-only", action="store_true",
                        help="only produce the DTC plot; skip RAC and distributions")
    args = parser.parse_args()

    xlsx_path = args.input
    input_stem = os.path.splitext(os.path.basename(xlsx_path))[0]
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(xlsx_path), f"{input_stem}_analysis")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Input:  {xlsx_path}")
    print(f"Output: {output_dir}")

    # ── Load data ────────────────────────────────────────────
    cases, df_turns_raw, df_cases_raw, rac_skip_cases = load_cases(xlsx_path)

    # ── Load graphs & encoder ────────────────────────────────
    device = "cpu" if args.cpu else None
    encoder = MedCPTEncoder(device=device)

    cg_nodes, cg_index, cg_id_col, cg_name_col = load_nodes_with_faiss(
        config.CLINGRAPH_NODES, encoder)
    pk_nodes, pk_index, pk_id_col, pk_name_col = load_nodes_with_faiss(
        config.PRIME_NODES, encoder)
    CG_NET = robust_load_graph(config.CLINGRAPH_EDGES)
    PK_NET = robust_load_graph(config.PRIME_EDGES)

    text2vec = batch_encode_all(cases, encoder)

    # ── Compute metrics ──────────────────────────────────────
    print("\nComputing two metrics (DTC / RAC)...")
    df_case, df_turn = compute_all_metrics(
        cases, text2vec, rac_skip_cases,
        cg_nodes, cg_index, cg_id_col, cg_name_col, CG_NET,
        pk_nodes, pk_index, pk_id_col, pk_name_col, PK_NET)

    # ── Summary ──────────────────────────────────────────────
    df_summary = build_summary(df_case, df_turn)

    # ── Console report ───────────────────────────────────────
    print_report(df_case, df_turn)

    # ── Filter stats (dry-run) ───────────────────────────────
    filter_stats, df_filter = compute_filter_stats(df_case, df_turn)
    print_filter_stats(filter_stats)

    # ── Plots ────────────────────────────────────────────────
    print("Generating plots...")
    df_turn_plot = df_turn if args.max_turns is None else df_turn[df_turn["turn"] <= args.max_turns]
    prefix = input_stem
    suffix = f"_maxT{args.max_turns}" if args.max_turns else ""
    plot_dtc_by_turn(df_turn_plot, os.path.join(output_dir, f"{prefix}_dtc_by_turn{suffix}.png"), filter_stats=filter_stats)
    if not args.dtc_only:
        plot_rac_by_turn(df_turn_plot, os.path.join(output_dir, f"{prefix}_rac_by_turn{suffix}.png"))
        plot_distributions(df_case, os.path.join(output_dir, f"{prefix}_distributions{suffix}.png"))

    # ── Add filter stats to summary ──────────────────────────
    for k, v in filter_stats.items():
        df_summary = pd.concat([df_summary, pd.DataFrame([{
            "section": "Filter (dry-run)", "metric": k, "value": v,
        }])], ignore_index=True)

    # ── Excel output ─────────────────────────────────────────
    print("\nGenerating Excel report...")
    xlsx_out = os.path.join(output_dir, "analysis_report.xlsx")
    save_xlsx(df_case, df_turn, df_summary, df_turns_raw, df_cases_raw, xlsx_out)

    print(f"\nDone. All output in: {output_dir}")
