"""
gt_map.py — Build a {case_id: gt_diagnosis} lookup table from a results directory.
"""
import os
import json


def build_gt_map(results_base: str, priority_model: str = None) -> dict:
    all_folders = sorted([
        f for f in os.listdir(results_base)
        if os.path.isdir(os.path.join(results_base, f))
    ])
    if priority_model and priority_model in all_folders:
        all_folders = [priority_model] + [f for f in all_folders if f != priority_model]

    gt_map = {}
    print(f"Building GT index, scanning {len(all_folders)} folders...")

    for folder in all_folders:
        folder_path = os.path.join(results_base, folder)
        for fname in os.listdir(folder_path):
            if not fname.endswith(".json"):
                continue
            case_id = fname.replace(".json", "")
            if case_id in gt_map:
                continue
            try:
                with open(os.path.join(folder_path, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
                gc   = data.get("generate_case", {})
                diag = gc.get("diagnosis_results", "") or gc.get("final_diagnosis", "")
                if not diag:
                    rd   = data.get("checked_rare_disease", [])
                    diag = rd[0] if rd else ""
                if diag:
                    gt_map[case_id] = str(diag).strip()
            except Exception:
                continue

    print(f"GT index built: {len(gt_map)} case_ids have GT")
    return gt_map


def get_gt_diagnosis(case_data: dict) -> str:
    gc   = case_data.get("generate_case", {})
    diag = gc.get("diagnosis_results", "") or gc.get("final_diagnosis", "")
    if not diag:
        rd   = case_data.get("checked_rare_disease", [])
        diag = rd[0] if rd else ""
    return str(diag).strip()