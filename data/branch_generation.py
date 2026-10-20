"""
branch_generation.py

Generate branch trajectories with Baichuan-M3: for each case, re-launch new
diagnostic paths from intermediate turns of the completed trunk trajectories.
"""

import json
import asyncio
import argparse
import time
import random
from datetime import datetime, timezone

import aiohttp
import pandas as pd
import nest_asyncio
from vllm import SamplingParams

nest_asyncio.apply()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MEDACTION_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
JSON_FOLDER = os.path.join(MEDACTION_DIR, "cases")
TRUNK_XLSX = os.path.join(MEDACTION_DIR, "trunk", "infer_result_baichuan.xlsx")
TEST_CASE_FILE = os.path.join(MEDACTION_DIR, "case_ids.txt")
DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "branch_data"))

# ---------------------------------------------------------------------------
# Import reusable functions from inference_baichuan.py (local copy)
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(__file__))
from inference_baichuan import (
    # Config
    OPENAI_KEY, OPENAI_URL, ORACLE_MODEL, ORACLE_CONCURRENCY,
    MAX_TURNS, RECENT_TURNS_WINDOW,
    # Prompts
    SYSTEM_PROMPT, INITIAL_PROMPT, FOLLOWUP_PROMPT,
    build_oracle_system_prompt, PARSE_TESTS_SYSTEM,
    # Model
    init_vllm, vllm_batch_generate,
    # API
    _openai_call, _oracle_call, parse_tests_from_response,
    # Parsing
    strip_think_blocks, parse_response,
    # Context
    _extract_unavailable_tests, build_turn_detail, build_global_status,
    # Data loading
    parse_case_json,
)


# ---------------------------------------------------------------------------
# Step 0: Case sampling and trunk data loading
# ---------------------------------------------------------------------------

def load_trunk_data(trunk_xlsx):
    """Load trunk Excel with Turn Details and Case Summary."""
    df_turns = pd.read_excel(trunk_xlsx, sheet_name="Turn Details")
    df_cases = pd.read_excel(trunk_xlsx, sheet_name="Case Summary")
    return df_turns, df_cases


def sample_cases(df_cases, num_cases, seed, json_folder, test_case_file):
    """Sample num_cases from trunk cases, excluding the test cases."""
    # Load test case IDs to exclude
    test_ids = set()
    if os.path.exists(test_case_file):
        with open(test_case_file) as f:
            test_ids = {line.strip() for line in f if line.strip()}
        print(f"  Excluding {len(test_ids)} test cases from {test_case_file}")

    all_ids = df_cases["case_id"].tolist()
    # Only keep cases that have a corresponding JSON file AND are not test cases
    valid_ids = [cid for cid in all_ids
                 if os.path.exists(os.path.join(json_folder, f"{cid}.json"))
                 and cid not in test_ids]
    dropped = len(all_ids) - len(valid_ids)
    if dropped:
        print(f"  Dropped {dropped} cases (no JSON or in test set)")
    print(f"  {len(valid_ids)} eligible cases for sampling")
    random.seed(seed)
    selected = random.sample(valid_ids, min(num_cases, len(valid_ids)))
    return selected


def save_selected_cases(selected_ids, df_turns, df_cases, data_dir):
    """Save selected case IDs and their trunk data."""
    os.makedirs(data_dir, exist_ok=True)

    # Save case IDs
    cases_json = os.path.join(data_dir, "selected_cases.json")
    with open(cases_json, "w") as f:
        json.dump(selected_ids, f, indent=2)
    print(f"Saved {len(selected_ids)} selected case IDs to {cases_json}")

    # Save filtered trunk Excel
    trunk_xlsx = os.path.join(data_dir, "selected_data.xlsx")
    sel_set = set(selected_ids)
    df_t = df_turns[df_turns["case_id"].isin(sel_set)]
    df_c = df_cases[df_cases["case_id"].isin(sel_set)]
    with pd.ExcelWriter(trunk_xlsx, engine="openpyxl") as writer:
        df_t.to_excel(writer, sheet_name="Turn Details", index=False)
        df_c.to_excel(writer, sheet_name="Case Summary", index=False)
    print(f"Saved trunk data for {len(df_c)} cases to {trunk_xlsx}")

    # Print turn distribution
    turns_per_case = df_c["turns_used"].tolist()
    print(f"Turn distribution: min={min(turns_per_case)}, max={max(turns_per_case)}, "
          f"mean={sum(turns_per_case)/len(turns_per_case):.1f}")
    from collections import Counter
    dist = Counter(turns_per_case)
    for t in sorted(dist):
        print(f"  {t} turns: {dist[t]} cases")


# ---------------------------------------------------------------------------
# Step 1: Trunk state reconstruction
# ---------------------------------------------------------------------------

async def reconstruct_trunk_state(case_id, branch_turn, df_turns, case_json,
                                   session, oracle_sem):
    """
    Reconstruct the exact state the model saw before `branch_turn`.

    For branch_turn == 1: return empty state (fresh start).
    For branch_turn >= 2: return state accumulated from trunk turns 1..(branch_turn-1).

    Uses GPT-4o parse_tests_from_response on each trunk turn's raw_response
    to get the exact same test list the original pipeline produced, avoiding
    fragile comma-splitting of the tests_parsed string.

    Also collects full trunk turn data so each branch output is a complete
    trajectory (inherited trunk turns + generated branch turns).

    Returns (state, inherited_turns).
    """
    state = {
        "case": case_json,
        "done": False,
        "recent_turns": [],
        "turn_logs": [],
        "done_tests": [],
        "unavailable_tests": [],
        "last_top1_ddx": "",
        "last_conclusion": "",
        "last_parse_valid": True,
        "current_oracle": "",
    }
    inherited_turns = []

    if branch_turn == 1:
        return state, inherited_turns

    # Get trunk turns for this case, sorted
    case_turns = df_turns[df_turns["case_id"] == case_id].sort_values("turn")

    for _, row in case_turns.iterrows():
        turn_num = int(row["turn"])
        if turn_num >= branch_turn:
            break

        raw_resp = str(row["raw_response"]) if pd.notna(row["raw_response"]) else ""
        oracle_text = str(row["oracle_response"]) if pd.notna(row["oracle_response"]) else ""
        clean_response = strip_think_blocks(raw_resp) if raw_resp.strip() else ""
        prompt_text = str(row["prompt"]) if pd.notna(row["prompt"]) else ""

        # Re-extract tests via GPT-4o — same call the original pipeline used,
        # avoids fragile comma-splitting of tests_parsed string
        tests_list = await parse_tests_from_response(
            session, oracle_sem, clean_response
        ) if clean_response.strip() else []

        # Accumulate done_tests
        done_lower = {t.lower().strip() for t in state["done_tests"]}
        for t in tests_list:
            if t.lower().strip() not in done_lower:
                state["done_tests"].append(t)
                done_lower.add(t.lower().strip())

        # Extract unavailable tests from oracle response
        if oracle_text.strip():
            new_unavail = _extract_unavailable_tests(oracle_text)
            unavail_lower = {t.lower().strip() for t in state["unavailable_tests"]}
            for t in new_unavail:
                if t.lower().strip() not in unavail_lower:
                    state["unavailable_tests"].append(t)
                    unavail_lower.add(t.lower().strip())

        # Rebuild turn detail for recent_turns sliding window
        parsed = parse_response(raw_resp) if raw_resp.strip() else {
            "ddx_list": "", "pivot": "", "conclusion": "",
            "diag_status": "", "top1_ddx": "",
        }
        turn_detail = build_turn_detail(turn_num, parsed, oracle_text, tests_list)
        state["recent_turns"].append(turn_detail)

        # Track last oracle response
        state["current_oracle"] = oracle_text

        # Track metadata
        if parsed.get("top1_ddx"):
            state["last_top1_ddx"] = parsed["top1_ddx"]
        if parsed.get("conclusion"):
            state["last_conclusion"] = parsed["conclusion"]

        # Store full trunk turn for output
        inherited_turns.append({
            "turn": turn_num,
            "prompt": prompt_text,
            "raw_response": clean_response,
            "top1_ddx": parsed.get("top1_ddx", ""),
            "conclusion": parsed.get("conclusion", ""),
            "diag_status": parsed.get("diag_status", ""),
            "model_done": parsed.get("model_done", False),
            "rule_done": parsed.get("rule_done", False),
            "tests_parsed": tests_list,
            "oracle_response": oracle_text,
            "is_done": parsed.get("is_done", False),
            "source": "trunk",
        })

    return state, inherited_turns


# ---------------------------------------------------------------------------
# Step 2: Branch runner (interactive multi-turn loop from branch point)
# ---------------------------------------------------------------------------

def _build_branch_messages(state, turn):
    """Build messages for a branch turn — same logic as original pipeline."""
    case = state["case"]
    if turn == 1:
        user_content = INITIAL_PROMPT.format(case=case["initial_obs"])
    else:
        done_tests_block, unavailable_tests_block = build_global_status(state)
        recent = state["recent_turns"][-RECENT_TURNS_WINDOW:]
        recent_turns_block = "\n\n".join(recent) if recent else "(first follow-up turn)"
        user_content = FOLLOWUP_PROMPT.format(
            case=case["initial_obs"],
            done_tests_block=done_tests_block,
            unavailable_tests_block=unavailable_tests_block,
            window_size=len(recent),
            recent_turns_block=recent_turns_block,
            oracle_results=state["current_oracle"],
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Step 3: Output helpers
# ---------------------------------------------------------------------------

def branch_output_path(data_dir, case_id, branch_turn, path_id):
    return os.path.join(
        data_dir, case_id, f"branch_t{branch_turn}_p{path_id}", "turn_logs.json"
    )


def branch_exists(data_dir, case_id, branch_turn, path_id):
    return os.path.exists(branch_output_path(data_dir, case_id, branch_turn, path_id))


def save_branch_result(data_dir, case_id, branch_turn, path_id, result):
    out_dir = os.path.join(data_dir, case_id, f"branch_t{branch_turn}_p{path_id}")
    os.makedirs(out_dir, exist_ok=True)

    result["path_id"] = path_id

    with open(os.path.join(out_dir, "turn_logs.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    summary = {
        "case_id": result["case_id"],
        "branch_turn": result["branch_turn"],
        "path_id": path_id,
        "total_branch_turns": result["total_branch_turns"],
        "final_conclusion": result["final_conclusion"],
        "final_top1_ddx": result["final_top1_ddx"],
        "trunk_context": result["trunk_context"],
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def save_trunk_summary(data_dir, case_id, df_turns, case_json):
    """Save trunk metadata for a case (once per case)."""
    out_dir = os.path.join(data_dir, case_id)
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "trunk_summary.json")
    if os.path.exists(summary_path):
        return

    case_turns = df_turns[df_turns["case_id"] == case_id].sort_values("turn")
    trunk = {
        "case_id": case_id,
        "gt_diagnosis": case_json["gt_diagnosis"],
        "total_turns": len(case_turns),
        "turns": [],
    }
    for _, row in case_turns.iterrows():
        trunk["turns"].append({
            "turn": int(row["turn"]),
            "top1_ddx": str(row.get("top1_ddx", "")),
            "conclusion": str(row.get("conclusion", "")),
            "diag_status": str(row.get("diag_status", "")),
            "tests_parsed": str(row.get("tests_parsed", "")),
            "is_done": bool(row.get("is_done", False)),
        })
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(trunk, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Step 4: Main generation loop
# ---------------------------------------------------------------------------

async def run_branch_generation(args):
    """Main: sample cases, reconstruct trunk states, generate branch paths.

    Uses cross-case batching: all branches across all cases are generated
    together in lockstep, just like the original inference_baichuan.py
    processes all trunk cases at once.
    """
    import copy

    # Load trunk data
    print("Loading trunk data...")
    df_turns, df_cases = load_trunk_data(TRUNK_XLSX)
    print(f"  {len(df_cases)} cases, {len(df_turns)} turns")

    # Use custom data dir if provided
    data_dir = args.data_dir

    # Sample or load selected cases
    cases_json_path = os.path.join(data_dir, "selected_cases.json")
    if os.path.exists(cases_json_path) and args.resume:
        with open(cases_json_path) as f:
            selected_ids = json.load(f)
        print(f"Loaded {len(selected_ids)} previously selected case IDs")
    else:
        if args.num_cases in (None,0):
            # Use ALL trunk cases (excluding test cases)
            test_ids = set()
            if os.path.exists(TEST_CASE_FILE):
                with open(TEST_CASE_FILE) as f:
                    test_ids = {line.strip() for line in f if line.strip()}
            all_ids = df_cases["case_id"].tolist()
            selected_ids = [cid for cid in all_ids
                            if os.path.exists(os.path.join(JSON_FOLDER, f"{cid}.json"))
                            and cid not in test_ids]
            print(f"  Using ALL {len(selected_ids)} trunk cases (excluding {len(test_ids)} test cases)")
        else:
            selected_ids = sample_cases(df_cases, args.num_cases, args.seed, JSON_FOLDER, TEST_CASE_FILE)
        save_selected_cases(selected_ids, df_turns, df_cases, data_dir)

    # Load case JSONs from the case folder
    print("\nLoading case JSONs...")
    case_jsons = {}
    for cid in selected_ids:
        path = os.path.join(JSON_FOLDER, f"{cid}.json")
        if os.path.exists(path):
            case_jsons[cid] = parse_case_json(path)
        else:
            print(f"  [Missing] {path}")
    print(f"  Loaded {len(case_jsons)} / {len(selected_ids)} case JSONs")

    # Build work list — all remaining branches
    total_branches = 0
    skip_count = 0
    work_items = []  # (case_id, branch_turn, path_id)
    for cid in selected_ids:
        if cid not in case_jsons:
            continue
        case_turns = df_turns[df_turns["case_id"] == cid]
        n_turns = len(case_turns)
        for bt in range(2, n_turns + 1):
            for pid in range(1, args.k + 1):
                total_branches += 1
                if args.resume and branch_exists(data_dir, cid, bt, pid):
                    skip_count += 1
                else:
                    work_items.append((cid, bt, pid))

    remaining = len(work_items)
    print(f"\nTotal branches: {total_branches}")
    print(f"Already done (skip): {skip_count}")
    print(f"Remaining: {remaining}")

    if not work_items:
        print("All branches already generated. Nothing to do.")
        return

    # Initialize vLLM
    print("\nInitializing BaichuanM3-235B...")
    llm = init_vllm(tensor_parallel_size=args.tp, max_model_len=args.max_model_len)
    sampling_params = SamplingParams(
        temperature=0.8, top_p=0.95, top_k=20, max_tokens=8000,
    )

    oracle_sem = asyncio.Semaphore(ORACLE_CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=ORACLE_CONCURRENCY + 5)

    completed = 0
    failed = 0
    t_start = time.time()

    async with aiohttp.ClientSession(connector=connector, trust_env=True) as session:

        # Phase 1: Reconstruct trunk states for all branches
        # Cache per (case_id, branch_turn) to avoid redundant GPT-4o calls
        print("\nReconstructing trunk states...")
        trunk_cache = {}  # (case_id, branch_turn) → (state, inherited_turns)
        unique_keys = sorted(set((cid, bt) for cid, bt, pid in work_items))
        for i, (cid, bt) in enumerate(unique_keys):
            trunk_cache[(cid, bt)] = await reconstruct_trunk_state(
                cid, bt, df_turns, case_jsons[cid], session, oracle_sem
            )
            save_trunk_summary(data_dir, cid, df_turns, case_jsons[cid])
            if (i + 1) % 20 == 0 or i + 1 == len(unique_keys):
                print(f"  {i + 1}/{len(unique_keys)} trunk states reconstructed")

        # Phase 2: Initialize all branch states
        print(f"\nInitializing {remaining} branches...")
        active = []
        for cid, bt, pid in work_items:
            trunk_state, inherited_turns = trunk_cache[(cid, bt)]
            active.append({
                "case_id": cid,
                "branch_turn": bt,
                "path_id": pid,
                "state": copy.deepcopy(trunk_state),
                "case_json": case_jsons[cid],
                "inherited_turns": inherited_turns,
                "trunk_state_ref": trunk_state,  # read-only ref for result output
                "branch_turns_log": [],
                "current_turn": bt,
                "done": False,
            })

        # Phase 3: Main generation loop — all branches advance together
        print(f"\nStarting cross-case batched generation...")
        step = 0
        while active:
            elapsed = time.time() - t_start
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (remaining - completed) / rate / 60 if rate > 0 else 0
            print(f"\n--- Step {step + 1}: {len(active)} active branches "
                  f"({completed}/{remaining} done) [ETA: {eta:.0f} min] ---")

            try:
                # Build prompts for all active branches
                messages_list = [
                    _build_branch_messages(b["state"], b["current_turn"])
                    for b in active
                ]

                # Batch vLLM generate — all branches at once
                t0 = time.time()
                responses = vllm_batch_generate(llm, messages_list, sampling_params)
                gen_time = time.time() - t0
                print(f"  vLLM: {len(responses)} responses in {gen_time:.1f}s")

                # Parse all responses
                parsed_list = [parse_response(r) for r in responses]
                clean_list = [strip_think_blocks(r) for r in responses]

                # Batch extract tests via GPT-4o (all concurrent)
                tests_results = await asyncio.gather(*[
                    parse_tests_from_response(session, oracle_sem, clean)
                    for clean in clean_list
                ])

                # Update done_tests for each branch
                for i, b in enumerate(active):
                    done_lower = {t.lower().strip() for t in b["state"]["done_tests"]}
                    for t in tests_results[i]:
                        if t.lower().strip() not in done_lower:
                            b["state"]["done_tests"].append(t)
                            done_lower.add(t.lower().strip())

                # Batch oracle calls (skip branches that are done)
                oracle_tasks = []
                oracle_indices = []
                for i, b in enumerate(active):
                    if not parsed_list[i]["is_done"]:
                        oracle_tasks.append(
                            _oracle_call(
                                session, oracle_sem,
                                [
                                    {"role": "system", "content": build_oracle_system_prompt(
                                        b["case_json"]["full_case_summary"],
                                        b["case_json"]["ancillary_tests"],
                                    )},
                                    {"role": "user", "content": (
                                        "The physician submitted the following diagnostic response.\n"
                                        "Identify all tests they requested and provide results or "
                                        "UNAVAILABLE status for each.\n\n"
                                        + clean_list[i]
                                    )},
                                ],
                                max_tokens=800,
                            )
                        )
                        oracle_indices.append(i)

                oracle_results = await asyncio.gather(*oracle_tasks) if oracle_tasks else []
                oracle_texts = [""] * len(active)
                for j, oi in enumerate(oracle_indices):
                    oracle_texts[oi] = oracle_results[j]

            except Exception as e:
                # If vLLM or API calls fail for the whole step, log and retry next step
                print(f"  STEP FAILED: {e}")
                failed += len(active)
                log_path = os.path.join(data_dir, "generation_log.jsonl")
                with open(log_path, "a") as f:
                    for b in active:
                        f.write(json.dumps({
                            "case_id": b["case_id"],
                            "branch_turn": b["branch_turn"],
                            "path_id": b["path_id"],
                            "status": "failed",
                            "error": str(e),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }) + "\n")
                break

            # Update states, log turns, save finished branches
            still_active = []
            for i, b in enumerate(active):
                parsed = parsed_list[i]
                clean_response = clean_list[i]
                oracle_text = oracle_texts[i]

                # Update unavailable tests
                if oracle_text:
                    new_unavail = _extract_unavailable_tests(oracle_text)
                    unavail_lower = {t.lower().strip() for t in b["state"]["unavailable_tests"]}
                    for t in new_unavail:
                        if t.lower().strip() not in unavail_lower:
                            b["state"]["unavailable_tests"].append(t)
                            unavail_lower.add(t.lower().strip())

                # Update state
                b["state"]["current_oracle"] = oracle_text
                b["state"]["recent_turns"].append(
                    build_turn_detail(b["current_turn"], parsed, oracle_text, tests_results[i])
                )
                if parsed["top1_ddx"]:
                    b["state"]["last_top1_ddx"] = parsed["top1_ddx"]
                if parsed["conclusion"]:
                    b["state"]["last_conclusion"] = parsed["conclusion"]

                # Log turn
                prompt_text = "\n\n".join(
                    f"[{m['role']}]\n{m['content']}" for m in messages_list[i]
                )
                b["branch_turns_log"].append({
                    "turn": b["current_turn"],
                    "prompt": prompt_text,
                    "raw_response": clean_response,
                    "top1_ddx": parsed["top1_ddx"],
                    "conclusion": parsed["conclusion"],
                    "diag_status": parsed["diag_status"],
                    "model_done": parsed["model_done"],
                    "rule_done": parsed["rule_done"],
                    "tests_parsed": tests_results[i],
                    "oracle_response": oracle_text,
                    "is_done": parsed["is_done"],
                })

                # Check if branch is finished
                is_finished = parsed["is_done"] or b["current_turn"] >= MAX_TURNS
                if is_finished:
                    result = {
                        "case_id": b["case_id"],
                        "branch_turn": b["branch_turn"],
                        "path_id": b["path_id"],
                        "trunk_context": {
                            "turns_inherited": b["branch_turn"] - 1,
                            "done_tests_inherited": list(b["trunk_state_ref"]["done_tests"]),
                            "unavailable_tests_inherited": list(b["trunk_state_ref"]["unavailable_tests"]),
                        },
                        "inherited_turns": b["inherited_turns"],
                        "branch_turns": b["branch_turns_log"],
                        "total_branch_turns": len(b["branch_turns_log"]),
                        "final_conclusion": b["state"]["last_conclusion"],
                        "final_top1_ddx": b["state"]["last_top1_ddx"],
                        "metadata": {
                            "model": "BaichuanM3-235B-FP8",
                            "temperature": 0.8,
                            "top_p": 0.95,
                            "max_tokens": 8000,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    }
                    save_branch_result(data_dir, b["case_id"], b["branch_turn"], b["path_id"], result)
                    completed += 1
                    print(f"    Done: {b['case_id']} t{b['branch_turn']}_p{b['path_id']} "
                          f"({len(b['branch_turns_log'])} turns, "
                          f"conclusion: {b['state']['last_conclusion'][:50]})")
                else:
                    b["current_turn"] += 1
                    still_active.append(b)

            active = still_active
            step += 1

    elapsed = time.time() - t_start

    # Save summary report
    report = {
        "total_cases": len(selected_ids),
        "total_branches_expected": total_branches,
        "previously_completed": skip_count,
        "completed_this_run": completed,
        "failed_this_run": failed,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    report_path = os.path.join(data_dir, "summary_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Branch generation complete")
    print(f"  Completed: {completed}")
    print(f"  Failed: {failed}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Report: {report_path}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Branch data generation")
    parser.add_argument(
        "--num_cases", type=int, default=None,
        help="Number of cases to sample from the trunk set",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for case sampling",
    )
    parser.add_argument(
        "--k", type=int, default=2,
        help="Number of branch paths per turn position",
    )
    parser.add_argument(
        "--tp", type=int, default=4,
        help="Tensor parallel size for vLLM",
    )
    parser.add_argument(
        "--max_model_len", type=int, default=16384,
        help="Max model context length",
    )
    parser.add_argument(
        "--data-dir", type=str, default=DATA_DIR,
        help="Output directory for branch data",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip already-completed branches",
    )
    args = parser.parse_args()

    asyncio.run(run_branch_generation(args))


if __name__ == "__main__":
    main()