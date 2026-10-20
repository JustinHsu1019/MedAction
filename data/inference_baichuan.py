"""

Multi-turn diagnostic inference with Baichuan-M3-235B as the diagnostician and GPT-4o
as the test-result oracle (inference only, no evaluation).
Flow:
  1. For each case, run up to MAX_TURNS rounds:
       - Baichuan-M3 generates DDx + test orders (vLLM batch)
       - GPT-4o oracle returns test results (or UNAVAILABLE)
       - Repeat until model says DONE
  2. Save all turn-by-turn results to Excel
"""

import os

import re
import json
import asyncio
import argparse
import time
import pickle

import aiohttp
import pandas as pd
import nest_asyncio
from vllm import LLM, SamplingParams

nest_asyncio.apply()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Local model path or HF model ID. Override via env: BAICHUAN_M3_PATH=...
MODEL_PATH = os.environ.get("BAICHUAN_M3_PATH", "baichuan-inc/Baichuan-M3-235B")
JSON_FOLDER = os.path.join(os.path.dirname(__file__), "cases")

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY env var is required (GPT-4o oracle). "
        "Export it before running: `export OPENAI_API_KEY=sk-...`"
    )
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

ORACLE_MODEL = "gpt-4o"
ORACLE_CONCURRENCY = 20

MAX_TURNS = 8
RECENT_TURNS_WINDOW = 2

DEFAULT_CASE_ID = "PMC11342857"
CASE_ID_FILE = os.path.join(os.path.dirname(__file__), "case_ids.txt")


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an experienced clinical physician performing a structured diagnostic workup."
)

INITIAL_PROMPT = """\
You are seeing this patient for the first time. Analyze the case and respond using EXACTLY \
the section headers below in EXACTLY this order. Do not skip or rename any section.

Patient Case Summary:
{case}

---

### Chain of Thought:
[Think step by step. Label each step <step 1>, <step 2>, etc.]

### DDx List:
[Ranked differentials. Be specific.]
1. Diagnosis name - one sentence reasoning
2. Diagnosis name - one sentence reasoning
3. Diagnosis name - one sentence reasoning

### Pivot:
[One paragraph: the single most important diagnostic question right now and why.]

### Primary Actions:
[Most targeted, high-yield tests to resolve top differentials THIS turn. Specific test names only.]
1. test name - purpose
2. test name - purpose

### Additional Information Required:
[Any broader workup beyond Primary Actions: routine labs, imaging, history.
If a test comes back normal or shows no significant findings, do NOT order
any further tests in the same category - move to a completely different
diagnostic category or test a different aspect of the presentation.
Write "Not required." only if you are fully confident in the final diagnosis.]
1. category: specific test or data needed

### Diagnostic Status:
[Write EXACTLY one word: either DONE or CONTINUE.
Write DONE only if you have sufficient information to make a definitive final diagnosis right now.
Write CONTINUE if you need more test results before concluding.]

### Conclusion:
[Your single best current diagnosis. Be specific. One sentence only.]\
"""

FOLLOWUP_PROMPT = """\
Patient Case Summary:
{case}

--- GLOBAL STATUS (cumulative across all turns) ---

All Tests Ordered So Far:
{done_tests_block}

Tests Confirmed UNAVAILABLE (do NOT re-order these):
{unavailable_tests_block}
NOTE: When key diagnostic tests are UNAVAILABLE, this is a signal that you cannot confirm \
your current leading hypothesis through the expected pathway. You should treat UNAVAILABLE \
tests as a reason to PIVOT your diagnostic direction — consider alternative diagnoses that \
can be confirmed or excluded using different, available tests rather than persisting with \
the same differential.

--- RECENT DIAGNOSTIC HISTORY (last {window_size} turn(s)) ---

{recent_turns_block}

--- NEW RESULTS THIS TURN ---

{oracle_results}

---

Respond using EXACTLY the section headers below in EXACTLY this order.

### Chain of Thought:
[Step-by-step reasoning.
CRITICAL RULES:
- If a result is "Normal", you MUST accept this as a definitive fact that rules out \
your previous hypothesis. Immediately PIVOT to different diagnostic possibilities.
- If key tests are UNAVAILABLE, you cannot confirm your current hypothesis through \
that pathway. Reassess whether a different diagnosis — one testable with available \
methods — better fits the data. Do NOT simply wait or repeat the same differential.]

### DDx List:
[Updated differentials. Reflect that Normal results have excluded certain possibilities \
and that UNAVAILABLE tests may mean your leading hypothesis cannot be verified — consider \
alternative diagnoses.]

### Pivot:
[Explain how Normal results and/or UNAVAILABLE tests shifted your diagnostic direction \
and what different path you are pursuing now.]

### Primary Actions:
[New targeted tests ONLY.
RULES:
1. If a test is listed under "Tests Confirmed UNAVAILABLE" above, you are FORBIDDEN from re-ordering it.
2. If a test is listed under "All Tests Ordered So Far", do NOT re-order it.
3. You must either:
   a) Order a different, alternative diagnostic test that IS available.
   b) Move to a final "Conclusion" if no other available tests can help.
4. DO NOT use words like "Await", "Wait for", or "Review previous tests".]

### Additional Information Required:
[Any broader workup. Write "Not required." if you are shifting to a final conclusion.]

### Diagnostic Status:
[Write EXACTLY one word: either DONE or CONTINUE.
Write DONE only if you have sufficient information right now to make a definitive final diagnosis.
Write CONTINUE if further workup is still needed.]

### Conclusion:
[Single best current diagnosis given all available info.]\
"""


def build_oracle_system_prompt(full_case_summary, ancillary_tests):
    """System prompt for the GPT-4o oracle that returns test results."""
    anc = ancillary_tests if ancillary_tests else "None available."
    return (
        "You are a hospital administrator providing status updates on diagnostic requests.\n\n"
        "Guidelines:\n"
        "1. If the requested test is present in the Ancillary Test Results below, provide the specific findings.\n"
        "2. If the test is NOT in the data, respond with:\n"
        '   "[test name]: This test is currently UNAVAILABLE due to equipment maintenance '
        'or lack of specialized personnel. You must proceed with clinical diagnosis or alternative available testing."\n\n'
        f"Patient Case\n{full_case_summary}\n\n"
        f"Ancillary Test Results\n{anc}"
    )


PARSE_TESTS_SYSTEM = (
    "You are a medical text organizer.\n"
    "Given a physician's full diagnostic response, extract ALL diagnostic tests or examinations "
    "they have requested or ordered into a flat JSON array.\n\n"
    "Rules:\n"
    "- Extract ONLY test/examination names, not purposes or reasoning.\n"
    '- Split compound entries: "CBC, CMP, LFTs" into three separate items.\n'
    "- Look in all sections: Primary Actions, Additional Information Required, or anywhere tests are mentioned.\n"
    '- If no tests are requested, return [].\n'
    '- Be specific: "serum troponin I" not just "troponin".\n\n'
    "Return ONLY a valid JSON array, no markdown, no extra text:\n"
    '["test name 1", "test name 2", ...]'
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_case_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    case_id = data.get("id", os.path.splitext(os.path.basename(json_path))[0])
    case_summary = data["generate_case"]["case_summary"]

    initial_obs = case_summary.strip()
    ancillary = ""
    if "Ancillary Tests" in case_summary:
        paragraphs = case_summary.strip().split("\n")
        for idx, para in enumerate(paragraphs):
            if "Ancillary Tests" in para:
                initial_obs = "\n".join(paragraphs[:idx]).strip()
                ancillary = "\n".join(paragraphs[idx:]).strip()
                break

    gt_tests = [
        item["test_name"].strip()
        for item in data.get("gt_info_split", [])
        if item.get("test_name", "").strip()
    ]
    gt_diagnosis = data["generate_case"].get("final_diagnosis", "").strip()

    return {
        "case_id": case_id,
        "initial_obs": initial_obs,
        "ancillary_tests": ancillary,
        "full_case_summary": case_summary,
        "gt_tests": gt_tests,
        "gt_diagnosis": gt_diagnosis,
    }


def load_single_case(json_folder, case_id):
    path = os.path.join(json_folder, f"{case_id}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Case file not found: {path}")
    return parse_case_json(path)


def load_cases_by_ids(json_folder, case_id_file):
    with open(case_id_file, encoding="utf-8") as f:
        case_ids = [line.strip() for line in f if line.strip()]
    print(f"Case IDs loaded: {len(case_ids)} from {case_id_file}")

    cases = []
    for case_id in case_ids:
        path = os.path.join(json_folder, f"{case_id}.json")
        if not os.path.exists(path):
            print(f"  [Missing] {path}")
            continue
        try:
            cases.append(parse_case_json(path))
        except Exception as e:
            print(f"  [Parse error] {path}: {e}")

    print(f"Loaded {len(cases)}/{len(case_ids)} cases")
    return cases


def load_distillation_cases(json_folder, exclude_file):
    """Load all cases from json_folder EXCLUDING those listed in exclude_file (test set)."""
    # Load test case IDs to exclude
    exclude_ids = set()
    if exclude_file and os.path.exists(exclude_file):
        with open(exclude_file, encoding="utf-8") as f:
            exclude_ids = {line.strip() for line in f if line.strip()}
    print(f"Excluding {len(exclude_ids)} test cases from {exclude_file}")

    # Scan all JSON files in folder
    import glob
    all_jsons = sorted(glob.glob(os.path.join(json_folder, "*.json")))
    cases = []
    skipped = 0
    for path in all_jsons:
        case_id = os.path.splitext(os.path.basename(path))[0]
        if case_id in exclude_ids:
            skipped += 1
            continue
        try:
            cases.append(parse_case_json(path))
        except Exception as e:
            print(f"  [Parse error] {path}: {e}")

    print(f"Distillation pool: {len(cases)} cases ({skipped} test cases excluded, "
          f"{len(all_jsons)} total in folder)")
    return cases


# ---------------------------------------------------------------------------
# vLLM diagnostician
# ---------------------------------------------------------------------------

def init_vllm(tensor_parallel_size=4, max_model_len=16384):
    print(f"Loading Baichuan-M3-235B FP8 with TP={tensor_parallel_size}, "
          f"max_model_len={max_model_len} ...")
    t0 = time.time()
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=tensor_parallel_size,
        quantization="fp8",
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.92,
        enforce_eager=False,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")
    return llm


def vllm_batch_generate(llm, messages_list, sampling_params):
    """Apply chat template and run vLLM batch generation."""
    tokenizer = llm.get_tokenizer()
    prompts = []
    for msgs in messages_list:
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)

    outputs = llm.generate(prompts, sampling_params)
    return [out.outputs[0].text for out in outputs]


# ---------------------------------------------------------------------------
# OpenAI API helpers (oracle only)
# ---------------------------------------------------------------------------

async def _openai_call(session, semaphore, messages, *, model, max_tokens=1200, json_mode=False):
    headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": 0.0,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    async with semaphore:
        for attempt in range(3):
            try:
                async with session.post(
                    OPENAI_URL, headers=headers, json=payload,
                    timeout=aiohttp.ClientTimeout(total=180),
                ) as resp:
                    if resp.status != 200:
                        print(f"  [{model} {resp.status}] {(await resp.text())[:200]}")
                        await asyncio.sleep(2 ** attempt)
                        continue
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print(f"  [{model} attempt {attempt + 1}] {e}")
                await asyncio.sleep(2 ** attempt)
    raise RuntimeError(f"{model} call failed after 3 attempts")


async def _oracle_call(session, sem, messages, max_tokens=1200, json_mode=False):
    return await _openai_call(
        session, sem, messages, model=ORACLE_MODEL, max_tokens=max_tokens, json_mode=json_mode,
    )


async def parse_tests_from_response(session, sem, raw_response):
    """Ask GPT-4o to extract the list of test names from a model response."""
    if not raw_response.strip():
        return []
    raw = await _oracle_call(
        session, sem,
        [{"role": "system", "content": PARSE_TESTS_SYSTEM},
         {"role": "user", "content": raw_response}],
        max_tokens=600,
    )
    raw = re.sub(r"^```[\w]*\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw).strip()
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return [str(t).strip() for t in result if str(t).strip()]
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Think-block stripping
# ---------------------------------------------------------------------------

def strip_think_blocks(text):
    """Remove thinking content from model output.

    Handles three cases:
      1. <think>...</think> — full block
      2. ...</think> — vLLM consumes <think> as part of generation prompt
      3. <think>... (unclosed) — model ran out of tokens mid-thinking
    """
    # Case 1: full <think>...</think> blocks
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Case 2: output starts with thinking content followed by </think> (no opening tag)
    if "</think>" in clean:
        clean = clean.split("</think>", 1)[1]
    # Case 3: unclosed <think>
    if "<think>" in clean:
        clean = clean[:clean.index("<think>")]
    return clean.strip()


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

RESPONSE_HEADERS = [
    "### Chain of Thought:",
    "### DDx List:",
    "### Pivot:",
    "### Primary Actions:",
    "### Additional Information Required:",
    "### Diagnostic Status:",
    "### Conclusion:",
]


def _normalize_headers(response):
    canonical = {}
    for h in RESPONSE_HEADERS:
        bare = h.lstrip("# ").rstrip(":").strip()
        canonical[bare.lower()] = h

    lines = response.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip().lstrip("#").strip()
        stripped = stripped.replace("**", "").replace("*", "").strip()
        stripped = stripped.rstrip(":").strip()
        if stripped.lower() in canonical:
            lines[i] = canonical[stripped.lower()]
    return "\n".join(lines)


def extract_section(text, header):
    bare = header.lstrip("# ").strip()
    target = header if header in text else (bare if bare in text else None)
    if target is None:
        return ""
    after = text.split(target)[-1]
    for other in RESPONSE_HEADERS:
        if other == header:
            continue
        other_bare = other.lstrip("# ").strip()
        for variant in (other, other_bare):
            if variant in after:
                after = after.split(variant)[0]
    return after.strip()


def parse_response(response):
    clean = strip_think_blocks(response)
    clean = _normalize_headers(clean)

    ddx_list = extract_section(clean, "### DDx List:")
    primary_actions = extract_section(clean, "### Primary Actions:")
    additional_info = extract_section(clean, "### Additional Information Required:")
    diag_status_raw = extract_section(clean, "### Diagnostic Status:")
    conclusion = extract_section(clean, "### Conclusion:")
    pivot = extract_section(clean, "### Pivot:")

    top1_ddx = ""
    for line in ddx_list.splitlines():
        line = line.strip()
        if re.match(r"^1[.)]\s*", line):
            top1_ddx = re.sub(r"^1[.)]\s*", "", line).split("-")[0].strip()
            top1_ddx = re.sub(r"\*+", "", top1_ddx).strip()
            break

    conclusion_clean = re.sub(r"</?Answer>|\*+", "", conclusion).strip()

    final_patterns = ["move to a final conclusion", "final conclusion", "not required"]
    primary_done = not primary_actions.strip() or any(x in primary_actions.lower() for x in final_patterns)
    additional_done = not additional_info.strip() or "not required" in additional_info.lower()
    rule_done = primary_done and additional_done
    model_done = "done" in diag_status_raw.lower()[:20]

    parse_valid = bool(conclusion_clean) and bool(diag_status_raw.strip())

    return {
        "raw_response": response.strip(),
        "parse_valid": parse_valid,
        "ddx_list": ddx_list,
        "top1_ddx": top1_ddx,
        "pivot": pivot,
        "primary_actions": primary_actions,
        "additional_info": additional_info,
        "diag_status": diag_status_raw.strip()[:10],
        "conclusion": conclusion_clean,
        "rule_done": rule_done,
        "model_done": model_done,
        "is_done": rule_done or model_done,
    }


# ---------------------------------------------------------------------------
# Context management
# ---------------------------------------------------------------------------

def _extract_unavailable_tests(oracle_text):
    unavailable = []
    for line in oracle_text.splitlines():
        if "UNAVAILABLE" in line.upper():
            match = re.match(r"^[-\u2022*\d.)\s]*(.+?):\s*.*UNAVAILABLE", line, re.IGNORECASE)
            if match:
                unavailable.append(match.group(1).strip())
    return unavailable


def build_turn_detail(turn, parsed, oracle_text, tests_ordered):
    parts = [
        f"=== Turn {turn} ===",
        f"DDx List:\n{parsed['ddx_list']}" if parsed["ddx_list"] else "DDx List: (none)",
        f"Pivot:\n{parsed['pivot']}" if parsed["pivot"] else "",
        "Tests ordered this turn: " + (", ".join(tests_ordered) if tests_ordered else "(none)"),
        f"Oracle results:\n{oracle_text}" if oracle_text else "Oracle results: (none)",
        f"Conclusion: {parsed['conclusion']}" if parsed["conclusion"] else "",
        f"Diagnostic Status: {parsed['diag_status']}",
    ]
    return "\n".join(p for p in parts if p)


def build_global_status(state):
    ordered = "\n".join(f"- {t}" for t in state["done_tests"]) or "(none yet)"
    unavail = "\n".join(f"- {t}" for t in state["unavailable_tests"]) or "(none)"
    return ordered, unavail


# ---------------------------------------------------------------------------
# Multi-turn inference pipeline (no eval)
# ---------------------------------------------------------------------------

def _init_case_state(case):
    return {
        "case": case,
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


def _build_messages(state, turn):
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


async def _process_turn(state, parsed, messages, turn, session, sem):
    """Post-inference: extract tests, call oracle, update state."""
    case = state["case"]

    clean_response = strip_think_blocks(parsed["raw_response"])
    tests_this_turn = await parse_tests_from_response(session, sem, clean_response)

    done_lower = {t.lower().strip() for t in state["done_tests"]}
    for t in tests_this_turn:
        if t.lower().strip() not in done_lower:
            state["done_tests"].append(t)
            done_lower.add(t.lower().strip())

    oracle_text = ""
    if not parsed["is_done"]:
        oracle_text = await _oracle_call(
            session, sem,
            [
                {"role": "system", "content": build_oracle_system_prompt(
                    case["full_case_summary"], case["ancillary_tests"]
                )},
                {"role": "user", "content": (
                    "The physician submitted the following diagnostic response.\n"
                    "Identify all tests they requested and provide results or UNAVAILABLE status for each.\n\n"
                    + clean_response
                )},
            ],
            max_tokens=800,
        )

    if oracle_text:
        new_unavail = _extract_unavailable_tests(oracle_text)
        unavail_lower = {t.lower().strip() for t in state["unavailable_tests"]}
        for t in new_unavail:
            if t.lower().strip() not in unavail_lower:
                state["unavailable_tests"].append(t)
                unavail_lower.add(t.lower().strip())

    state["current_oracle"] = oracle_text
    state["recent_turns"].append(
        build_turn_detail(turn, parsed, oracle_text, tests_this_turn)
    )
    prompt_text = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)

    state["turn_logs"].append({
        "case_id": case["case_id"],
        "turn": turn,
        "prompt": prompt_text,
        "raw_response": clean_response,
        "top1_ddx": parsed["top1_ddx"],
        "conclusion": parsed["conclusion"],
        "diag_status": parsed["diag_status"],
        "model_done": parsed["model_done"],
        "rule_done": parsed["rule_done"],
        "tests_parsed": ", ".join(tests_this_turn),
        "oracle_response": oracle_text,
        "is_done": parsed["is_done"],
    })

    if parsed["is_done"]:
        state["done"] = True


def _checkpoint_path(output_xlsx):
    """Return checkpoint file path for a given output xlsx."""
    base, _ = os.path.splitext(output_xlsx)
    return f"{base}_checkpoint.pkl"


def save_checkpoint(state, start_turn, output_xlsx):
    """Save inference state to a checkpoint file after each turn (atomic write)."""
    ckpt_path = _checkpoint_path(output_xlsx)
    tmp_path = ckpt_path + ".tmp"
    case_ids = [s["case"]["case_id"] for s in state]
    data = {"state": state, "next_turn": start_turn, "case_ids": case_ids}
    with open(tmp_path, "wb") as f:
        pickle.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, ckpt_path)  # atomic on POSIX
    done_count = sum(1 for s in state if s["done"])
    print(f"  Checkpoint saved: {ckpt_path} (next_turn={start_turn}, done={done_count}/{len(state)})")


def load_checkpoint(output_xlsx, expected_case_ids=None):
    """Load checkpoint if it exists. Returns (state, next_turn) or None.
    If expected_case_ids is provided, validates that checkpoint matches."""
    ckpt_path = _checkpoint_path(output_xlsx)
    if not os.path.exists(ckpt_path):
        return None
    with open(ckpt_path, "rb") as f:
        data = pickle.load(f)
    # Validate case IDs match
    if expected_case_ids is not None:
        ckpt_ids = data.get("case_ids", [])
        if ckpt_ids != expected_case_ids:
            print(f"  WARNING: Checkpoint case IDs don't match current cases "
                  f"(ckpt={len(ckpt_ids)}, current={len(expected_case_ids)}). Ignoring checkpoint.")
            return None
    print(f"  Checkpoint loaded: {ckpt_path} (next_turn={data['next_turn']})")
    return data["state"], data["next_turn"]


def remove_checkpoint(output_xlsx):
    """Remove checkpoint file after successful completion."""
    ckpt_path = _checkpoint_path(output_xlsx)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print(f"  Checkpoint removed: {ckpt_path}")


async def run_inference(cases, llm, session, oracle_sem, resume=False, output_xlsx=None):
    """Run multi-turn inference only (no evaluation)."""
    start_turn = 1
    expected_ids = [c["case_id"] for c in cases]

    # Try to resume from checkpoint
    if resume and output_xlsx:
        ckpt = load_checkpoint(output_xlsx, expected_case_ids=expected_ids)
        if ckpt:
            state, start_turn = ckpt
            n = len(state)
            done_count = sum(1 for s in state if s["done"])
            print(f"  Resuming from turn {start_turn}, {done_count}/{n} already done")
        else:
            print("  No checkpoint found, starting fresh")
            state = [_init_case_state(c) for c in cases]
            n = len(state)
    else:
        state = [_init_case_state(c) for c in cases]
        n = len(state)

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        top_k=20,
        max_tokens=8000,
    )

    for turn in range(start_turn, MAX_TURNS + 1):
        active = [s for s in state if not s["done"]]
        if not active:
            break
        print(f"\n--- Turn {turn}/{MAX_TURNS}  ({len(active)} active cases) ---")

        messages_list = [_build_messages(s, turn) for s in active]

        t0 = time.time()
        responses = vllm_batch_generate(llm, messages_list, sampling_params)
        gen_time = time.time() - t0
        print(f"  Baichuan-M3 done ({len(responses)} responses, {gen_time:.1f}s)")

        parsed_list = [parse_response(r) for r in responses]
        for s, parsed in zip(active, parsed_list):
            if parsed["top1_ddx"]:
                s["last_top1_ddx"] = parsed["top1_ddx"]
            if parsed["conclusion"]:
                s["last_conclusion"] = parsed["conclusion"]
            s["last_parse_valid"] = parsed["parse_valid"]
            valid_tag = "" if parsed["parse_valid"] else "  [PARSE FAILED]"
            print(f"  {s['case']['case_id'][:20]:20s}  top1={parsed['top1_ddx'][:30]:30s}  "
                  f"status={parsed['diag_status']}  done={parsed['is_done']}{valid_tag}")

        await asyncio.gather(*[
            _process_turn(s, parsed, msgs, turn, session, oracle_sem)
            for s, parsed, msgs in zip(active, parsed_list, messages_list)
        ])

        done_count = sum(1 for s in state if s["done"])
        print(f"  Turn {turn} complete. Done: {done_count}/{n}")

        # Save checkpoint after each turn
        if output_xlsx:
            save_checkpoint(state, turn + 1, output_xlsx)

    results = []
    for s in state:
        results.append({
            "case_id": s["case"]["case_id"],
            "gt_diagnosis": s["case"]["gt_diagnosis"],
            "gt_tests": s["case"]["gt_tests"],
            "turns_used": len(s["turn_logs"]),
            "final_conclusion": s["last_conclusion"],
            "final_top1_ddx": s["last_top1_ddx"],
            "parse_valid": s["last_parse_valid"],
            "turn_logs": s["turn_logs"],
        })
    return results


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(results, output_xlsx):
    all_logs = [log for r in results for log in r["turn_logs"]]

    case_summary = []
    for r in results:
        case_summary.append({
            "case_id": r["case_id"],
            "gt_diagnosis": r["gt_diagnosis"],
            "turns_used": r["turns_used"],
            "final_conclusion": r["final_conclusion"],
            "final_top1_ddx": r["final_top1_ddx"],
            "parse_valid": r["parse_valid"],
        })

    df_logs = pd.DataFrame(all_logs)
    df_cases = pd.DataFrame(case_summary).sort_values("case_id").reset_index(drop=True)

    os.makedirs(os.path.dirname(output_xlsx) or ".", exist_ok=True)
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        df_logs.to_excel(writer, sheet_name="Turn Details", index=False)
        df_cases.to_excel(writer, sheet_name="Case Summary", index=False)

    print(f"\nSaved: {output_xlsx}")
    print(f"  Cases: {len(df_cases)}")
    print(f"  Total turns: {len(df_logs)}")
    for _, row in df_cases.iterrows():
        print(f"  {row['case_id']}: {row['turns_used']} turns -> {row['final_conclusion'][:60]}")

    return df_cases, df_logs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Multi-turn diagnostic inference with Baichuan-M3-235B (FP8). Inference only."
    )
    parser.add_argument(
        "--mode", choices=["single", "distill"], default="distill",
        help="single: run one case; distill: run non-test cases (default: distill)",
    )
    parser.add_argument(
        "--case_id", type=str, default=DEFAULT_CASE_ID,
        help=f"Case ID for single mode (default: {DEFAULT_CASE_ID})",
    )
    parser.add_argument(
        "--case_id_file", type=str, default=None,
        help="File with case IDs (one per line). Used in distill mode to specify cases.",
    )
    parser.add_argument(
        "-n", "--num_cases", type=int, default=None,
        help="Number of cases to run",
    )
    parser.add_argument(
        "--tp", type=int, default=4,
        help="Tensor parallel size (default: 4)",
    )
    parser.add_argument(
        "--max_model_len", type=int, default=16384,
        help="Max model context length (default: 16384)",
    )
    parser.add_argument(
        "-k", type=int, default=1,
        help="Number of rejection sampling runs (default: 1). Each run produces a separate file.",
    )
    parser.add_argument(
        "--start_run", type=int, default=1,
        help="Starting run ID (default: 1). Useful to skip runs already completed.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint if available. Without this flag, starts fresh (ignores checkpoints).",
    )
    args = parser.parse_args()

    if args.mode == "single":
        output_xlsx = os.path.join(
            os.path.dirname(__file__),
            f"infer_result_baichuan_{args.case_id}.xlsx",
        )
    else:
        output_xlsx = os.path.join(
            os.path.dirname(__file__),
            f"infer_result_baichuan_{args.num_cases or 'all'}.xlsx",
        )
    print(f"Model: Baichuan-M3-235B (FP8)  TP={args.tp}")

    llm = init_vllm(tensor_parallel_size=args.tp, max_model_len=args.max_model_len)

    if args.mode == "single":
        cases = [load_single_case(JSON_FOLDER, args.case_id)]
    elif args.case_id_file:
        cases = load_cases_by_ids(JSON_FOLDER, args.case_id_file)
        cases = cases[:args.num_cases]
    else:
        cases = load_distillation_cases(JSON_FOLDER, CASE_ID_FILE)
        cases = cases[:args.num_cases]
    print(f"Running inference on {len(cases)} case(s), k={args.k} run(s)")

    for run_id in range(args.start_run, args.start_run + args.k):
        if args.k == 1 and args.start_run == 1:
            run_output = output_xlsx
        else:
            base, ext = os.path.splitext(output_xlsx)
            run_output = f"{base}_run{run_id}{ext}"

        # In resume mode, skip runs that already completed (output exists, no checkpoint)
        ckpt_path = _checkpoint_path(run_output)
        if args.resume and os.path.exists(run_output) and not os.path.exists(ckpt_path):
            print(f"\n  Run {run_id} already completed ({run_output} exists). Skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"  Run {run_id} of {args.start_run + args.k - 1}")
        print(f"  Output: {run_output}")
        if args.resume and os.path.exists(ckpt_path):
            print(f"  Resuming from checkpoint")
        print(f"{'='*60}")

        oracle_sem = asyncio.Semaphore(ORACLE_CONCURRENCY)
        connector = aiohttp.TCPConnector(limit=ORACLE_CONCURRENCY + 5)

        t_start = time.time()
        async with aiohttp.ClientSession(connector=connector, trust_env=True) as session:
            results = await run_inference(cases, llm, session, oracle_sem,
                                          resume=args.resume, output_xlsx=run_output)
        total_time = time.time() - t_start

        print(f"\nRun {run_id} inference time: {total_time:.1f}s")
        save_results(results, run_output)
        remove_checkpoint(run_output)  # only remove after xlsx is safely saved


if __name__ == "__main__":
    asyncio.run(main())

