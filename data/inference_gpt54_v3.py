"""
inference_gpt54_v3.py

Multi-turn diagnostic inference — GPT-5.4 as diagnostician, GPT-4o as oracle/evaluator.

Flow:
  1. For each case, run up to MAX_TURNS rounds:
       - GPT-5.4 generates DDx + test orders
       - GPT-4o oracle returns test results (or UNAVAILABLE)
       - Repeat with updated context until model says DONE
  2. Evaluate: GPT-4o scores test recall/precision and diagnosis accuracy
  3. Save results to Excel
"""

import os
import re
import json
import asyncio
import argparse

import aiohttp
import numpy as np
import pandas as pd
import nest_asyncio

nest_asyncio.apply()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

JSON_FOLDER = os.path.join(os.path.dirname(__file__), "cases")

OPENAI_KEY         = os.environ.get("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY env var is required (GPT-5.4 diagnostician + GPT-4o oracle/evaluator). "
        "Export it before running: `export OPENAI_API_KEY=sk-...`"
    )
OPENAI_URL         = "https://api.openai.com/v1/chat/completions"

# Diagnostician model
DIAGNOSTICIAN_MODEL = "gpt-5.4"
DIAGNOSTICIAN_CONCURRENCY = 10   # concurrent diagnostician calls per turn

# Oracle + evaluation model
ORACLE_MODEL       = "gpt-4o"
ORACLE_CONCURRENCY = 20

MAX_TURNS    = 8
RECENT_TURNS_WINDOW = 2
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


# GPT-4o system prompts for evaluation
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

MATCH_SYSTEM = (
    "You are a medical test matching expert.\n\n"
    "Given PREDICTED tests and GROUND TRUTH (GT) tests, determine which GT items are covered.\n\n"
    "RULES:\n"
    '1. Name equivalence: "CBC" == "Complete Blood Count", "CXR" == "Chest X-ray"\n'
    '2. Parent covers children: "MRI Brain" covers "MRI Brain T1", "MRI Brain FLAIR"\n'
    "3. Compound GT items (comma/slash separated): treat as ONE unit.\n"
    "4. One predicted test can cover multiple GT items. One GT item counted once only.\n"
    "5. Do NOT mark covered if overlap is minor.\n\n"
    "Return ONLY valid JSON:\n"
    '{\n  "gt_covered":   ["..."],\n  "gt_uncovered": ["..."],\n'
    '  "pred_used":    ["..."],\n  "pred_unused":  ["..."]\n}'
)

DIAG_EVAL_SYSTEM = (
    "You are a medical evaluation assistant.\n"
    "Determine if Predicted Diagnosis and Ground Truth Diagnosis are clinically equivalent.\n"
    'Return ONLY valid JSON: {"match": true, "reasoning": "..."}'
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_case_json(json_path):
    """
    Parse a MedRBench case JSON file.

    Returns a dict with:
      - case_id, initial_obs, ancillary_tests, full_case_summary
      - gt_tests (list), gt_diagnosis (str)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    case_id      = data.get("id", os.path.splitext(os.path.basename(json_path))[0])
    case_summary = data["generate_case"]["case_summary"]

    # Split case summary into initial observations and ancillary test results
    initial_obs = case_summary.strip()
    ancillary   = ""
    if "Ancillary Tests" in case_summary:
        paragraphs = case_summary.strip().split("\n")
        for idx, para in enumerate(paragraphs):
            if "Ancillary Tests" in para:
                initial_obs = "\n".join(paragraphs[:idx]).strip()
                ancillary   = "\n".join(paragraphs[idx:]).strip()
                break

    gt_tests = [
        item["test_name"].strip()
        for item in data.get("gt_info_split", [])
        if item.get("test_name", "").strip()
    ]
    gt_diagnosis = data["generate_case"].get("final_diagnosis", "").strip()

    return {
        "case_id":           case_id,
        "initial_obs":       initial_obs,
        "ancillary_tests":   ancillary,
        "full_case_summary": case_summary,
        "gt_tests":          gt_tests,
        "gt_diagnosis":      gt_diagnosis,
    }


def load_cases_by_ids(json_folder, case_id_file):
    """Load cases whose IDs are listed in case_id_file (one PMC ID per line)."""
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


# ---------------------------------------------------------------------------
# OpenAI API helpers
# ---------------------------------------------------------------------------

async def _openai_call(
    session, semaphore, messages, *,
    model, max_tokens=1200, json_mode=False,
    reasoning_effort=None,
):
    """
    Single async OpenAI call with exponential-backoff retry.

    Args:
        model: which model to call (e.g. "gpt-5.4" or "gpt-4o")
        reasoning_effort: if set, adds top-level "reasoning_effort" to payload.
                          Valid for GPT-5.4: "high", "medium", "low".
    """
    headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
    }
    # Only set temperature for non-reasoning models; reasoning models ignore it
    if reasoning_effort is None:
        payload["temperature"] = 0.0
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort

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
    """GPT-4o call for oracle / evaluation (no reasoning effort)."""
    return await _openai_call(
        session, sem, messages,
        model=ORACLE_MODEL, max_tokens=max_tokens, json_mode=json_mode,
    )


async def _diagnostician_call(session, sem, messages, reasoning_effort, max_tokens=8000):
    """GPT-5.4 call for the diagnostician (with reasoning effort)."""
    return await _openai_call(
        session, sem, messages,
        model=DIAGNOSTICIAN_MODEL, max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )


# ---------------------------------------------------------------------------
# GPT-4o helpers (oracle + evaluation)
# ---------------------------------------------------------------------------

async def parse_tests_from_response(session, sem, raw_response):
    """Ask GPT-4o to extract the list of test names from a model response."""
    if not raw_response.strip():
        return []
    raw = await _oracle_call(
        session, sem,
        [{"role": "system", "content": PARSE_TESTS_SYSTEM}, {"role": "user", "content": raw_response}],
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


async def gpt_match_tests(session, sem, full_response, gt_tests):
    """Compare predicted tests against ground-truth tests; return precision/recall components."""
    if not full_response.strip() or not gt_tests:
        return {"gt_covered": [], "gt_uncovered": list(gt_tests), "pred_used": [], "pred_unused": []}

    raw = await _oracle_call(
        session, sem,
        [
            {"role": "system", "content": MATCH_SYSTEM},
            {"role": "user", "content": (
                f"PHYSICIAN RESPONSE:\n{full_response}\n\n"
                f"GROUND TRUTH tests:\n{json.dumps(gt_tests, ensure_ascii=False)}\n\n"
                "Identify which GT tests the physician ordered anywhere in their response. Return JSON."
            )},
        ],
        max_tokens=1200, json_mode=True,
    )
    result = json.loads(raw)
    for k in ("gt_covered", "gt_uncovered", "pred_used", "pred_unused"):
        result.setdefault(k, [])
    return result


async def gpt_eval_diagnosis(session, sem, full_response, gt_diagnosis):
    """Ask GPT-4o whether the model's diagnosis matches ground truth."""
    if not full_response.strip():
        return {"match": False, "reasoning": "no response"}
    raw = await _oracle_call(
        session, sem,
        [
            {"role": "system", "content": DIAG_EVAL_SYSTEM},
            {"role": "user", "content": f"PHYSICIAN RESPONSE:\n{full_response}\n\nGround Truth Diagnosis: {gt_diagnosis}"},
        ],
        max_tokens=200, json_mode=True,
    )
    return json.loads(raw)


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
    """Normalize section headers to canonical '### Header:' form.

    Handles variations like:
      **DDx List:**  /  ## DDx List:  /  DDx List:  /  **DDx List**:
    """
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
    """Extract the content under a given markdown section header.

    Falls back to matching just the bare header name (without '### ')
    if the canonical form is not found.
    """
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
    """
    Parse the model's structured response into a dict of sections.

    Also determines whether the model considers the workup complete
    (either by explicit DONE status or by writing "Not required." in actions).
    """
    # Remove closed <think>...</think> blocks
    clean = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    # Handle unclosed <think> (model ran out of tokens mid-thinking):
    # Strip from <think> up to the first section header, preserving the rest.
    if "<think>" in clean:
        header_pattern = re.compile(
            r"^(#{1,3}\s*.+?:|(?:Chain of Thought|DDx List|Pivot|Primary Actions|"
            r"Additional Information Required|Diagnostic Status|Conclusion)\s*:)",
            re.MULTILINE | re.IGNORECASE,
        )
        think_start = clean.index("<think>")
        after_think = clean[think_start:]
        m = header_pattern.search(after_think)
        if m:
            clean = clean[:think_start] + after_think[m.start():]
        else:
            clean = clean[:think_start]
    clean = _normalize_headers(clean)

    ddx_list        = extract_section(clean, "### DDx List:")
    primary_actions = extract_section(clean, "### Primary Actions:")
    additional_info = extract_section(clean, "### Additional Information Required:")
    diag_status_raw = extract_section(clean, "### Diagnostic Status:")
    conclusion      = extract_section(clean, "### Conclusion:")
    pivot           = extract_section(clean, "### Pivot:")

    # Extract top-1 DDx for logging
    top1_ddx = ""
    for line in ddx_list.splitlines():
        line = line.strip()
        if re.match(r"^1[.)]\s*", line):
            top1_ddx = re.sub(r"^1[.)]\s*", "", line).split("-")[0].strip()
            top1_ddx = re.sub(r"\*+", "", top1_ddx).strip()
            break

    conclusion_clean = re.sub(r"</?Answer>|\*+", "", conclusion).strip()

    # Determine if workup is complete
    final_patterns  = ["move to a final conclusion", "final conclusion", "not required"]
    primary_done    = not primary_actions.strip() or any(x in primary_actions.lower() for x in final_patterns)
    additional_done = not additional_info.strip() or "not required" in additional_info.lower()
    rule_done       = primary_done and additional_done
    model_done      = "done" in diag_status_raw.lower()[:20]

    parse_valid = bool(conclusion_clean) and bool(diag_status_raw.strip())

    return {
        "raw_response":     response.strip(),
        "parse_valid":      parse_valid,
        "ddx_list":         ddx_list,
        "top1_ddx":         top1_ddx,
        "pivot":            pivot,
        "primary_actions":  primary_actions,
        "additional_info":  additional_info,
        "diag_status":      diag_status_raw.strip()[:10],
        "conclusion":       conclusion_clean,
        "rule_done":        rule_done,
        "model_done":       model_done,
        "is_done":          rule_done or model_done,
    }


# ---------------------------------------------------------------------------
# Context management (sliding window + global tracking)
# ---------------------------------------------------------------------------

def _extract_unavailable_tests(oracle_text):
    """Parse oracle response to find tests marked UNAVAILABLE."""
    unavailable = []
    for line in oracle_text.splitlines():
        if "UNAVAILABLE" in line.upper():
            match = re.match(r"^[-\u2022*\d.)\s]*(.+?):\s*.*UNAVAILABLE", line, re.IGNORECASE)
            if match:
                unavailable.append(match.group(1).strip())
    return unavailable


def build_turn_detail(turn, parsed, oracle_text, tests_ordered):
    """Build a full-detail record of one turn (no truncation). Stored for the sliding window."""
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
    """Build the compact global status block: cumulative tests and unavailable tests."""
    ordered = "\n".join(f"- {t}" for t in state["done_tests"]) or "(none yet)"
    unavail = "\n".join(f"- {t}" for t in state["unavailable_tests"]) or "(none)"
    return ordered, unavail


# ---------------------------------------------------------------------------
# Multi-turn pipeline
# ---------------------------------------------------------------------------

async def run_pipeline(cases, reasoning_effort, session, diag_sem, oracle_sem):
    """
    Run the full multi-turn diagnostic pipeline across all cases.

    Each turn:
      1. Build prompts for all still-active cases
      2. Concurrent GPT-5.4 calls (diagnostician)
      3. Parse responses
      4. Async GPT-4o: extract tests ordered + get oracle results
      5. Mark cases as done if model signals DONE

    After all turns: evaluate test recall/precision and diagnosis accuracy.
    """
    n     = len(cases)
    state = [_init_case_state(c) for c in cases]

    for turn in range(1, MAX_TURNS + 1):
        active = [s for s in state if not s["done"]]
        if not active:
            break
        print(f"\n--- Turn {turn}/{MAX_TURNS}  ({len(active)} active cases) ---")

        # Build prompts
        messages_list = [_build_messages(s, turn) for s in active]

        # Concurrent GPT-5.4 diagnostician calls
        responses = await asyncio.gather(*[
            _diagnostician_call(session, diag_sem, msgs, reasoning_effort)
            for msgs in messages_list
        ])
        print(f"  GPT-5.4 done ({len(responses)} responses)")

        # Parse and log
        parsed_list = [parse_response(r) for r in responses]
        for s, parsed in zip(active, parsed_list):
            if parsed["top1_ddx"]:   s["last_top1_ddx"]  = parsed["top1_ddx"]
            if parsed["conclusion"]: s["last_conclusion"] = parsed["conclusion"]
            s["last_parse_valid"] = parsed["parse_valid"]
            valid_tag = "" if parsed["parse_valid"] else "  [PARSE FAILED]"
            print(f"  {s['case']['case_id'][:20]:20s}  top1={parsed['top1_ddx'][:30]:30s}  "
                  f"status={parsed['diag_status']}  done={parsed['is_done']}{valid_tag}")

        # Async: extract tests + call oracle for each active case
        await asyncio.gather(*[
            _process_turn(s, parsed, msgs, turn, session, oracle_sem)
            for s, parsed, msgs in zip(active, parsed_list, messages_list)
        ])

        done_count = sum(1 for s in state if s["done"])
        print(f"  Turn {turn} complete. Done: {done_count}/{n}")

    print("\nAll turns complete. Running evaluation...")
    eval_results = await asyncio.gather(*[
        _evaluate_case(s, session, oracle_sem) for s in state
    ])
    return eval_results


def _init_case_state(case):
    return {
        "case":              case,
        "done":              False,
        "recent_turns":      [],
        "turn_logs":         [],
        "done_tests":        [],
        "unavailable_tests": [],
        "last_top1_ddx":     "",
        "last_conclusion":   "",
        "last_parse_valid":  True,
        "current_oracle":    "",
    }


def _build_messages(state, turn):
    """Construct the message list for a single case at the given turn."""
    case = state["case"]

    if turn == 1:
        user_content = INITIAL_PROMPT.format(case=case["initial_obs"])
    else:
        done_tests_block, unavailable_tests_block = build_global_status(state)

        recent = state["recent_turns"][-RECENT_TURNS_WINDOW:]
        recent_turns_block = "\n\n".join(recent) if recent else "(first follow-up turn)"

        user_content = FOLLOWUP_PROMPT.format(
            case                    = case["initial_obs"],
            done_tests_block        = done_tests_block,
            unavailable_tests_block = unavailable_tests_block,
            window_size             = len(recent),
            recent_turns_block      = recent_turns_block,
            oracle_results          = state["current_oracle"],
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


async def _process_turn(state, parsed, messages, turn, session, sem):
    """Post-inference async work for one case: extract tests, call oracle, update state."""
    case = state["case"]

    # Extract tests ordered this turn
    tests_this_turn = await parse_tests_from_response(session, sem, parsed["raw_response"])

    # Add new tests to the cumulative list (deduped)
    done_lower = {t.lower().strip() for t in state["done_tests"]}
    for t in tests_this_turn:
        if t.lower().strip() not in done_lower:
            state["done_tests"].append(t)
            done_lower.add(t.lower().strip())

    # Call oracle to get test results (skip if model is already done)
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
                    + parsed["raw_response"]
                )},
            ],
            max_tokens=800,
        )

    # Track unavailable tests from oracle response
    if oracle_text:
        new_unavail = _extract_unavailable_tests(oracle_text)
        unavail_lower = {t.lower().strip() for t in state["unavailable_tests"]}
        for t in new_unavail:
            if t.lower().strip() not in unavail_lower:
                state["unavailable_tests"].append(t)
                unavail_lower.add(t.lower().strip())

    # Update state
    state["current_oracle"] = oracle_text
    state["recent_turns"].append(
        build_turn_detail(turn, parsed, oracle_text, tests_this_turn)
    )
    prompt_text = "\n\n".join(
        f"[{m['role']}]\n{m['content']}" for m in messages
    )

    state["turn_logs"].append({
        "case_id":         case["case_id"],
        "turn":            turn,
        "prompt":          prompt_text,
        "raw_response":    parsed["raw_response"],
        "top1_ddx":        parsed["top1_ddx"],
        "conclusion":      parsed["conclusion"],
        "diag_status":     parsed["diag_status"],
        "model_done":      parsed["model_done"],
        "rule_done":       parsed["rule_done"],
        "tests_parsed":    ", ".join(tests_this_turn),
        "oracle_response": oracle_text,
        "is_done":         parsed["is_done"],
    })

    if parsed["is_done"]:
        state["done"] = True


async def _evaluate_case(state, session, sem):
    """Score one case: test recall/precision + diagnosis accuracy."""
    case     = state["case"]
    gt_tests = case["gt_tests"]

    all_responses = "\n\n".join(log["raw_response"] for log in state["turn_logs"] if log["raw_response"])
    last_response = next(
        (log["raw_response"] for log in reversed(state["turn_logs"]) if log["raw_response"]), ""
    )

    match    = await gpt_match_tests(session, sem, all_responses, gt_tests)
    ddx_eval = await gpt_eval_diagnosis(session, sem, last_response, case["gt_diagnosis"])

    tp = len(match["gt_covered"])
    fp = len(match["pred_unused"])
    fn = len(match["gt_uncovered"])
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / len(gt_tests) if gt_tests else 0.0

    return {
        "case_result": {
            "case_id":        case["case_id"],
            "parse_valid":    state["last_parse_valid"],
            "gt_diagnosis":   case["gt_diagnosis"],
            "total_gt_tests": len(gt_tests),
            "turns_used":     len(state["turn_logs"]),
            "ddx_top1":       state["last_top1_ddx"],
            "conclusion":     state["last_conclusion"],
            "dx_correct":     ddx_eval.get("match", False),
            "tp": tp, "fp": fp, "fn": fn,
            "precision":      round(precision, 4),
            "recall":         round(recall, 4),
        },
        "turn_logs": state["turn_logs"],
    }


# ---------------------------------------------------------------------------
# Results aggregation & saving
# ---------------------------------------------------------------------------

def aggregate_and_save(eval_results, output_xlsx):
    """Compute summary statistics and write results to Excel."""
    all_results = [r["case_result"] for r in eval_results]
    all_logs    = [log for r in eval_results for log in r["turn_logs"]]

    df_all     = pd.DataFrame(all_results).sort_values("case_id").reset_index(drop=True)
    df_logs    = pd.DataFrame(all_logs)

    # Split valid vs invalid cases; only valid ones count toward metrics
    df_valid   = df_all[df_all["parse_valid"] == True].reset_index(drop=True)
    df_invalid = df_all[df_all["parse_valid"] == False].reset_index(drop=True)
    n_invalid  = len(df_invalid)

    if n_invalid > 0:
        print(f"\n[WARNING] {n_invalid} case(s) had unparseable final responses and are excluded from metrics:")
        for _, row in df_invalid.iterrows():
            print(f"  - {row['case_id']}")

    if len(df_valid) > 0:
        precision_mean = round(float(np.mean(df_valid["precision"])), 4)
        precision_std  = round(float(np.std(df_valid["precision"])),  4)
        recall_mean    = round(float(np.mean(df_valid["recall"])),    4)
        recall_std     = round(float(np.std(df_valid["recall"])),     4)
        dx_accuracy    = round(float(np.mean(df_valid["dx_correct"].astype(int))), 4)
        avg_turns      = round(float(np.mean(df_valid["turns_used"])), 2)
    else:
        precision_mean = precision_std = recall_mean = recall_std = dx_accuracy = avg_turns = 0.0

    df_summary = pd.DataFrame([
        {"metric": "Precision_mean",  "value": precision_mean},
        {"metric": "Precision_std",   "value": precision_std},
        {"metric": "Recall_mean",     "value": recall_mean},
        {"metric": "Recall_std",      "value": recall_std},
        {"metric": "Dx_accuracy",     "value": dx_accuracy},
        {"metric": "Avg_turns_used",  "value": avg_turns},
        {"metric": "Valid_cases",     "value": len(df_valid)},
        {"metric": "Invalid_cases",   "value": n_invalid},
        {"metric": "Total_cases",     "value": len(df_all)},
    ])

    print("\n" + "=" * 50)
    print(df_summary.to_string(index=False))
    print("=" * 50)

    os.makedirs(os.path.dirname(output_xlsx), exist_ok=True)
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        df_logs.to_excel(writer, sheet_name="Turn Details", index=False)
        df_all.to_excel( writer, sheet_name="Case Results", index=False)
        df_summary.to_excel(writer, sheet_name="Summary",   index=False)

    print(f"Saved: {output_xlsx}")
    return df_all, df_summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Run multi-turn diagnostic inference with GPT-5.4."
    )
    parser.add_argument(
        "--reasoning_effort",
        choices=["high", "medium"],
        default="medium",
        help="GPT-5.4 reasoning effort level (default: medium)",
    )
    parser.add_argument(
        "-n", "--num_cases",
        type=int, default=None,
        help="Only run the first N cases (default: all)",
    )
    args = parser.parse_args()

    reasoning_effort = args.reasoning_effort
    output_xlsx = os.path.join(
        os.path.dirname(__file__),
        f"inference_results_gpt54_{reasoning_effort}.xlsx",
    )
    print(f"Model: {DIAGNOSTICIAN_MODEL}  reasoning_effort: {reasoning_effort}")

    cases = load_cases_by_ids(JSON_FOLDER, CASE_ID_FILE)
    if args.num_cases is not None:
        cases = cases[:args.num_cases]
        print(f"Running on first {len(cases)} case(s) only")

    diag_sem   = asyncio.Semaphore(DIAGNOSTICIAN_CONCURRENCY)
    oracle_sem = asyncio.Semaphore(ORACLE_CONCURRENCY)
    connector  = aiohttp.TCPConnector(
        limit=DIAGNOSTICIAN_CONCURRENCY + ORACLE_CONCURRENCY + 5
    )

    async with aiohttp.ClientSession(connector=connector, trust_env=True) as session:
        eval_results = await run_pipeline(
            cases, reasoning_effort, session, diag_sem, oracle_sem
        )

    return aggregate_and_save(eval_results, output_xlsx)


if __name__ == "__main__":
    asyncio.run(main())