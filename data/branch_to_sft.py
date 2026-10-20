"""
branch_to_sft.py

Convert branch trajectory logs into SFT JSONL format.
"""

import os
import json
import argparse


DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "branch_data"))

SYSTEM_PROMPT = (
    "You are an experienced clinical physician performing a structured diagnostic workup."
)


def parse_prompt_to_messages(prompt_text):
    """Parse stored prompt back to [system, user] messages.

    The prompt is stored as:
        [system]
        <system content>

        [user]
        <user content>
    """
    marker = "\n\n[user]\n"
    if marker in prompt_text:
        user_content = prompt_text.split(marker, 1)[1]
    else:
        user_content = prompt_text
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content.strip()},
    ]


def main():
    parser = argparse.ArgumentParser(description="Convert branch data to SFT JSONL")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Branch data directory")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    args = parser.parse_args()

    print(f"Scanning branch data in {args.data_dir}...")
    records = []
    branch_count = 0

    for case_id in sorted(os.listdir(args.data_dir)):
        case_dir = os.path.join(args.data_dir, case_id)
        if not os.path.isdir(case_dir) or case_id.startswith("."):
            continue

        for branch_name in sorted(os.listdir(case_dir)):
            branch_dir = os.path.join(case_dir, branch_name)
            if not os.path.isdir(branch_dir) or not branch_name.startswith("branch_t"):
                continue

            logs_path = os.path.join(branch_dir, "turn_logs.json")
            if not os.path.exists(logs_path):
                continue

            with open(logs_path, encoding="utf-8") as f:
                result = json.load(f)

            # Combine inherited trunk turns + new branch turns = full trajectory
            inherited = result.get("inherited_turns", [])
            branch_turns = result.get("branch_turns", [])
            all_turns = inherited + branch_turns
            branch_count += 1

            for turn_data in all_turns:
                prompt_text = turn_data.get("prompt", "")
                response_text = turn_data.get("raw_response", "")
                if not prompt_text or not response_text:
                    continue

                messages = parse_prompt_to_messages(prompt_text)
                messages.append({"role": "assistant", "content": response_text})

                records.append({
                    "case_id": case_id,
                    "branch": branch_name,
                    "turn_number": turn_data.get("turn", 0),
                    "is_final": turn_data.get("is_done", False),
                    "messages": messages,
                })

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  {branch_count} branches → {len(records)} SFT records")
    print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
