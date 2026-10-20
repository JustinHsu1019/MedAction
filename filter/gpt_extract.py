"""
gpt_extract.py — Use GPT to extract DDX / PRIMARY_ACTIONS / PIVOT from raw_response.
"""
import json
import requests
from config import OPENAI_API_KEY, GPT_MODEL

EXTRACTION_SYSTEM_PROMPT = """You are a medical NLP assistant.
Extract structured information from a doctor's diagnostic reasoning text.

Always respond in valid JSON with exactly these keys:
{
  "PRIMARY_ACTIONS": [],
  "DDX": [],
  "PIVOT": ""
}

Rules:
- PRIMARY_ACTIONS: specific test names only (e.g. "Chest X-ray", "CBC")
- DDX: disease/condition names only, ranked, no descriptions
- PIVOT: "Disease A vs Disease B", greatest clinical contrast pair. Empty string if DDX < 2.
- If Additional Information Required says "Not required.", PRIMARY_ACTIONS must be []
- If Conclusion shows single converged diagnosis, DDX must be [] and PIVOT must be ""
"""

EMPTY = {"PRIMARY_ACTIONS": [], "DDX": [], "PIVOT": ""}


def gpt_extract_decision(answer_text: str) -> dict:
    if not answer_text or str(answer_text).strip() in ("", "nan"):
        return EMPTY
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GPT_MODEL,
                "messages": [
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user",   "content": str(answer_text)},
                ],
                "temperature": 0.0,
                "max_tokens": 1500,
                "response_format": {"type": "json_object"},
            },
            timeout=30,
        )
        resp.raise_for_status()
        result = json.loads(resp.json()["choices"][0]["message"]["content"])
        return {k: result.get(k, v) for k, v in EMPTY.items()}
    except Exception as e:
        print(f"    [warn] GPT extraction failed: {e}")
        return EMPTY
