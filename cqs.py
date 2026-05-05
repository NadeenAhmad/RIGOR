"""
cqs.py — Competency Question Generator for RIGOR Evaluation

Generates 5 CQs per table for each database schema using
Mistral Small 24B Instruct via OpenRouter.

CQs are saved as individual .txt files in the format expected
by eval.py:
    <question text>

    <answer explanation>

One file per table: <save_dir>/<db_name>/<table_name>_cqs.txt

Usage:
    export OPENROUTER_API_KEY="your_key_here"
    python3 cqs.py
"""

import os
import re
import json
import time
from typing import List, Tuple, Dict

from openai import OpenAI

# =========================================================
# CONFIGURATION
# =========================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "mistralai/mistral-small-24b-instruct-2501"

BASE_PATH = "YOUR_BASE_PATH"

SCHEMAS = {
  #  "real_world": f"{BASE_PATH}/sql_schema/schema_rd.json",
  #  "eicu_crd":   f"{BASE_PATH}/sql_schema/schema_icu.json",
           "chinook": f"{BASE_PATH}/sql_schema/schema_chinook.json",
}

CQS_OUTPUT_DIR = f"{BASE_PATH}/cqs"   # eval.py reads from here

# Optional headers recommended by OpenRouter
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "http://localhost")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "RIGOR-CQ-Generator")

# =========================================================
# CLIENT SETUP
# =========================================================

def get_openrouter_client() -> OpenAI:
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. "
            "Run: export OPENROUTER_API_KEY='your_key_here'"
        )

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

# =========================================================
# SCHEMA LOADING
# =========================================================

def load_schema(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_columns(table_value: dict) -> dict:
    """
    Handle both schema formats:
      Format A: {col: type, ...}
      Format B: {"columns": {col: type, ...}, "foreign_keys": [...]}
    """
    if isinstance(table_value, dict) and "columns" in table_value:
        return table_value["columns"]
    return table_value

# =========================================================
# CQ GENERATION
# =========================================================

def build_prompt(table_name: str, columns: dict) -> str:
    col_lines = "\n".join(f"  - {col}: {dtype}" for col, dtype in columns.items())
    return f"""Given the SQL table schema for table '{table_name}':
{col_lines}

Generate 5 competency questions (CQs) that this table's ontology should answer.
For each question, also provide a short answer explaining how the ontology would
answer it using the table schema.

Use exactly this format:

1. [Question]
Answer: [Explanation]

2. [Question]
Answer: [Explanation]

3. [Question]
Answer: [Explanation]

4. [Question]
Answer: [Explanation]

5. [Question]
Answer: [Explanation]

Output only these 5 question-answer pairs and nothing else."""


def call_openrouter(
    client: OpenAI,
    prompt: str,
    model: str = OPENROUTER_MODEL,
    temperature: float = 0.5,
    max_tokens: int = 900,
    retries: int = 3,
) -> str:
    """
    Call OpenRouter chat completions API using the OpenAI SDK.
    """
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers={
                    "HTTP-Referer": OPENROUTER_SITE_URL,
                    "X-Title": OPENROUTER_APP_NAME,
                },
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You generate competency questions from SQL schemas. "
                            "Be precise and follow the requested output format exactly."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Model returned empty content.")

            return content

        except Exception as e:
            last_error = e
            if attempt < retries:
                sleep_seconds = 2 ** (attempt - 1)
                print(f"    Retry {attempt}/{retries} after error: {e}")
                time.sleep(sleep_seconds)
            else:
                raise last_error


def parse_cqs(raw_output: str, prompt: str) -> List[Tuple[str, str]]:
    """
    Parse the LLM output into a list of (question, answer) tuples.
    Strips the prompt from the output if it was echoed back,
    then extracts numbered Q/A pairs.
    """
    text = raw_output

    if prompt.strip() in text:
        text = text[text.index(prompt.strip()) + len(prompt.strip()):]

    pattern = r"(?:^|\n)\**\s*(\d)\s*[.)]\s*\**(.+?)(?:\n\s*[-•]?\s*\**[Aa]nswer\**:?\s*)(.+?)(?=\n\**\s*\d\s*[.)]|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)

    cqs = []
    for _, question, answer in matches:
        q = question.strip().strip("*").strip()
        a = answer.strip().strip("*").strip()
        if q and a:
            cqs.append((q, a))

    if not cqs:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        i = 0
        while i < len(lines):
            if re.match(r"^\**\d[.)]\s+", lines[i]):
                q_text = re.sub(r"^\**\d[.)]\s+", "", lines[i]).strip("*").strip()
                a_text = ""

                if i + 1 < len(lines) and re.match(r"^[-•]?\s*\**[Aa]nswer", lines[i + 1]):
                    a_text = re.sub(r"^[-•]?\s*\**[Aa]nswer\**:?\s*", "", lines[i + 1]).strip()
                    i += 2
                else:
                    i += 1

                if q_text:
                    cqs.append((q_text, a_text or "See schema."))
            else:
                i += 1

    return cqs[:5]


def save_cqs(cqs: List[Tuple[str, str]], table_name: str, save_dir: str) -> str:
    """
    Save CQs in the format eval.py expects:
      question text\n\nanswer text
    One CQ per file section, separated by blank lines.
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{table_name}_cqs.txt")

    with open(file_path, "w", encoding="utf-8") as f:
        for i, (question, answer) in enumerate(cqs):
            f.write(question.strip())
            f.write("\n\n")
            f.write(answer.strip())
            if i < len(cqs) - 1:
                f.write("\n\n---\n\n")

    return file_path


def generate_cqs_for_table(
    table_name: str,
    table_value: dict,
    client: OpenAI,
    save_dir: str,
) -> List[Tuple[str, str]]:
    columns = get_columns(table_value)
    if not columns:
        print(f"    Skipping {table_name} — no columns found")
        return []

    prompt = build_prompt(table_name, columns)

    try:
        raw_text = call_openrouter(
            client=client,
            prompt=prompt,
            model=OPENROUTER_MODEL,
            temperature=0.5,
            max_tokens=900,
        )

        cqs = parse_cqs(raw_text, prompt)

        if not cqs:
            print(f"    Warning: no CQs parsed for {table_name} — saving raw output")
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, f"{table_name}_raw.txt"), "w", encoding="utf-8") as f:
                f.write(raw_text)
            return []

        file_path = save_cqs(cqs, table_name, save_dir)
        print(f"    Saved {len(cqs)} CQs -> {file_path}")
        return cqs

    except Exception as e:
        print(f"    Error generating CQs for {table_name}: {e}")
        return []

# =========================================================
# MAIN
# =========================================================

def main():
    client = get_openrouter_client()

    print(f"Using OpenRouter model: {OPENROUTER_MODEL}")
    print("Client ready.\n")

    for db_name, schema_path in SCHEMAS.items():
        print(f"{'=' * 55}")
        print(f"Database: {db_name}")
        print(f"{'=' * 55}")

        if not os.path.exists(schema_path):
            print(f"  Schema not found: {schema_path} — skipping")
            continue

        schema = load_schema(schema_path)
        save_dir = os.path.join(CQS_OUTPUT_DIR, db_name)
        total = len(schema)

        print(f"  {total} tables found")

        for idx, (table_name, table_value) in enumerate(schema.items(), start=1):
            print(f"\n  [{idx}/{total}] {table_name}")

            existing = os.path.join(save_dir, f"{table_name}_cqs.txt")
            if os.path.exists(existing):
                print(f"    Already exists — skipping")
                continue

            generate_cqs_for_table(table_name, table_value, client, save_dir)

        print(f"\n  Done: {db_name} CQs saved to {save_dir}")

    print("\nAll CQs generated.")


if __name__ == "__main__":
    main()