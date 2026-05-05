"""
baseline.py — RIGOR Baseline Ontology Generation

Generates ontologies using ONLY the database schema (no RAG, no external
ontologies, no documents, no Judge-LLM). This is the lower bound comparison.

Runs for both schemas x 3 LLMs = 6 outputs.

Usage:
    python baseline.py
"""

import os
import json
import re
import requests
from rdflib import Graph, RDF, OWL, RDFS, XSD, Namespace, URIRef, Literal

# =========================================================
# CONFIGURATION
# =========================================================

BASE_PATH = "YOUR_BASE_PATH"

SCHEMAS = {
    #"real_world": f"{BASE_PATH}/sql_schema/schema_rd.json",
    #"eicu_crd":   f"{BASE_PATH}/sql_schema/schema_icu.json",
         "chinook": f"{BASE_PATH}/sql_schema/schema_chinook.json",
}

MODELS = {
    "claude":   "anthropic/claude-opus-4-6",
    "mistral":  "mistralai/mistral-small-24b-instruct-2501",
    "deepseek": "deepseek/deepseek-chat",
}

OUTPUT_BASE = f"{BASE_PATH}/output/baseline"
ONTOLOGY_IRI = "http://example.org/ontology"

# =========================================================
# SCHEMA LOADING
# =========================================================

def load_schema(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_columns(table_value):
    """Handle both flat and nested schema formats."""
    if isinstance(table_value, dict) and "columns" in table_value:
        return table_value["columns"]
    return table_value if isinstance(table_value, dict) else {}

# =========================================================
# PROMPT
# =========================================================

def build_prompt(table_name, columns):
    """
    Baseline prompt: schema only, no external context.
    Follows Figure 3(c) in the paper — maps table schema to OWL Manchester Syntax.
    """
    col_lines = "\n".join(f"  - {col}: {dtype}" for col, dtype in columns.items())
    return f"""Generate an OWL 2 ontology fragment in Manchester Syntax for the database table '{table_name}'.

Table schema:
{col_lines}

Instructions:
- Create an OWL Class for the table
- Create a DataProperty for each non-FK column with correct domain and range
- Create an ObjectProperty for each FK column
- Use xsd: datatypes (xsd:string, xsd:integer, xsd:boolean, xsd:float, xsd:dateTime)
- Every property MUST have exactly one Domain and one Range
- Output ONLY valid Manchester Syntax, nothing else

[OUTPUT]"""

# =========================================================
# OUTPUT PARSER — reuse from app.py logic
# =========================================================

def parse_manchester_to_rdf(llm_output, base_ns):
    """
    Parse Manchester Syntax LLM output into an rdflib Graph.
    Returns Graph with parsed triples.
    """
    g = Graph()
    g.bind("owl", OWL)
    g.bind("xsd", XSD)
    g.bind("rdfs", RDFS)
    g.bind("", base_ns)

    XSD_MAP = {
        "string": XSD.string, "integer": XSD.integer, "float": XSD.float,
        "boolean": XSD.boolean, "datetime": XSD.dateTime, "date": XSD.date,
        "decimal": XSD.decimal,
    }

    blocks, cur = [], []
    for line in llm_output.split("\n"):
        s = line.strip()
        if not s or s.startswith("```"):
            continue
        if any(s.lower().startswith(k) for k in ["class:", "dataproperty:", "objectproperty:"]):
            if cur:
                blocks.append(cur)
            cur = [s]
        else:
            cur.append(s)
    if cur:
        blocks.append(cur)

    for block in blocks:
        hdr, rest = block[0], block[1:]

        def fv(kw):
            for ln in rest:
                if ln.lower().strip().startswith(kw):
                    parts = ln.strip().split(None, 1)
                    return parts[1].strip() if len(parts) > 1 else None
            return None

        try:
            low = hdr.lower()
            if low.startswith("class:"):
                name = hdr.split(":", 1)[1].strip()
                g.add((base_ns[name], RDF.type, OWL.Class))

            elif low.startswith("dataproperty:"):
                name = hdr.split(":", 1)[1].strip()
                dom, rng = fv("domain"), fv("range")
                if dom and rng:
                    uri = base_ns[name]
                    g.add((uri, RDF.type, OWL.DatatypeProperty))
                    g.add((uri, RDFS.domain, base_ns[dom]))
                    rng_clean = rng.lower().replace("xsd:", "")
                    g.add((uri, RDFS.range, XSD_MAP.get(rng_clean, XSD.string)))

            elif low.startswith("objectproperty:"):
                name = hdr.split(":", 1)[1].strip()
                dom, rng = fv("domain"), fv("range")
                if dom and rng:
                    uri = base_ns[name]
                    g.add((uri, RDF.type, OWL.ObjectProperty))
                    g.add((uri, RDFS.domain, base_ns[dom]))
                    g.add((uri, RDFS.range, base_ns[rng]))
        except Exception:
            continue

    return g

# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 60)
    print("Baseline Ontology Generation")
    print("=" * 60)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set.")
        print("  export OPENROUTER_API_KEY=your_key")
        return

    def call_llm(model_name, prompt):
        """Call OpenRouter API and return generated text."""
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                timeout=120,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model":       model_name,
                    "messages":    [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens":  4000,
                },
            )
            if r.status_code != 200:
                print(f"  API error {r.status_code}: {r.text[:200]}")
                return ""
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  API call failed: {e}")
            return ""

    for model_label, model_name in MODELS.items():
        print(f"\nModel: {model_name}")

        for db_name, schema_path in SCHEMAS.items():
            print(f"\n  Schema: {db_name}")

            if not os.path.exists(schema_path):
                print(f"  Schema not found: {schema_path} — skipping")
                continue

            schema    = load_schema(schema_path)
            base_ns   = Namespace(f"{ONTOLOGY_IRI}#")
            core_graph = Graph()
            core_graph.bind("", base_ns)
            core_graph.bind("owl", OWL)
            core_graph.bind("xsd", XSD)
            core_graph.bind("rdfs", RDFS)

            out_dir = os.path.join(OUTPUT_BASE, db_name, model_label)
            os.makedirs(out_dir, exist_ok=True)
            raw_dir = os.path.join(out_dir, "raw")
            os.makedirs(raw_dir, exist_ok=True)

            total = len(schema)
            for idx, (table_name, table_value) in enumerate(schema.items(), 1):
                print(f"    [{idx}/{total}] {table_name}")

                columns = get_columns(table_value)
                if not columns:
                    print(f"    No columns — skipping")
                    continue

                prompt = build_prompt(table_name, columns)

                generated = call_llm(model_name, prompt).strip()
                if not generated:
                    print(f"    LLM returned empty response")

                # Save raw output
                raw_path = os.path.join(raw_dir, f"{table_name}_raw.txt")
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(generated)

                # Parse and merge into core graph
                if generated:
                    delta = parse_manchester_to_rdf(generated, base_ns)
                    core_graph += delta
                    n_cls = len(list(delta.subjects(RDF.type, OWL.Class)))
                    n_dp  = len(list(delta.subjects(RDF.type, OWL.DatatypeProperty)))
                    n_op  = len(list(delta.subjects(RDF.type, OWL.ObjectProperty)))
                    print(f"    Parsed: {n_cls} classes, {n_dp} dataprops, {n_op} objectprops")

            # Save merged ontology
            out_path = os.path.join(out_dir, "baseline_ontology.owl")
            core_graph.serialize(out_path, format="xml")
            print(f"  Saved: {out_path} ({len(core_graph)} triples)")

    print("\nBaseline generation complete.")


if __name__ == "__main__":
    main()
