"""
run_all.py — Generate enriched ontologies for all LLM × schema combinations.

Runs the RIGOR enrichment pipeline for:
  - 3 LLMs  : Claude, Mistral, DeepSeek
  - 2 schemas: Real-world liver cancer DB, eICU-CRD

Total: 6 runs.  Results are saved to output/<schema>/<model>/enriched_ontology.owl

Usage:
    export OPENROUTER_API_KEY=your_key_here
    python run_all.py
"""

import os
import sys
import time
from app import run_semantic_enrichment

# =========================================================
# BASE PATH
# =========================================================

BASE = "YOUR_BASE_PATH"

# =========================================================
# SCHEMAS
# Two schemas: real-world liver cancer registry and eICU-CRD
# Update eicu_schema path to wherever you have stored it.
# =========================================================

SCHEMAS = {
   "real_world": {
        "schema_path":   f"{BASE}/sql_schema/schema_rd.json",
        "core_owl_path": f"{BASE}/core_ontology/core.owl",
    },
    "eicu_crd": {
        "schema_path":   f"{BASE}/sql_schema/schema_icu.json",
        "core_owl_path": f"{BASE}/core_ontology/core_icu.owl",
    },
}

# ================================
# 
# =========================
# SHARED PATHS  (same for all runs)
# =========================================================

DOCS_PATH      = f"{BASE}/docs"
ONTOLOGY_PATH  = f"{BASE}/ontologies"

# =========================================================
# LLMs  (OpenRouter model strings)
# =========================================================

MODELS = {
    "claude":   "anthropic/claude-opus-4-6",
    "mistral":  "mistralai/mistral-small-24b-instruct-2501",
    "deepseek": "deepseek/deepseek-chat",
}

# =========================================================
# RUNNER
# =========================================================

def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set.")
        print("  Mac/Linux : export OPENROUTER_API_KEY=your_key")
        print("  Windows   : set OPENROUTER_API_KEY=your_key")
        sys.exit(1)

    total   = len(SCHEMAS) * len(MODELS)
    current = 0
    results = []

    for schema_name, schema_cfg in SCHEMAS.items():
        for model_name, model_id in MODELS.items():
            current += 1
            print("\n" + "=" * 65)
            print(f"RUN {current}/{total}  |  Schema: {schema_name}  |  Model: {model_name}")
            print("=" * 65)

            output_dir = f"{BASE}/output/{schema_name}/{model_name}"
            os.makedirs(output_dir, exist_ok=True)

            output_path         = f"{output_dir}/enriched_ontology.owl"
            direct_mappings_dir = f"{output_dir}/direct_mappings"

            # Patch the model inside app.py's OpenRouterLLM
            # by temporarily setting an env var the pipeline reads
            os.environ["RIGOR_MODEL"] = model_id

            start = time.time()
            try:
                run_semantic_enrichment(
                    schema_path         = schema_cfg["schema_path"],
                    docs_path           = DOCS_PATH,
                    ontology_path       = ONTOLOGY_PATH,
                    core_owl_path       = schema_cfg["core_owl_path"],
                    output_path         = output_path,
                    direct_mappings_dir = direct_mappings_dir,
                    model               = model_id,
                )
                elapsed = time.time() - start
                results.append({
                    "schema": schema_name,
                    "model":  model_name,
                    "status": "OK",
                    "output": output_path,
                    "time":   f"{elapsed/60:.1f} min",
                })
                print(f"\n  Completed in {elapsed/60:.1f} min -> {output_path}")

            except Exception as e:
                elapsed = time.time() - start
                results.append({
                    "schema": schema_name,
                    "model":  model_name,
                    "status": f"FAILED: {e}",
                    "output": None,
                    "time":   f"{elapsed/60:.1f} min",
                })
                print(f"\n  FAILED after {elapsed/60:.1f} min: {e}")
                import traceback
                traceback.print_exc()

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    for r in results:
        status = "✓" if r["status"] == "OK" else "✗"
        print(f"  {status}  {r['schema']:12}  {r['model']:10}  {r['time']:8}  {r['status']}")
    print()


if __name__ == "__main__":
    main()
