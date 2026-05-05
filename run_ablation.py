"""
run_ablation.py — Generate ablated ontology variants for RIGOR ablation study.

This script runs Chinook ablations using the current RIGOR app.py.

Important design choice:
    Deterministic Direct Mapping is ALWAYS retained and merged by app.py.
    Ablations hide selected context sources from the Gen-LLM prompt, but do not
    remove the direct-mapping backbone. This keeps schema coverage constant and
    isolates the contribution of contextual inputs.

Ablation variants:
  no_rag                       — direct mapping retained; no schema/core context, no external ontologies, no documents
  only_schema_context          — direct mapping retained; schema/core context only
  only_external_ontologies     — direct mapping retained; external ontology hints only
  only_relevant_documents      — direct mapping retained; document hints only
  without_schema_context       — direct mapping retained; external ontologies + documents
  without_external_ontologies  — direct mapping retained; schema/core context + documents
  without_relevant_documents   — direct mapping retained; schema/core context + external ontologies

Output:
  ablation/chinook/<variant>/enriched_ontology.owl

Usage:
    export OPENROUTER_API_KEY=your_key_here
    python run_ablation.py
    python run_ablation.py --variant no_rag
    python run_ablation.py --force
"""

import os

# Same environment safeguards used in the working RIGOR script.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import time
import argparse
import traceback
from unittest.mock import patch


# =========================================================
# CONFIGURATION
# =========================================================

BASE = "/Users/nadeen/Downloads/RIGORFrameworksemanticweb/RIGOR_Framework"

SCHEMAS = {
    "chinook": {
        "schema_path": f"{BASE}/sql_schema/schema_chinook.json",
        "core_owl_path": f"{BASE}/core_ontology/core.owl",
    },
}

DOCS_PATH = f"{BASE}/documents_chinook"
ONTOLOGY_PATH = f"{BASE}/external_ontologies_chinook"
OUTPUT_BASE = f"{BASE}/ablation_chinook"
MODEL = "anthropic/claude-opus-4-6"


# =========================================================
# ABLATION VARIANTS
# =========================================================

VARIANTS = {
    "no_rag": {
        "use_schema": False,
        "use_ontologies": False,
        "use_docs": False,
        "description": "Direct mapping retained; no schema/core context, no documents, no external ontology hints",
    },
    "only_schema_context": {
        "use_schema": True,
        "use_ontologies": False,
        "use_docs": False,
        "description": "Schema/core context only",
    },
    "only_external_ontologies": {
        "use_schema": False,
        "use_ontologies": True,
        "use_docs": False,
        "description": "External ontology hints only",
    },
    "only_relevant_documents": {
        "use_schema": False,
        "use_ontologies": False,
        "use_docs": True,
        "description": "Retrieved document context only",
    },
    "without_schema_context": {
        "use_schema": False,
        "use_ontologies": True,
        "use_docs": True,
        "description": "Documents + external ontology hints; no schema/core context",
    },
    "without_external_ontologies": {
        "use_schema": True,
        "use_ontologies": False,
        "use_docs": True,
        "description": "Schema/core context + documents; no external ontology hints",
    },
    "without_relevant_documents": {
        "use_schema": True,
        "use_ontologies": True,
        "use_docs": False,
        "description": "Schema/core context + external ontology hints; no documents",
    },
}


# =========================================================
# ABLATED PIPELINE RUNNER
# =========================================================

def run_ablation_variant(
    variant_name: str,
    variant_cfg: dict,
    schema_name: str,
    schema_cfg: dict,
    output_path: str,
    dm_dir: str,
):
    """
    Run app.run_semantic_enrichment() while hiding selected context sources
    from the Gen-LLM prompt.

    Direct mapping is not disabled. It is still generated and merged by app.py
    before enrichment, preserving schema coverage across all variants.
    """
    import app

    use_schema = variant_cfg["use_schema"]
    use_ontologies = variant_cfg["use_ontologies"]
    use_docs = variant_cfg["use_docs"]

    print(f"\n  Schema/core context : {'ON ' if use_schema else 'OFF'}")
    print(f"  External ontologies : {'ON ' if use_ontologies else 'OFF'}")
    print(f"  Documents           : {'ON ' if use_docs else 'OFF'}")
    print("  Direct Mapping      : ON  (always retained for fair coverage)")

    # -----------------------------------------------------
    # Patch 1: document loading
    # -----------------------------------------------------
    original_load_docs = app.load_text_documents

    def patched_load_docs(docs_path):
        if use_docs:
            return original_load_docs(docs_path)
        print("   [ABLATION] Documents disabled — no document FAISS index")
        return {}

    # -----------------------------------------------------
    # Patch 2: external ontology loading
    # -----------------------------------------------------
    original_load_ontos = app.load_external_ontologies

    def patched_load_ontos(ontology_path):
        if use_ontologies:
            return original_load_ontos(ontology_path)
        print("   [ABLATION] External ontologies disabled — no ontology FAISS index")
        return []

    # -----------------------------------------------------
    # Patch 3: growing core ontology retrieval context
    # -----------------------------------------------------
    original_build_core_chunks = app.build_core_ontology_chunks

    def patched_build_core_chunks(core):
        if use_schema:
            return original_build_core_chunks(core)
        print("   [ABLATION] Growing core ontology context disabled")
        return []

    # -----------------------------------------------------
    # Patch 4: Gen-LLM prompt construction
    # -----------------------------------------------------
    original_build_prompt = app.OntologyLLM._build_prompt

    def patched_build_prompt(
        self,
        table_name,
        direct_mapping,
        schema_str,
        foreign_keys_str,
        documents,
        core_context,
        external_context,
        correction_hint,
    ):
        if not use_schema:
            direct_mapping = (
                "# Direct Mapping is retained internally for deterministic coverage, "
                "but hidden from the Gen-LLM in this ablation variant."
            )
            schema_str = "{}"
            foreign_keys_str = "None"
            core_context = "None"

        if not use_docs:
            documents = "None"

        if not use_ontologies:
            external_context = "None"

        return original_build_prompt(
            self,
            table_name,
            direct_mapping,
            schema_str,
            foreign_keys_str,
            documents,
            core_context,
            external_context,
            correction_hint,
        )

    # -----------------------------------------------------
    # Run app.py with patches
    # -----------------------------------------------------
    with patch.object(app, "load_text_documents", patched_load_docs), \
         patch.object(app, "load_external_ontologies", patched_load_ontos), \
         patch.object(app, "build_core_ontology_chunks", patched_build_core_chunks), \
         patch.object(app.OntologyLLM, "_build_prompt", patched_build_prompt):

        app.run_semantic_enrichment(
            schema_path=schema_cfg["schema_path"],
            docs_path=DOCS_PATH,
            ontology_path=ONTOLOGY_PATH,
            core_owl_path=schema_cfg["core_owl_path"],
            output_path=output_path,
            direct_mappings_dir=dm_dir,
            model=MODEL,
            use_llm=True,
        )


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate Chinook ablated RIGOR ontology variants using Mistral"
    )
    parser.add_argument(
        "--schema",
        choices=list(SCHEMAS.keys()),
        help="Run only this schema. Default: all.",
    )
    parser.add_argument(
        "--variant",
        choices=list(VARIANTS.keys()),
        help="Run only this variant. Default: all.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate output even if enriched_ontology.owl already exists.",
    )

    args, _ = parser.parse_known_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set.")
        print("  Mac/Linux: export OPENROUTER_API_KEY=your_key")
        sys.exit(1)

    schema_keys = [args.schema] if args.schema else list(SCHEMAS.keys())
    variant_keys = [args.variant] if args.variant else list(VARIANTS.keys())

    total = len(schema_keys) * len(variant_keys)
    current = 0
    results = []

    print("=" * 70)
    print("RIGOR ABLATION STUDY — Chinook")
    print(f"Schemas : {schema_keys}")
    print(f"Variants: {variant_keys}")
    print(f"Model   : {MODEL}")
    print(f"Total   : {total} runs")
    print("=" * 70)

    for schema_name in schema_keys:
        schema_cfg = SCHEMAS[schema_name]

        for variant_name in variant_keys:
            current += 1
            variant_cfg = VARIANTS[variant_name]

            print(f"\n{'=' * 70}")
            print(f"RUN {current}/{total}")
            print(f"Schema : {schema_name}")
            print(f"Variant: {variant_name}")
            print(f"Meaning: {variant_cfg['description']}")
            print(f"{'=' * 70}")

            output_dir = os.path.join(OUTPUT_BASE, schema_name, variant_name)
            output_path = os.path.join(output_dir, "enriched_ontology.owl")
            dm_dir = os.path.join(output_dir, "direct_mappings")

            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(dm_dir, exist_ok=True)

            if os.path.exists(output_path) and not args.force:
                print(f"  Already exists — skipping: {output_path}")
                results.append({
                    "schema": schema_name,
                    "variant": variant_name,
                    "status": "SKIPPED",
                    "output": output_path,
                    "time": "—",
                })
                continue

            start = time.time()

            try:
                run_ablation_variant(
                    variant_name=variant_name,
                    variant_cfg=variant_cfg,
                    schema_name=schema_name,
                    schema_cfg=schema_cfg,
                    output_path=output_path,
                    dm_dir=dm_dir,
                )

                elapsed = time.time() - start
                results.append({
                    "schema": schema_name,
                    "variant": variant_name,
                    "status": "OK",
                    "output": output_path,
                    "time": f"{elapsed / 60:.1f} min",
                })

                print(f"\n  Done in {elapsed / 60:.1f} min -> {output_path}")

            except Exception as e:
                elapsed = time.time() - start
                results.append({
                    "schema": schema_name,
                    "variant": variant_name,
                    "status": f"FAILED: {e}",
                    "output": None,
                    "time": f"{elapsed / 60:.1f} min",
                })

                print(f"\n  FAILED after {elapsed / 60:.1f} min: {e}")
                traceback.print_exc()

    print("\n" + "=" * 70)
    print("ABLATION RUN SUMMARY")
    print("=" * 70)
    print(f"  {'Schema':<12} {'Variant':<32} {'Time':>9}  Status")
    print(f"  {'-' * 12} {'-' * 32} {'-' * 9}  {'-' * 20}")

    for r in results:
        icon = "✓" if r["status"] in {"OK", "SKIPPED"} else "✗"
        print(f"  {icon} {r['schema']:<12} {r['variant']:<32} {r['time']:>9}  {r['status']}")

    print("\nNext step: run your evaluation script on these ontology paths:")
    for schema_name in schema_keys:
        for variant_name in variant_keys:
            path = os.path.join(OUTPUT_BASE, schema_name, variant_name, "enriched_ontology.owl")
            print(f"  [{schema_name}][{variant_name}] = {path}")


if __name__ == "__main__":
    main()