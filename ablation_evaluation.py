"""
ablation.py — RIGOR Ablation Study Evaluator

Purpose:
  - Gold standard replaced with three independent reference sources:
      1. Database schema (table + column names)
      2. External ontology concepts (BioPortal)
      3. Documentation corpus (RAG documents)
  - Both exact lexical match AND semantic similarity match reported
    so Table 4 can show both scores side-by-side
  - All ablation variants evaluated in one run
  - No-RAG class F1 explained via schema-signal analysis
  - Results saved as CSV (Table 4 format) and detailed JSON

Ablation variants evaluated (must have corresponding OWL files in the ablation folder):
  no_rag
  only_schema_context
  only_external_ontologies
  only_relevant_documents
  without_schema_context
  without_external_ontologies
  without_relevant_documents
  full_rigor  (kept for reference / backward compatibility)
"""

import os
import json
import re
from typing import Dict, List, Set, Tuple, Optional

import pandas as pd
from owlready2 import get_ontology, Thing
from rdflib import Graph, RDF, OWL, RDFS, URIRef
from sentence_transformers import SentenceTransformer, util
import torch 
from docx import Document

#from RIGOR_Framework.run_all import BASE
# =========================================================
# CONFIGURATION
# =========================================================

BASE_PATH = "YOUR_BASE_PATH"

# Schemas
SCHEMAS = {
 #  "real_world": f"{BASE_PATH}/sql_schema/schema_rd.json",
  #  "eicu_crd":   f"{BASE_PATH}/sql_schema/schema_icu.json",
    "chinook": f"{BASE_PATH}/sql_schema/schema_chinook.json",
}

# External ontology folder (BioPortal OWL files)
EXTERNAL_ONTOLOGY_PATH = f"{BASE_PATH}/external_ontologies_chinook"

# Documentation corpus (plain text / docx converted to txt)
DOCS_PATH = f"{BASE_PATH}/documents_chinook"

# Ablation variant ontology files per database
# Keys must match the variant names in Table 4 of the paper
ABLATION_VARIANTS = {
 #   "real_world": {
  #      "no_rag":                      f"{BASE_PATH}/output/ablation/real_world/no_rag/enriched_ontology.owl",
   #     "only_schema_context":         f"{BASE_PATH}/output/ablation/real_world/only_schema_context/enriched_ontology.owl",
    #    "only_external_ontologies":    f"{BASE_PATH}/output/ablation/real_world/only_external_ontologies/enriched_ontology.owl",
     #   "only_relevant_documents":     f"{BASE_PATH}/output/ablation/real_world/only_relevant_documents/enriched_ontology.owl",
     #   "without_schema_context":      f"{BASE_PATH}/output/ablation/real_world/without_schema_context/enriched_ontology.owl",
     #   "without_external_ontologies": f"{BASE_PATH}/output/ablation/real_world/without_external_ontologies/enriched_ontology.owl",
     #   "without_relevant_documents":  f"{BASE_PATH}/output/ablation/real_world/without_relevant_documents/enriched_ontology.owl",
     #   "full_rigor": f"{BASE_PATH}/output/RIGOR/real_world/mistral/enriched_ontology.owl",
   # },
    # "eicu_crd": {
     #    "no_rag":                      f"{BASE_PATH}/output/ablation/eicu_crd/no_rag.owl",
      #   "only_schema_context":         f"{BASE_PATH}/output/ablation/eicu_crd/only_schema_context.owl",
       #  "only_external_ontologies":    f"{BASE_PATH}/output/ablation/eicu_crd/only_external_ontologies.owl",
       #  "only_relevant_documents":     f"{BASE_PATH}/output/ablation/eicu_crd/only_relevant_documents.owl",
       #  "without_schema_context":      f"{BASE_PATH}/output/ablation/eicu_crd/without_schema_context.owl",
       #  "without_external_ontologies": f"{BASE_PATH}/output/ablation/eicu_crd/without_external_ontologies.owl",
       #  "without_relevant_documents":  f"{BASE_PATH}/output/ablation/eicu_crd/without_relevant_documents.owl",
       #  "full_rigor": f"{BASE_PATH}/output/RIGOR/eicu_crd/mistral/enriched_ontology.owl",
    # },
    "chinook": {
        "no_rag":                      f"{BASE_PATH}/ablation_chinook/chinook/no_rag/enriched_ontology.owl",
        "only_schema_context":         f"{BASE_PATH}/ablation_chinook/chinook/only_schema_context/enriched_ontology.owl",
        "only_external_ontologies":    f"{BASE_PATH}/ablation_chinook/chinook/only_external_ontologies/enriched_ontology.owl",     
        "only_relevant_documents":     f"{BASE_PATH}/ablation_chinook/chinook/only_relevant_documents/enriched_ontology.owl",
        "without_schema_context":      f"{BASE_PATH}/ablation_chinook/chinook/without_schema_context/enriched_ontology.owl",
        "without_external_ontologies": f"{BASE_PATH}/ablation_chinook/chinook/without_external_ontologies/enriched_ontology.owl",
        "without_relevant_documents":  f"{BASE_PATH}/ablation_chinook/chinook/without_relevant_documents/enriched_ontology.owl",
        "full_rigor": f"{BASE_PATH}/output/RIGOR/chinook/claude/enriched_ontology.owl",
    },
}

OUTPUT_DIR = f"{BASE_PATH}/evaluation/ablation"

# Semantic similarity threshold
SIM_THRESHOLD = 0.55

# =========================================================
# SENTENCE TRANSFORMER
# =========================================================

print("Loading sentence transformer...")
sent_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# =========================================================
# REFERENCE SOURCE BUILDERS
# =========================================================

def build_schema_reference(schema: Dict) -> Dict[str, Set[str]]:
    """
    Build reference sets from the database schema.
    Returns:
        {
          "classes":    set of table names (CamelCase)
          "properties": set of column names
        }
    """
    def to_class(name): return "".join(p.capitalize() for p in name.split("_"))

    classes    = set()
    properties = set()

    for table_name, table_val in schema.items():
        classes.add(to_class(table_name))
        classes.add(table_name)  # also add snake_case for flexibility

        cols = table_val.get("columns", table_val) if isinstance(table_val, dict) else table_val
        for col in cols.keys():
            properties.add(col)
            properties.add(col.lower())

    return {"classes": classes, "properties": properties}

def build_external_ontology_reference(onto_path: str) -> Dict[str, Set[str]]:
    classes = set()
    properties = set()

    if not os.path.exists(onto_path):
        print(f"  Warning: external ontology path not found: {onto_path}")
        return {"classes": classes, "properties": properties}

    for root, _, files in os.walk(onto_path):
        for fname in files:
            if not fname.endswith((".owl", ".rdf", ".ttl", ".nt", ".n3")):
                continue

            fpath = os.path.join(root, fname)
            suffix = os.path.splitext(fname)[1].lower()

            if suffix == ".ttl":
                formats = ("turtle", "n3", "xml", "application/rdf+xml", "nt")
            elif suffix == ".nt":
                formats = ("nt", "turtle", "xml", "application/rdf+xml", "n3")
            elif suffix == ".n3":
                formats = ("n3", "turtle", "xml", "application/rdf+xml", "nt")
            else:
                formats = ("xml", "application/rdf+xml", "pretty-xml", "turtle", "n3", "nt")

            g = Graph()
            parsed = False
            last_error = None

            for fmt in formats:
                try:
                    g.parse(fpath, format=fmt)
                    parsed = True
                    break
                except Exception as e:
                    last_error = e

            if not parsed:
                print(f"  Skipped {fname}: {last_error}")
                continue

            for cls in g.subjects(RDF.type, OWL.Class):
                name = str(cls).split("#")[-1].split("/")[-1]
                if name:
                    classes.add(name)
                    classes.add(name.lower())

            props = (
                list(g.subjects(RDF.type, OWL.ObjectProperty))
                + list(g.subjects(RDF.type, OWL.DatatypeProperty))
                + list(g.subjects(RDF.type, RDF.Property))
            )

            for prop in props:
                name = str(prop).split("#")[-1].split("/")[-1]
                if name:
                    properties.add(name)
                    properties.add(name.lower())

    print(f"  External ontology reference: {len(classes)} classes, {len(properties)} props")
    return {"classes": classes, "properties": properties}

def build_docs_reference(docs_path: str) -> Set[str]:
    """
    Extract terms from documentation files (.txt, .md, .docx).
    Returns a flat set of terms.
    """
    terms = set()

    if not os.path.exists(docs_path):
        print(f"  Docs path not found: {docs_path}")
        return terms

    for root, _, files in os.walk(docs_path):
        for fname in files:
            if not fname.endswith((".txt", ".md", ".docx")):
                continue

            fpath = os.path.join(root, fname)

            try:
                # ---- HANDLE FILE TYPES ----
                if fname.endswith(".docx"):
                    doc = Document(fpath)
                    text = "\n".join(p.text for p in doc.paragraphs)
                else:
                    with open(fpath, encoding="utf-8", errors="ignore") as f:
                        text = f.read()

                # ---- TOKEN EXTRACTION ----
                for word in re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{2,}\b", text):
                    terms.add(word)
                    terms.add(word.lower())

            except Exception as e:
                print(f"  Skipped {fpath}: {e}")

    print(f"  Documentation reference: {len(terms)} terms extracted")
    return terms

# =========================================================
# ONTOLOGY ELEMENT EXTRACTOR
# =========================================================

def extract_elements_owlready(owl_path: str) -> Dict[str, Set[str]]:
    """
    Extract class and property names from an OWL file.

    Returns:
        {
            "classes": set(),
            "data_properties": set(),
            "object_properties": set()
        }

    If a property is typed as both owl:DatatypeProperty and owl:ObjectProperty,
    it is removed from data_properties to avoid double counting during evaluation.
    """
    result = {
        "classes": set(),
        "data_properties": set(),
        "object_properties": set(),
    }

    if not os.path.exists(owl_path):
        return result

    try:
        onto = get_ontology(owl_path).load()

        for cls in onto.classes():
            if cls != Thing and getattr(cls, "name", None):
                result["classes"].add(cls.name)

        for dp in onto.data_properties():
            if getattr(dp, "name", None):
                result["data_properties"].add(dp.name)

        for op in onto.object_properties():
            if getattr(op, "name", None):
                result["object_properties"].add(op.name)

    except Exception as e:
        print(f"  owlready2 parse error for {owl_path}: {e}")
        try:
            g = Graph()
            g.parse(owl_path, format="xml")

            for cls in g.subjects(RDF.type, OWL.Class):
                name = str(cls).split("#")[-1].split("/")[-1]
                if name:
                    result["classes"].add(name)

            for dp in g.subjects(RDF.type, OWL.DatatypeProperty):
                name = str(dp).split("#")[-1].split("/")[-1]
                if name:
                    result["data_properties"].add(name)

            for op in g.subjects(RDF.type, OWL.ObjectProperty):
                name = str(op).split("#")[-1].split("/")[-1]
                if name:
                    result["object_properties"].add(name)

        except Exception as e2:
            print(f"  rdflib fallback also failed for {owl_path}: {e2}")
            return result

    # Fix overlapping property types
    overlap = result["data_properties"] & result["object_properties"]
    if overlap:
        print(f"  Warning: overlapping property types in {owl_path}: {sorted(overlap)}")
        result["data_properties"] -= overlap

    return result
# =========================================================
# EXACT MATCH METRICS
# =========================================================

def exact_metrics(candidate: Set[str], reference: Set[str]) -> Dict:
    """
    Standard precision / recall / F1 using exact string match.
    Used for backward-compatible Table 4 values.
    """
    # Case-insensitive matching
    cand_lower = {s.lower() for s in candidate}
    ref_lower  = {s.lower() for s in reference}

    tp = len(cand_lower & ref_lower)
    fp = len(cand_lower - ref_lower)
    fn = len(ref_lower - cand_lower)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "f1":        round(f1, 4),
    }

# =========================================================
# SEMANTIC SIMILARITY MATCH METRICS
# =========================================================

def semantic_metrics(
    candidate: Set[str],
    reference: Set[str],
    threshold: float = SIM_THRESHOLD,
) -> Dict:
    """
    Precision / recall / F1 using semantic similarity instead of exact match.
    Each candidate element is matched to the closest reference element
    by cosine similarity; a match is counted if similarity >= threshold.

    This addresses the reviewer concern about exact lexical match being
    too strict (e.g., MedicalHistory vs medical_history).
    """
    if not candidate or not reference:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "threshold": threshold, "matched_pairs": []}

    cand_list = list(candidate)
    ref_list  = list(reference)

    cand_emb  = sent_model.encode(cand_list, convert_to_tensor=True)
    ref_emb   = sent_model.encode(ref_list,  convert_to_tensor=True)

    sim_matrix = util.cos_sim(cand_emb, ref_emb)  # [len(cand), len(ref)]

    # Precision: fraction of candidates that match at least one reference
    matched_cand   = 0
    matched_pairs  = []
    for i, cname in enumerate(cand_list):
        best_sim = float(sim_matrix[i].max())
        best_ref = ref_list[int(sim_matrix[i].argmax())]
        if best_sim >= threshold:
            matched_cand += 1
            matched_pairs.append((cname, best_ref, round(best_sim, 3)))

    # Recall: fraction of references matched by at least one candidate
    matched_ref = 0
    for j in range(len(ref_list)):
        if float(sim_matrix[:, j].max()) >= threshold:
            matched_ref += 1

    prec = matched_cand / len(cand_list) if cand_list else 0.0
    rec  = matched_ref  / len(ref_list)  if ref_list  else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "precision":     round(prec, 4),
        "recall":        round(rec, 4),
        "f1":            round(f1, 4),
        "threshold":     threshold,
        "matched_pairs": matched_pairs[:20],  # top 20 for logging
    }

# =========================================================
# NO-RAG SCHEMA SIGNAL ANALYSIS
# =========================================================

def explain_no_rag_f1(
    no_rag_classes: Set[str],
    schema_classes: Set[str],
    full_rigor_classes: Set[str],
) -> Dict:
    """
    Explain why No-RAG achieves a relatively high class F1.

    The hypothesis: No-RAG's class names are largely driven by table names
    from the schema, which are also what full RIGOR uses. So the high F1
    reflects schema signal, not retrieval quality.

    Returns breakdown of how much of No-RAG's class coverage comes from:
      a) direct schema alignment
      b) concepts not in schema (hallucinated or from training knowledge)
    """
    no_rag_lower   = {s.lower() for s in no_rag_classes}
    schema_lower   = {s.lower() for s in schema_classes}
    rigor_lower    = {s.lower() for s in full_rigor_classes}

    # No-RAG classes that match schema
    schema_derived  = no_rag_lower & schema_lower
    # No-RAG classes that match full RIGOR but not schema
    rigor_only      = (no_rag_lower & rigor_lower) - schema_lower
    # No-RAG classes that match neither
    unmatched       = no_rag_lower - schema_lower - rigor_lower

    return {
        "total_no_rag_classes":  len(no_rag_lower),
        "schema_derived":        len(schema_derived),
        "rigor_matched_non_schema": len(rigor_only),
        "unmatched":             len(unmatched),
        "schema_signal_rate":    round(len(schema_derived) / len(no_rag_lower), 4)
                                 if no_rag_lower else 0.0,
        "interpretation": (
    f"{round((len(schema_derived) / len(no_rag_lower)) * 100, 1) if no_rag_lower else 0.0}% "
    "of No-RAG classes match schema-derived names, suggesting that schema structure "
    "is the primary source of class coverage in this variant."
),
    }

# =========================================================
# SINGLE VARIANT EVALUATOR
# =========================================================

def evaluate_variant(
    variant_name:     str,
    owl_path:         str,
    schema_ref:       Dict[str, Set[str]],
    external_ref:     Dict[str, Set[str]],
    docs_ref:         Set[str],
    full_rigor_elems: Optional[Dict[str, Set[str]]] = None,
) -> Dict:
    """
    Evaluate one ablation variant against all three reference sources
    using both exact match and semantic similarity.
    """
    print(f"\n  Variant: {variant_name}")

    if not os.path.exists(owl_path):
        print(f"  File not found: {owl_path} — skipping")
        return {"variant": variant_name, "skipped": True}

    elems = extract_elements_owlready(owl_path)

    overlap = elems["data_properties"] & elems["object_properties"]
    if overlap:
        print(f"  Warning: overlapping property types in {variant_name}: {sorted(overlap)}")
        elems["data_properties"] -= overlap

    all_classes = elems["classes"]
    all_props   = elems["data_properties"] | elems["object_properties"]

    result = {"variant": variant_name}

    # ── 1. Against schema reference (primary reference) ──────────
    result["vs_schema"] = {
        "classes": {
            "exact":    exact_metrics(all_classes, schema_ref["classes"]),
            "semantic": semantic_metrics(all_classes, schema_ref["classes"]),
        },
        "properties": {
            "exact":    exact_metrics(all_props, schema_ref["properties"]),
            "semantic": semantic_metrics(all_props, schema_ref["properties"]),
        },
    }

    # ── 2. Against external ontology reference ────────────────────
    result["vs_external"] = {
        "classes": {
            "exact":    exact_metrics(all_classes, external_ref["classes"]),
            "semantic": semantic_metrics(all_classes, external_ref["classes"]),
        },
        "properties": {
            "exact":    exact_metrics(all_props, external_ref["properties"]),
            "semantic": semantic_metrics(all_props, external_ref["properties"]),
        },
    }

    # ── 3. Against docs reference (classes only — docs are richer) ─
    result["vs_docs"] = {
        "classes": {
            "exact":    exact_metrics(all_classes, docs_ref),
            "semantic": semantic_metrics(all_classes, docs_ref),
        },
    }

    # ── 4. Against full RIGOR (backward compatibility, Table 4) ───
    if full_rigor_elems:
        result["vs_full_rigor"] = {
            "classes": {
                "exact":    exact_metrics(all_classes, full_rigor_elems["classes"]),
                "semantic": semantic_metrics(all_classes, full_rigor_elems["classes"]),
            },
            "data_properties": {
                "exact":    exact_metrics(
                    elems["data_properties"],
                    full_rigor_elems["data_properties"]
                ),
                "semantic": semantic_metrics(
                    elems["data_properties"],
                    full_rigor_elems["data_properties"]
                ),
            },
            "object_properties": {
                "exact":    exact_metrics(
                    elems["object_properties"],
                    full_rigor_elems["object_properties"]
                ),
                "semantic": semantic_metrics(
                    elems["object_properties"],
                    full_rigor_elems["object_properties"]
                ),
            },
        }

    # ── 5. Element counts ─────────────────────────────────────────
    result["counts"] = {
        "classes":           len(all_classes),
        "data_properties":   len(elems["data_properties"]),
        "object_properties": len(elems["object_properties"]),
    }

    return result

# =========================================================
# TABLE 4 FORMATTER
# =========================================================

# =========================================================
# TABLE 4 FORMATTER
# =========================================================

def format_table4(all_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Build a DataFrame in the format of Table 4 in the paper.
    Reports both exact and semantic F1 for classes and properties against
    three independent reference sources:
      1. Database schema        — primary structural reference
      2. External ontologies    — BioPortal alignment reference
      3. Documentation corpus   — domain vocabulary reference (classes only)
    Plus full RIGOR reference for backward compatibility.
    """
    rows = []
    for variant, res in all_results.items():
        if res.get("skipped"):
            continue

        row = {"Variant": variant}

        # ── 1. Schema reference (primary) ─────────────────────────
        sch = res.get("vs_schema", {})
        row["CLS_P_schema_exact"]   = sch.get("classes", {}).get("exact", {}).get("precision", 0)
        row["CLS_R_schema_exact"]   = sch.get("classes", {}).get("exact", {}).get("recall", 0)
        row["CLS_F1_schema_exact"]  = sch.get("classes", {}).get("exact", {}).get("f1", 0)
        row["CLS_F1_schema_sem"]    = sch.get("classes", {}).get("semantic", {}).get("f1", 0)
        row["PROP_F1_schema_exact"] = sch.get("properties", {}).get("exact", {}).get("f1", 0)
        row["PROP_F1_schema_sem"]   = sch.get("properties", {}).get("semantic", {}).get("f1", 0)

        # ── 2. External ontology reference ────────────────────────
        ext = res.get("vs_external", {})
        row["CLS_F1_ext_exact"]  = ext.get("classes", {}).get("exact", {}).get("f1", 0)
        row["CLS_F1_ext_sem"]    = ext.get("classes", {}).get("semantic", {}).get("f1", 0)
        row["PROP_F1_ext_exact"] = ext.get("properties", {}).get("exact", {}).get("f1", 0)
        row["PROP_F1_ext_sem"]   = ext.get("properties", {}).get("semantic", {}).get("f1", 0)

        # ── 3. Documentation reference (classes only) ─────────────
        docs = res.get("vs_docs", {})
        row["CLS_F1_docs_exact"] = docs.get("classes", {}).get("exact", {}).get("f1", 0)
        row["CLS_F1_docs_sem"]   = docs.get("classes", {}).get("semantic", {}).get("f1", 0)

        # ── 4. Full RIGOR reference (backward compatibility) ──────
        rig = res.get("vs_full_rigor", {})
        row["CLS_F1_rigor_exact"]    = rig.get("classes", {}).get("exact", {}).get("f1", 0)
        row["CLS_F1_rigor_sem"]      = rig.get("classes", {}).get("semantic", {}).get("f1", 0)
        row["DP_F1_rigor_exact"]     = rig.get("data_properties", {}).get("exact", {}).get("f1", 0)
        row["DP_F1_rigor_sem"]       = rig.get("data_properties", {}).get("semantic", {}).get("f1", 0)
        row["OP_F1_rigor_exact"]     = rig.get("object_properties", {}).get("exact", {}).get("f1", 0)
        row["OP_F1_rigor_sem"]       = rig.get("object_properties", {}).get("semantic", {}).get("f1", 0)

        # ── 5. Element counts ─────────────────────────────────────
        counts = res.get("counts", {})
        row["N_Classes"]    = counts.get("classes", 0)
        row["N_DataProps"]  = counts.get("data_properties", 0)
        row["N_ObjectProps"]= counts.get("object_properties", 0)

        rows.append(row)

    return pd.DataFrame(rows)
# =========================================================
# MAIN
# =========================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for db_name, schema_path in SCHEMAS.items():
        print(f"\n{'='*65}")
        print(f"ABLATION STUDY: {db_name}")
        print(f"{'='*65}")

        if not os.path.exists(schema_path):
            print(f"Schema not found: {schema_path} — skipping")
            continue

        with open(schema_path, "r") as f:
            schema = json.load(f)

        # Build reference sources
        print("\nBuilding reference sources...")
        schema_ref   = build_schema_reference(schema)
        external_ref = build_external_ontology_reference(EXTERNAL_ONTOLOGY_PATH)
        docs_ref     = build_docs_reference(DOCS_PATH)

        print(f"  Schema reference   : {len(schema_ref['classes'])} classes, "
              f"{len(schema_ref['properties'])} properties")

        # Load full RIGOR elements for backward-compatible comparison
        rigor_path  = ABLATION_VARIANTS[db_name].get("full_rigor", "")
        rigor_elems = extract_elements_owlready(rigor_path) if os.path.exists(rigor_path) else None

        # Evaluate all variants
        all_results = {}
        variants    = ABLATION_VARIANTS.get(db_name, {})

        for variant_name, owl_path in variants.items():
            res = evaluate_variant(
                variant_name  = variant_name,
                owl_path      = owl_path,
                schema_ref    = schema_ref,
                external_ref  = external_ref,
                docs_ref      = docs_ref,
                full_rigor_elems = rigor_elems,
            )
            all_results[variant_name] = res

        # No-RAG schema signal explanation
        print("\nNo-RAG schema signal analysis...")
        no_rag_path = variants.get("no_rag", "")
        if os.path.exists(no_rag_path) and rigor_elems:
            no_rag_elems = extract_elements_owlready(no_rag_path)
            no_rag_explanation = explain_no_rag_f1(
                no_rag_classes     = no_rag_elems["classes"],
                schema_classes     = schema_ref["classes"],
                full_rigor_classes = rigor_elems["classes"],
            )
            print(f"  Schema signal rate: {no_rag_explanation['schema_signal_rate']}")
            print(f"  {no_rag_explanation['interpretation']}")
        else:
            no_rag_explanation = {"note": "No-RAG ontology not available"}

        # Build Table 4
        table4_df = format_table4(all_results)

        # Save outputs
        prefix = os.path.join(OUTPUT_DIR, db_name)

        # Table 4 CSV
        csv_path = f"{prefix}_table4.csv"
        table4_df.to_csv(csv_path, index=False)
        print(f"\nTable 4 saved: {csv_path}")

        # Full JSON
        json_path = f"{prefix}_ablation_full.json"
        with open(json_path, "w") as f:
            # Remove large matched_pairs lists before saving
            slim = {}
            for vname, vres in all_results.items():
                slim[vname] = json.loads(
                    json.dumps(vres, default=str)
                )
                for ref_key in ["vs_schema", "vs_external", "vs_docs", "vs_full_rigor"]:
                    if ref_key in slim[vname]:
                        for elem_key in slim[vname][ref_key]:
                            if "semantic" in slim[vname][ref_key][elem_key]:
                                slim[vname][ref_key][elem_key]["semantic"].pop(
                                    "matched_pairs", None
                                )
            json.dump({
                "database":         db_name,
                "no_rag_analysis":  no_rag_explanation,
                "results":          slim,
            }, f, indent=2)
        print(f"Full results saved: {json_path}")

        # Print Table 4 to console
        print(f"\n--- Table 4 ({db_name}) ---")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 160)
        print(table4_df.to_string(index=False))


if __name__ == "__main__":
    main()
