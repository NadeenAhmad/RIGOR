"""
eval.py — RIGOR Ontology Evaluation Script

Evaluates ontologies across multiple dimensions as required by
the ESWC 2026 reviewer feedback:

  1. CQ-based quality scoring (6 dimensions, Judge-LLM)
  2. Structural analysis (classes, object/data properties, axioms)
  3. Semantic coverage of database schema
  4. Syntax and logical consistency validation (rdflib)
  5. W3C Direct Mapping baseline included in all comparisons
  6. Runtime and token usage reporting

Outputs:
  - <output_prefix>_scores.json     : per-chunk and aggregate CQ scores
  - <output_prefix>_structural.json : structural metrics per ontology
  - <output_prefix>_coverage.json   : schema semantic coverage per ontology
  - <output_prefix>_summary.json    : full comparison table across all ontologies
"""

import os
import json
import re
import time
import textwrap
from typing import Dict, List, Tuple, Optional

import requests
from rdflib import Graph, URIRef, RDF, OWL, RDFS, Literal, Namespace
from rdflib.namespace import XSD
from sentence_transformers import SentenceTransformer, util
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# CONFIGURATION
# =========================================================

BASE_PATH = "YOUR_BASE_PATH"

# Competency questions directory (one .txt file per table)
COMPETENCY_QUESTIONS_DIR = f"{BASE_PATH}/cqs"

# Schema files
SCHEMAS = {
    "real_world": f"{BASE_PATH}/sql_schema/schema_rd.json",
   # "eicu_crd":   f"{BASE_PATH}/sql_schema/schema_icu.json",
  # "chinook":    f"{BASE_PATH}/sql_schema/schema_chinook.json",
}

# Ontologies to evaluate: {label: path}
# Add or remove entries as needed.
ONTOLOGIES = {
    "real_world": {
       # "direct_mapping": f"{BASE_PATH}/output/direct_mapping/direct_mapping_rd.owl",
       # "baseline_claude":    f"{BASE_PATH}/output/baseline/real_world/claude/baseline_ontology.owl",
       # "baseline_mistral":   f"{BASE_PATH}/output/baseline/real_world/mistral/baseline_ontology.owl",
       # "baseline_deepseek":  f"{BASE_PATH}/output/baseline/real_world/deepseek/baseline_ontology.owl",
       # "non_iterative_claude":   f"{BASE_PATH}/output/non_iterative/real_world/claude/non_iterative_ontology.owl",
       # "non_iterative_mistral":  f"{BASE_PATH}/output/non_iterative/real_world/mistral/non_iterative_ontology.owl",
       # "non_iterative_deepseek": f"{BASE_PATH}/output/non_iterative/real_world/deepseek/non_iterative_ontology.owl",
    # "RIGOR_claude":   f"{BASE_PATH}/output/RIGOR/real_world/claude/enriched_ontology.owl",
     #   "RIGOR_mistral":  f"{BASE_PATH}/output/RIGOR/real_world/mistral/enriched_ontology.owl",
       "RIGOR_deepseek": f"{BASE_PATH}/output/RIGOR/real_world/deepseek/enriched_ontology.owl",

   },
       
   #     "eicu_crd": {
        #"direct_mapping":    f"{BASE_PATH}/output/direct_mapping/direct_mapping_icu.owl",
       # "baseline_claude":   f"{BASE_PATH}/output/baseline/eicu_crd/claude/baseline_ontology.owl",
       # "baseline_mistral":  f"{BASE_PATH}/output/baseline/eicu_crd/mistral/baseline_ontology.owl",
       # "baseline_deepseek": f"{BASE_PATH}/output/baseline/eicu_crd/deepseek/baseline_ontology.owl",
        #"non_iterative_claude":   f"{BASE_PATH}/output/non_iterative/eicu_crd/claude/non_iterative_ontology.owl",
       # "non_iterative_mistral":  f"{BASE_PATH}/output/non_iterative/eicu_crd/mistral/non_iterative_ontology.owl",
       # "non_iterative_deepseek": f"{BASE_PATH}/output/non_iterative/eicu_crd/deepseek/non_iterative_ontology.owl",
    #  "RIGOR_claude":   f"{BASE_PATH}/output/RIGOR/eicu_crd/claude/enriched_ontology.owl",
     #  "RIGOR_mistral":  f"{BASE_PATH}/output/RIGOR/eicu_crd/mistral/enriched_ontology.owl",
     #   "RIGOR_deepseek": f"{BASE_PATH}/output/RIGOR/eicu_crd/deepseek/enriched_ontology.owl",
    #}
  #  "chinook": {
   #     "direct_mapping":    f"{BASE_PATH}/output/direct_mapping/direct_mapping_chinook.owl",
    #    "baseline_claude":   f"{BASE_PATH}/output/baseline/chinook/claude/baseline_ontology.owl",
     #   "baseline_mistral":  f"{BASE_PATH}/output/baseline/chinook/mistral/baseline_ontology.owl",
      #  "baseline_deepseek": f"{BASE_PATH}/output/baseline/chinook/deepseek/baseline_ontology.owl",
       # "non_iterative_claude":   f"{BASE_PATH}/output/non_iterative/chinook/claude/non_iterative_ontology.owl",
     #   "non_iterative_mistral":  f"{BASE_PATH}/output/non_iterative/chinook/mistral/non_iterative_ontology.owl",
      #  "non_iterative_deepseek": f"{BASE_PATH}/output/non_iterative/chinook/deepseek/non_iterative_ontology.owl",
      #  "RIGOR_claude":   f"{BASE_PATH}/output/RIGOR/chinook/claude/enriched_ontology.owl",
      #  "RIGOR_mistral":  f"{BASE_PATH}/output/RIGOR/chinook/mistral/enriched_ontology.owl",
      #  "RIGOR_deepseek": f"{BASE_PATH}/output/RIGOR/chinook/deepseek/enriched_ontology.owl",
   # }
}

OUTPUT_DIR    = f"{BASE_PATH}/evaluation"
# Judge-LLM: GPT-5.4 (latest) via OpenRouter
# GPT-5.4 is from a different model family than all three generators
# (Claude/Anthropic, Mistral, DeepSeek are generators — OpenAI GPT is the judge)
# This eliminates self-evaluation bias and uses the strongest available judge.
JUDGE_MODEL   = "openai/gpt-5.4"
MAX_CHUNK_SIZE = 2048  # characters per ontology chunk
# Judge-LLM toggle
ENABLE_CQ_JUDGE = False
# Semantic coverage similarity threshold (empirically tuned, per prior work)
COVERAGE_THRESHOLD = 0.55

# =========================================================
# LLM SETUP
# =========================================================

def call_judge_llm(prompt: str, api_key: str, retries: int = 3) -> str:
    """
    Call GPT-5.4 (latest) via OpenRouter as the Judge-LLM.
    GPT-5.4 is from a different model family than all three evaluated generators
    (Claude/Anthropic, Mistral, DeepSeek), eliminating self-evaluation bias.
    Retries up to 3 times on timeout or server error, with increasing wait time.
    """
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                timeout=120,   # increased from 60s
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model":       JUDGE_MODEL,
                    "messages":    [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens":  50,
                },
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            elif r.status_code in (429, 500, 502, 503):
                # Rate limit or server error — wait and retry
                wait = attempt * 10
                print(f"  Judge API {r.status_code} — retrying in {wait}s "
                      f"(attempt {attempt}/{retries})")
                time.sleep(wait)
            else:
                print(f"  Judge API error {r.status_code}: {r.text[:200]}")
                return ""
        except requests.exceptions.Timeout:
            wait = attempt * 15
            print(f"  Judge API timed out — retrying in {wait}s "
                  f"(attempt {attempt}/{retries})")
            time.sleep(wait)
        except Exception as e:
            print(f"  Judge API call failed: {e}")
            return ""
    print(f"  Judge API failed after {retries} attempts — score will be 0.0")
    return ""

print("Loading sentence transformer for semantic coverage...")
sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "") if ENABLE_CQ_JUDGE else ""

# =========================================================
# COMPETENCY QUESTION LOADING
# =========================================================

def load_competency_questions(cqs_dir: str) -> List[Dict]:
    """Load all CQ .txt files from cqs_dir."""
    questions = []
    if not os.path.exists(cqs_dir):
        print(f"  Warning: CQs directory not found: {cqs_dir}")
        return questions
    for file in sorted(os.listdir(cqs_dir)):
        if file.endswith(".txt"):
            with open(os.path.join(cqs_dir, file), "r") as f:
                content = f.read().split("\n\n")
                questions.append({
                    "question": content[0].strip(),
                    "answer":   content[1].strip() if len(content) > 1 else "",
                })
    print(f"  Loaded {len(questions)} competency questions")
    return questions

# =========================================================
# ONTOLOGY LOADING & CHUNKING
# =========================================================
def load_graph(owl_path: str) -> Optional[Graph]:
    """
    Parse an ontology file into an rdflib Graph.
    Supports:
      - RDF/XML, Turtle, N3 via rdflib
      - simplified Manchester Syntax used by direct mapping files
    """
    if not os.path.exists(owl_path):
        print(f"  Warning: ontology not found: {owl_path}")
        return None

    # Read file first to detect Manchester syntax
    with open(owl_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    stripped = text.lstrip()

    # Detect Manchester syntax
    if stripped.startswith("Prefix:") or stripped.startswith("Ontology:") or "Class:" in stripped:
        try:
            return parse_manchester_direct_mapping(text)
        except Exception as e:
            print(f"  Warning: Manchester parsing failed for {owl_path}: {e}")
            return None

    # Otherwise try standard rdflib formats
    g = Graph()
    for fmt in ("xml", "turtle", "n3"):
        try:
            g.parse(owl_path, format=fmt)
            return g
        except Exception:
            continue

    print(f"  Warning: could not parse {owl_path}")
    return None

def parse_manchester_direct_mapping(text: str) -> Graph:
    """
    Parse the simplified Manchester Syntax produced by mapping.py
    into an rdflib Graph.
    """
    g = Graph()
    base = Namespace("http://example.org/ontology#")
    prov = Namespace("http://www.w3.org/ns/prov#")

    g.bind("", base)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)
    g.bind("prov", prov)

    XSD_TYPES = {
        "xsd:string": XSD.string,
        "xsd:integer": XSD.integer,
        "xsd:float": XSD.float,
        "xsd:boolean": XSD.boolean,
        "xsd:date": XSD.date,
        "xsd:dateTime": XSD.dateTime,
        "xsd:time": XSD.time,
        "xsd:decimal": XSD.decimal,
        "xsd:double": XSD.double,
        "xsd:base64Binary": XSD.base64Binary,
    }

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    blocks = []
    current = []

    for line in lines:
        if line.startswith(("Class:", "ObjectProperty:", "DataProperty:", "Individual:")):
            if current:
                blocks.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append(current)

    for block in blocks:
        header = block[0]
        rest = block[1:]

        def get_value(prefix: str):
            for ln in rest:
                if ln.startswith(prefix):
                    return ln[len(prefix):].strip()
            return None

        def local_name(value: str) -> str:
            value = value.strip()
            if value.startswith(":"):
                return value[1:]
            return value

        if header.startswith("Class:"):
            cls_name = local_name(header.split(":", 1)[1].strip())
            cls_uri = base[cls_name]
            g.add((cls_uri, RDF.type, OWL.Class))

        elif header.startswith("ObjectProperty:"):
            prop_name = local_name(header.split(":", 1)[1].strip())
            prop_uri = base[prop_name]
            g.add((prop_uri, RDF.type, OWL.ObjectProperty))

            dom = get_value("Domain:")
            rng = get_value("Range:")
            if dom:
                g.add((prop_uri, RDFS.domain, base[local_name(dom)]))
            if rng:
                g.add((prop_uri, RDFS.range, base[local_name(rng)]))

        elif header.startswith("DataProperty:"):
            prop_name = local_name(header.split(":", 1)[1].strip())
            prop_uri = base[prop_name]
            g.add((prop_uri, RDF.type, OWL.DatatypeProperty))

            dom = get_value("Domain:")
            rng = get_value("Range:")
            if dom:
                g.add((prop_uri, RDFS.domain, base[local_name(dom)]))
            if rng and rng in XSD_TYPES:
                g.add((prop_uri, RDFS.range, XSD_TYPES[rng]))

        elif header.startswith("Individual:"):
            ind_name = local_name(header.split(":", 1)[1].strip())
            ind_uri = base[ind_name]

            types_val = get_value("Types:")
            if types_val:
                if types_val == "prov:Activity":
                    g.add((ind_uri, RDF.type, prov.Activity))
                elif types_val == "prov:Entity":
                    g.add((ind_uri, RDF.type, prov.Entity))
                elif types_val == "prov:SoftwareAgent":
                    g.add((ind_uri, RDF.type, prov.SoftwareAgent))

            for ln in rest:
                if "prov:used" in ln:
                    target = ln.split("prov:used", 1)[1].strip().rstrip(",")
                    g.add((ind_uri, prov.used, base[local_name(target)]))
                elif "prov:wasAssociatedWith" in ln:
                    target = ln.split("prov:wasAssociatedWith", 1)[1].strip().rstrip(",")
                    g.add((ind_uri, prov.wasAssociatedWith, base[local_name(target)]))

    return g

def load_ontology_chunks(g: Graph) -> List[str]:
    """Convert an rdflib Graph into text chunks for LLM evaluation."""
    chunks = []

    # Classes with their properties
    for cls in g.subjects(predicate=RDF.type, object=OWL.Class):
        try:
            class_name = g.qname(cls)
        except Exception:
            class_name = str(cls).split("#")[-1].strip("<>")

        chunk    = f"Class: {class_name}\n"
        props    = []
        label    = next(g.objects(cls, RDFS.label), None)
        comment  = next(g.objects(cls, RDFS.comment), None)
        if label:
            chunk += f"  Label: {label}\n"
        if comment:
            chunk += f"  Comment: {comment}\n"

        for prop in g.subjects(predicate=RDFS.domain, object=cls):
            try:
                prop_name = g.qname(prop)
            except Exception:
                prop_name = str(prop).split("#")[-1].strip("<>")
            ranges = []
            for rng in g.objects(prop, RDFS.range):
                try:
                    ranges.append(g.qname(rng) if isinstance(rng, URIRef) else str(rng))
                except Exception:
                    ranges.append(str(rng).split("#")[-1].strip("<>"))
            props.append(f"  - {prop_name}: Range={', '.join(ranges)}")

        if props:
            chunk += "Properties:\n" + "\n".join(props)
            chunks.append(chunk)

    # Object properties (standalone)
    for prop in g.subjects(predicate=RDF.type, object=OWL.ObjectProperty):
        try:
            prop_name = g.qname(prop)
        except Exception:
            prop_name = str(prop).split("#")[-1]
        info  = [f"Object Property: {prop_name}"]
        doms  = []
        for d in g.objects(prop, RDFS.domain):
            try:
                doms.append(g.qname(d) if isinstance(d, URIRef) else str(d))
            except Exception:
                doms.append(str(d).split("#")[-1].strip("<>"))
        rngs  = []
        for r in g.objects(prop, RDFS.range):
            try:
                rngs.append(g.qname(r) if isinstance(r, URIRef) else str(r))
            except Exception:
                rngs.append(str(r).split("#")[-1].strip("<>"))
        if doms:
            info.append(f"  Domain: {', '.join(doms)}")
        if rngs:
            info.append(f"  Range: {', '.join(rngs)}")
        chunks.append("\n".join(info))

    return chunks

# =========================================================
# STRUCTURAL ANALYSIS
# =========================================================

def structural_analysis(g: Graph) -> Dict:
    """
    Count classes, object properties, data properties, axioms,
    labels, comments, and provenance annotations.
    """
    n_classes    = len(list(g.subjects(RDF.type, OWL.Class)))
    n_obj_props  = len(list(g.subjects(RDF.type, OWL.ObjectProperty)))
    n_data_props = len(list(g.subjects(RDF.type, OWL.DatatypeProperty)))
    n_axioms     = len(g)   # total triples as proxy for axiom count
    n_labels     = len(list(g.subject_objects(RDFS.label)))
    n_comments   = len(list(g.subject_objects(RDFS.comment)))

    # Count prov:wasDerivedFrom
    PROV = URIRef("http://www.w3.org/ns/prov#")
    n_prov = len(list(g.subject_objects(URIRef(str(PROV) + "wasDerivedFrom"))))

    # Count SubClassOf axioms (excluding owl:Thing)
    n_subclass = len([
        (s, o) for s, o in g.subject_objects(RDFS.subClassOf)
        if str(o) != str(OWL.Thing)
    ])

    # Count DisjointWith axioms
    n_disjoint = len(list(g.subject_objects(OWL.disjointWith)))

    return {
        "classes":            n_classes,
        "object_properties":  n_obj_props,
        "data_properties":    n_data_props,
        "total_axioms":       n_axioms,
        "labels":             n_labels,
        "comments":           n_comments,
        "subclass_axioms":    n_subclass,
        "disjoint_axioms":    n_disjoint,
        "provenance":         n_prov,
    }

# =========================================================
# SYNTAX & LOGICAL CONSISTENCY VALIDATION
# =========================================================

def validate_ontology(owl_path: str, g: Optional[Graph]) -> Dict:
    """
    Validate syntax and check for common modelling pitfalls using rdflib.
    Returns a dict of findings.
    """
    result = {
        "syntax_valid":           False,
        "parse_format":           None,
        "punning_violations":     [],   # URI declared as both ObjProp and DataProp
        "missing_domain_range":   [],   # properties without domain or range
        "self_referential_props": [],   # ObjectProperty domain == range
        "date_typed_as_float":    [],   # date columns still xsd:float
    }

    if g is None:
        return result

    result["syntax_valid"] = True

    # Detect punning: same URI as both ObjectProperty and DatatypeProperty
    obj_props  = set(g.subjects(RDF.type, OWL.ObjectProperty))
    data_props = set(g.subjects(RDF.type, OWL.DatatypeProperty))
    for uri in obj_props & data_props:
        result["punning_violations"].append(str(uri).split("#")[-1])

    # Detect missing domain or range
    all_props = obj_props | data_props
    for prop in all_props:
        has_domain = any(True for _ in g.objects(prop, RDFS.domain))
        has_range  = any(True for _ in g.objects(prop, RDFS.range))
        if not has_domain or not has_range:
            result["missing_domain_range"].append(str(prop).split("#")[-1])

    # Detect self-referential ObjectProperties
    for prop in obj_props:
        doms = list(g.objects(prop, RDFS.domain))
        rngs = list(g.objects(prop, RDFS.range))
        for d in doms:
            for r in rngs:
                if d == r:
                    result["self_referential_props"].append(str(prop).split("#")[-1])

    # Detect date columns still typed as xsd:float
    date_keywords = ("date", "time", "created_at", "timestamp")
    for prop in data_props:
        name = str(prop).split("#")[-1].lower()
        if any(kw in name for kw in date_keywords):
            for rng in g.objects(prop, RDFS.range):
                if str(rng) == str(XSD.float):
                    result["date_typed_as_float"].append(str(prop).split("#")[-1])

    return result

# =========================================================
# SEMANTIC COVERAGE OF DATABASE SCHEMA
# =========================================================

def semantic_coverage(g: Optional[Graph], schema: Dict) -> Dict:
    """
    Measure how well the ontology covers the database schema.

    For each table:
      - Check if a corresponding class exists (exact or semantic match)
      - Check if each column is represented as a DataProperty or ObjectProperty

    Returns coverage rates at table and column level.
    """
    if g is None:
        return {"table_coverage_rate": 0.0, "column_coverage_rate": 0.0, "details": {}}

    # Collect ontology class names and property names
    onto_class_names = []
    for cls in g.subjects(RDF.type, OWL.Class):
        name = str(cls).split("#")[-1].lower()
        onto_class_names.append(name)
        # Also try rdfs:label
        for lbl in g.objects(cls, RDFS.label):
            onto_class_names.append(str(lbl).lower())

    onto_prop_names = []
    for prop in g.subjects(RDF.type, OWL.DatatypeProperty):
        onto_prop_names.append(str(prop).split("#")[-1].lower())
    for prop in g.subjects(RDF.type, OWL.ObjectProperty):
        onto_prop_names.append(str(prop).split("#")[-1].lower())

    # Embed ontology names
    if onto_class_names:
        cls_embeddings  = sentence_model.encode(onto_class_names,  convert_to_tensor=True)
    if onto_prop_names:
        prop_embeddings = sentence_model.encode(onto_prop_names, convert_to_tensor=True)

    details             = {}
    matched_tables      = 0
    matched_cols        = 0
    total_tables        = 0
    total_cols          = 0

    for table_name, table_val in schema.items():
        total_tables += 1
        cols = table_val.get("columns", table_val) if isinstance(table_val, dict) else table_val

        # Table-level match
        table_emb = sentence_model.encode([table_name.lower()], convert_to_tensor=True)
        if onto_class_names:
            sims       = util.cos_sim(table_emb, cls_embeddings)[0]
            best_sim   = float(sims.max())
            best_match = onto_class_names[int(sims.argmax())]
            table_matched = best_sim >= COVERAGE_THRESHOLD
        else:
            best_sim, best_match, table_matched = 0.0, "", False

        if table_matched:
            matched_tables += 1

        # Column-level match
        col_details   = {}
        for col_name in cols.keys():
            total_cols += 1
            col_emb     = sentence_model.encode([col_name.lower()], convert_to_tensor=True)
            if onto_prop_names:
                sims        = util.cos_sim(col_emb, prop_embeddings)[0]
                col_sim     = float(sims.max())
                col_match   = onto_prop_names[int(sims.argmax())]
                col_matched = col_sim >= COVERAGE_THRESHOLD
            else:
                col_sim, col_match, col_matched = 0.0, "", False

            if col_matched:
                matched_cols += 1
            col_details[col_name] = {
                "matched":    col_matched,
                "best_match": col_match,
                "similarity": round(col_sim, 3),
            }

        details[table_name] = {
            "table_matched":    table_matched,
            "table_similarity": round(best_sim, 3),
            "table_best_match": best_match,
            "column_details":   col_details,
        }

    table_rate  = round(matched_tables / total_tables, 4) if total_tables else 0.0
    column_rate = round(matched_cols   / total_cols,   4) if total_cols   else 0.0

    return {
        "table_coverage_rate":  table_rate,
        "column_coverage_rate": column_rate,
        "matched_tables":       matched_tables,
        "total_tables":         total_tables,
        "matched_columns":      matched_cols,
        "total_columns":        total_cols,
        "details":              details,
    }

# =========================================================
# HEATMAP VISUALISATION
# =========================================================

def plot_coverage_heatmap(
    coverage_result: Dict,
    ontology_label:  str,
    db_name:         str,
    output_dir:      str,
) -> None:
    """
    Generate a heatmap of class-to-column semantic similarity scores.

    Rows    = schema tables
    Columns = top-N ontology classes by similarity to that table
    Cell    = cosine similarity score

    Produces one heatmap per ontology showing how well each
    schema table is represented in the ontology.
    """
    details = coverage_result.get("details", {})
    if not details:
        print(f"  No coverage details for heatmap: {ontology_label}")
        return

    # Build matrix: rows = tables, cols = unique best-match class names
    table_names  = list(details.keys())
    class_names  = [details[t]["table_best_match"] for t in table_names]
    sim_scores   = [details[t]["table_similarity"]  for t in table_names]

    # For per-table column similarity, build a second heatmap showing
    # column coverage within each table
    col_heatmap_data = {}
    for table_name, tdata in details.items():
        col_sims = {
            col: info["similarity"]
            for col, info in tdata["column_details"].items()
        }
        col_heatmap_data[table_name] = col_sims

    os.makedirs(output_dir, exist_ok=True)

    # ── Heatmap 1: Table-level coverage ──────────────────────────
    fig, ax = plt.subplots(figsize=(8, max(4, len(table_names) * 0.45)))
    matrix = np.array(sim_scores).reshape(-1, 1)
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=["Best Match Similarity"],
        yticklabels=table_names,
        vmin=0.0, vmax=1.0,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(
        f"Table Coverage — {ontology_label}\n"
        f"({db_name}, threshold={COVERAGE_THRESHOLD})",
        fontsize=11,
    )
    ax.set_ylabel("Schema Tables")
    plt.tight_layout()
    out_path = os.path.join(
        output_dir,
        f"{db_name}_{ontology_label}_table_coverage.png",
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Heatmap saved: {out_path}")

    # ── Heatmap 2: Column-level coverage per table ────────────────
    # Only show tables that have columns (skip empty lookup tables)
    tables_with_cols = {
        t: d for t, d in col_heatmap_data.items() if d
    }
    if not tables_with_cols:
        return

    # Collect all unique column names across all tables (too many to show all)
    # Instead: show one row per table, columns = similarity of each schema col
    # Limit to tables with the most columns for readability
    top_tables = sorted(
        tables_with_cols.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:10]  # top 10 tables by column count

    for t_name, col_sims in top_tables:
        col_labels = list(col_sims.keys())
        col_vals   = np.array(list(col_sims.values())).reshape(1, -1)

        fig_w = max(10, len(col_labels) * 0.55)
        fig, ax = plt.subplots(figsize=(fig_w, 2.5))
        sns.heatmap(
            col_vals,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            xticklabels=col_labels,
            yticklabels=[t_name],
            vmin=0.0, vmax=1.0,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title(
            f"Column Coverage — {t_name} | {ontology_label}\n"
            f"({db_name}, threshold={COVERAGE_THRESHOLD})",
            fontsize=10,
        )
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.tight_layout()
        safe_table = t_name.replace("/", "_")
        out_path   = os.path.join(
            output_dir,
            f"{db_name}_{ontology_label}_{safe_table}_col_coverage.png",
        )
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  Column heatmaps saved for top {len(top_tables)} tables")


def plot_comparison_heatmap(
    all_coverage: Dict[str, Dict],
    db_name:      str,
    output_dir:   str,
) -> None:
    """
    Generate a single comparison heatmap showing table coverage rate
    across ALL evaluated ontologies side by side.

    Rows    = schema tables
    Columns = ontology variants (direct_mapping, baseline_claude, etc.)
    Cell    = table similarity score for that ontology

    This is the key paper figure showing RIGOR vs Direct Mapping vs Baseline.
    """
    if not all_coverage:
        return

    # Collect all table names from the first ontology
    first_cov = next(iter(all_coverage.values()))
    table_names = list(first_cov.get("details", {}).keys())
    if not table_names:
        return

    onto_labels = list(all_coverage.keys())
    matrix = np.zeros((len(table_names), len(onto_labels)))

    for j, onto_label in enumerate(onto_labels):
        details = all_coverage[onto_label].get("details", {})
        for i, t_name in enumerate(table_names):
            matrix[i, j] = details.get(t_name, {}).get("table_similarity", 0.0)

    os.makedirs(output_dir, exist_ok=True)
    fig_h = max(6, len(table_names) * 0.45)
    fig_w = max(8, len(onto_labels) * 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=onto_labels,
        yticklabels=table_names,
        vmin=0.0, vmax=1.0,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(
        f"Semantic Table Coverage Comparison — {db_name}\n"
        f"(cosine similarity, threshold={COVERAGE_THRESHOLD})",
        fontsize=11,
    )
    ax.set_xlabel("Ontology Variant")
    ax.set_ylabel("Schema Tables")
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{db_name}_coverage_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Comparison heatmap saved: {out_path}")


# =========================================================
# CQ-BASED QUALITY SCORING
# =========================================================

def get_metric_criteria(metric: str) -> str:
    criteria = {
        "accuracy":      "1. Correctness of domain representation\n2. Alignment with competency questions",
        "completeness":  "1. Coverage of required concepts\n2. Presence of necessary relationships",
        "conciseness":   "1. Absence of redundancy\n2. Minimal complexity",
        "adaptability":  "1. Extensibility\n2. Modular design\n3. Clear naming",
        "clarity":       "1. Readability\n2. Unambiguous definitions\n3. Documentation quality",
        "consistency":   "1. Logical coherence\n2. Valid relationships\n3. Proper inheritance",
      "domain_enrichment" : "1. The degree to which the ontology transcends structural schema mirroring to capture domain-level semantics: (a) annotations that explain the clinical or domain meaning of classes and properties, including expansion of abbreviated names and documentation of measurement units; (b) value encoding annotations that explain the meaning of coded integer values (e.g., ECOG 0=fully active, 4=disabled); (c) class hierarchies (SubClassOf) reflecting established domain taxonomies; (d) provenance metadata linking each element to its source table or documentation."
    }
    return criteria.get(metric, "")

def evaluate_metric(metric: str, context: Dict) -> Tuple[float, int]:
    """Score one metric via GPT-5.4 Judge-LLM. Returns (score, approx_tokens)."""
    """
    Score one metric for one ontology chunk.
    Returns (score, approx_tokens_used).
    """
    prompt = f"""You are an ontology evaluation expert. Analyze the ontology fragment below and provide a numerical score (0-5) for the dimension: {metric}.

Evaluation criteria for {metric}:
{get_metric_criteria(metric)}

Competency Questions:
{context['questions']}

Ontology Fragment:
{context['ontology_chunk']}

Database Schema:
{context['schema']}

Return ONLY a single numerical score between 0.0 and 5.0 with one decimal place.
Format: Score: X.X"""

    approx_tokens = len(prompt) // 4  # rough estimate

    response = call_judge_llm(prompt, OPENROUTER_API_KEY)

    patterns = [
        r"Score:\s*([0-5]\.\d)",
        r"\b([0-5]\.\d)\b",
        r"([0-5]\.[0-9])/5",
        r"final score:\s*([0-5]\.\d)",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            return max(0.0, min(5.0, round(score, 2))), approx_tokens

    print(f"  No valid score found for {metric}: {response}")
    return 0.0, approx_tokens

def evaluate_cq_scores(
    chunks:    List[str],
    questions: List[Dict],
    schema_str: str,
) -> Dict:
    """
    Run all 6 metrics across all chunks.
    Returns per-chunk scores, final averages, total tokens, and runtime.
    """
    metrics     = ["accuracy", "completeness", "conciseness",
                   "adaptability", "clarity", "consistency", "domain_enrichment"]
    results     = {m: [] for m in metrics}
    total_tokens = 0
    start_time   = time.time()

    q_text = "\n".join(
        f"Q: {q['question']}\nA: {q['answer']}" for q in questions
    )

    for i, chunk in enumerate(chunks):
        print(f"    Chunk {i+1}/{len(chunks)}")
        context = {
            "questions":     q_text,
            "ontology_chunk": chunk,
            "schema":         schema_str,
        }
        for metric in metrics:
            score, tokens = evaluate_metric(metric, context)
            results[metric].append(score)
            total_tokens += tokens

    elapsed      = time.time() - start_time
    final_scores = {
        m: round(sum(s) / len(s), 2) if s else 0.0
        for m, s in results.items()
    }
    overall = round(sum(final_scores.values()) / len(final_scores), 2)

    return {
        "chunk_scores":  results,
        "final_scores":  final_scores,
        "overall":       overall,
        "runtime_sec":   round(elapsed, 1),
        "approx_tokens": total_tokens,
    }

# =========================================================
# FULL EVALUATION OF ONE ONTOLOGY
# =========================================================

def evaluate_ontology(
    label: str,
    owl_path: str,
    schema: Dict,
    questions: List[Dict],
) -> Dict:
    """Run all evaluation components for a single ontology file."""
    print(f"\n  Evaluating: {label}")
    print(f"  File: {owl_path}")

    result = {
        "label": label,
        "owl_path": owl_path,
        "exists": os.path.exists(owl_path),
    }

    if not result["exists"]:
        print("  Skipped — file not found")
        return result

    # Load graph
    g = load_graph(owl_path)

    # 1. Structural analysis
    print("  Running structural analysis...")
    if g is not None:
        result["structural"] = structural_analysis(g)
    else:
        print("  Structural analysis skipped — ontology could not be parsed")
        result["structural"] = {}

    # 2. Syntax and consistency validation
    print("  Running validation...")
    result["validation"] = validate_ontology(owl_path, g)

    # 3. Semantic coverage
    print("  Running semantic coverage...")
    result["coverage"] = semantic_coverage(g, schema)

    # 4. CQ-based scoring (optional)
    if ENABLE_CQ_JUDGE:
        print(f"  Running CQ evaluation ({len(questions)} questions)...")
        chunks = load_ontology_chunks(g) if g is not None else []
        schema_str = json.dumps(schema)

        if not chunks:
            chunks = ["No ontology content could be extracted — file may be unparseable."]
            print("  Warning: no chunks extracted — using placeholder for CQ eval")

        if questions:
            result["cq_scores"] = evaluate_cq_scores(chunks, questions, schema_str)
        else:
            result["cq_scores"] = {
                "chunk_scores": {},
                "final_scores": {},
                "overall": 0.0,
                "runtime_sec": 0,
                "approx_tokens": 0,
                "skipped": True,
                "reason": "no competency questions found",
            }
            print("  Skipped CQ scoring — no competency questions found")
    else:
        print("  Skipping CQ evaluation — ENABLE_CQ_JUDGE=False")
        result["cq_scores"] = {
            "chunk_scores": {},
            "final_scores": {},
            "overall": 0.0,
            "runtime_sec": 0,
            "approx_tokens": 0,
            "skipped": True,
            "reason": "CQ Judge-LLM disabled",
        }

    return result

# =========================================================
# COMPARISON TABLE BUILDER
# =========================================================

def build_comparison_table(evaluations: List[Dict]) -> Dict:
    """
    Build a structured comparison table across all evaluated ontologies.
    Mirrors the format of Tables 1-3 in the paper.
    """
    table = {}
    for ev in evaluations:
        label = ev["label"]
        row   = {"label": label}

        # Structural
        s = ev.get("structural", {})
        row["classes"]           = s.get("classes", 0)
        row["object_properties"] = s.get("object_properties", 0)
        row["data_properties"]   = s.get("data_properties", 0)
        row["total_axioms"]      = s.get("total_axioms", 0)
        row["labels"]            = s.get("labels", 0)
        row["comments"]          = s.get("comments", 0)
        row["subclass_axioms"]   = s.get("subclass_axioms", 0)
        row["disjoint_axioms"]   = s.get("disjoint_axioms", 0)

        # Validation
        v = ev.get("validation", {})
        row["syntax_valid"]          = v.get("syntax_valid", False)
        row["punning_violations"]    = len(v.get("punning_violations", []))
        row["missing_domain_range"]  = len(v.get("missing_domain_range", []))
        row["self_referential_props"]= len(v.get("self_referential_props", []))
        row["date_typed_as_float"]   = len(v.get("date_typed_as_float", []))

        # Coverage
        c = ev.get("coverage", {})
        row["table_coverage_rate"]  = c.get("table_coverage_rate", 0.0)
        row["column_coverage_rate"] = c.get("column_coverage_rate", 0.0)

        # CQ scores
        cq = ev.get("cq_scores", {}).get("final_scores", {})
        row["cq_accuracy"]      = cq.get("accuracy", 0.0)
        row["cq_completeness"]  = cq.get("completeness", 0.0)
        row["cq_conciseness"]   = cq.get("conciseness", 0.0)
        row["cq_adaptability"]  = cq.get("adaptability", 0.0)
        row["cq_clarity"]       = cq.get("clarity", 0.0)
        row["cq_consistency"]   = cq.get("consistency", 0.0)
        row["cq_enrichment"]   = cq.get("domain_enrichment", 0.0)
        row["cq_overall"]       = ev.get("cq_scores", {}).get("overall", 0.0)

        # Runtime and tokens
        row["runtime_sec"]   = ev.get("cq_scores", {}).get("runtime_sec", 0)
        row["approx_tokens"] = ev.get("cq_scores", {}).get("approx_tokens", 0)

        table[label] = row

    return table

# =========================================================
# MAIN
# =========================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for db_name, schema_path in SCHEMAS.items():
        print(f"\n{'='*65}")
        print(f"DATABASE: {db_name}")
        print(f"{'='*65}")

        # Load schema
        if not os.path.exists(schema_path):
            print(f"Schema not found: {schema_path} — skipping")
            continue
        with open(schema_path, "r") as f:
            schema = json.load(f)

        # Load CQs
        cqs_path  = os.path.join(COMPETENCY_QUESTIONS_DIR, db_name)
        questions = load_competency_questions(cqs_path)

        # Evaluate each ontology for this database
        ontologies    = ONTOLOGIES.get(db_name, {})
        evaluations   = []
        total_runtime = 0.0

        all_coverage = {}   # collected for comparison heatmap
        heatmap_dir  = os.path.join(OUTPUT_DIR, "heatmaps", db_name)

        for label, owl_path in ontologies.items():
            ev = evaluate_ontology(label, owl_path, schema, questions)
            evaluations.append(ev)
            total_runtime += ev.get("cq_scores", {}).get("runtime_sec", 0)

            # Generate per-ontology heatmaps
            if ev.get("exists") and ev.get("coverage"):
                plot_coverage_heatmap(
                    coverage_result = ev["coverage"],
                    ontology_label  = label,
                    db_name         = db_name,
                    output_dir      = heatmap_dir,
                )
                all_coverage[label] = ev["coverage"]

            # Save individual result
            out_file = os.path.join(OUTPUT_DIR, f"{db_name}_{label}.json")
            with open(out_file, "w") as f:
                # Exclude verbose coverage details from individual file
                ev_slim = {k: v for k, v in ev.items() if k != "coverage"}
                ev_slim["coverage_summary"] = {
                    k: v for k, v in ev.get("coverage", {}).items()
                    if k != "details"
                }
                json.dump(ev_slim, f, indent=2)
            print(f"  Saved: {out_file}")

        # Generate cross-ontology comparison heatmap
        if all_coverage:
            plot_comparison_heatmap(
                all_coverage = all_coverage,
                db_name      = db_name,
                output_dir   = heatmap_dir,
            )

        # Build comparison table
        table = build_comparison_table(evaluations)

        # Save summary
        summary_file = os.path.join(OUTPUT_DIR, f"{db_name}_summary.json")
        with open(summary_file, "w") as f:
            json.dump({
                "database":     db_name,
                "total_runtime_sec": round(total_runtime, 1),
                "comparison":   table,
            }, f, indent=2)
        print(f"\nSummary saved: {summary_file}")

        # Print comparison table to console
        print(f"\n--- Comparison Table: {db_name} ---")
        header = f"{'Label':<25} {'Classes':>8} {'ObjP':>6} {'DataP':>6} "
        header += f"{'Axioms':>8} {'Labels':>7} {'TableCov':>9} {'ColCov':>7} "
        header += f"{'CQ_Avg':>7} {'Tokens':>8}"
        print(header)
        print("-" * len(header))
        for label, row in table.items():
            print(
                f"{label:<25} "
                f"{row['classes']:>8} "
                f"{row['object_properties']:>6} "
                f"{row['data_properties']:>6} "
                f"{row['total_axioms']:>8} "
                f"{row['labels']:>7} "
                f"{row['table_coverage_rate']:>9.3f} "
                f"{row['column_coverage_rate']:>7.3f} "
                f"{row['cq_overall']:>7.2f} "
                f"{row['approx_tokens']:>8}"
            )

    print(f"\nAll evaluations complete. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
