"""
ablation_class_property_alignment.py

Evaluate the 5 main ablation variants by comparing ontology classes and
properties against three retrieval resources:
  1. relational schema
  2. external ontologies
  3. documentation

Metric:
  Semantic precision / recall / F1 using cosine similarity >= threshold
  between embeddings from all-MiniLM-L6-v2.

This evaluates class/property alignment directly, not only annotations.
"""

import os
import re
import json
from typing import Dict, Set, List

import pandas as pd
from rdflib import Graph, RDF, OWL, RDFS, URIRef, BNode
from sentence_transformers import SentenceTransformer, util


# =========================================================
# CONFIGURATION
# =========================================================

BASE_PATH = "YOUR_BASE_PATH_HERE"  # <-- UPDATE THIS TO YOUR BASE PATH

SCHEMA_PATH = f"{BASE_PATH}/sql_schema/schema_chinook.json"
EXTERNAL_ONTOLOGY_PATH = f"{BASE_PATH}/external_ontologies_chinook"
DOCS_PATH = f"{BASE_PATH}/documents_chinook"

ABLATION_VARIANTS = {
    "no_rag": f"{BASE_PATH}/ablation_chinook/chinook/no_rag/no_rag_chinook_claude_ontology.owl",
    "only_schema": f"{BASE_PATH}/ablation_chinook/chinook/only_schema_context/only_schema_context_chinook_claude_ontology.owl",
    "only_external": f"{BASE_PATH}/ablation_chinook/chinook/only_external_ontologies/only_external_ontologies_chinook_claude_ontology.owl",
    "only_docs": f"{BASE_PATH}/ablation_chinook/chinook/only_relevant_documents/only_relevant_documents_chinook_claude_ontology.owl",
    "full_rigor": f"{BASE_PATH}/output/RIGOR/chinook/claude/rigor_chinook_claude_ontology.owl",
}

OUTPUT_DIR = f"{BASE_PATH}/evaluation/ablation_class_property_alignment"
SIM_THRESHOLD = 0.55
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# =========================================================
# HELPERS
# =========================================================

def local_name(uri) -> str:
    s = str(uri)
    return s.split("#")[-1].split("/")[-1]


def split_identifier(text: str) -> str:
    """
    Convert identifiers like InvoiceLine, invoice_line, TrackId
    into more embedding-friendly text.
    """
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    text = re.sub(r"\bId\b", "ID", text)
    return text.strip()


def clean_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def unique_clean(items: List[str]) -> List[str]:
    seen = set()
    out = []

    for item in items:
        item = clean_text(item)
        if not item:
            continue

        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)

    return out


# =========================================================
# ONTOLOGY EXTRACTION
# =========================================================

def extract_ontology_class_property_texts(owl_path: str) -> Dict[str, List[str]]:
    """
    Extract class and property names from an ontology.

    Returns:
      {
        "classes": [...],
        "properties": [...],
        "all": [...]
      }

    Uses URI local names and makes them embedding-friendly.
    """
    result = {
        "classes": [],
        "properties": [],
        "all": [],
    }

    if not os.path.exists(owl_path):
        print(f"Missing ontology file: {owl_path}")
        return result

    g = Graph()
    try:
        g.parse(owl_path, format="xml")
    except Exception as e:
        print(f"Parse error for {owl_path}: {e}")
        return result

    classes = []
    properties = []

    for s in g.subjects(RDF.type, OWL.Class):
        if isinstance(s, BNode):
            continue
        name = local_name(s)
        if name:
            classes.append(split_identifier(name))

        for label in g.objects(s, RDFS.label):
            label_text = clean_text(label)
            if label_text:
                classes.append(label_text)

    for prop_type in [OWL.ObjectProperty, OWL.DatatypeProperty]:
        for s in g.subjects(RDF.type, prop_type):
            if isinstance(s, BNode):
                continue
            name = local_name(s)
            if name:
                properties.append(split_identifier(name))

            for label in g.objects(s, RDFS.label):
                label_text = clean_text(label)
                if label_text:
                    properties.append(label_text)

    classes = unique_clean(classes)
    properties = unique_clean(properties)

    result["classes"] = classes
    result["properties"] = properties
    result["all"] = unique_clean(classes + properties)

    return result


# =========================================================
# REFERENCE BUILDERS
# =========================================================

def build_schema_reference(schema_path: str) -> Dict[str, List[str]]:
    """
    Build schema reference from table and column names.
    """
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    tables = []
    columns = []

    for table_name, table_val in schema.items():
        tables.append(split_identifier(table_name))

        cols = table_val.get("columns", table_val) if isinstance(table_val, dict) else table_val
        if isinstance(cols, dict):
            for col_name in cols.keys():
                columns.append(split_identifier(col_name))
                columns.append(split_identifier(f"{table_name} {col_name}"))

    tables = unique_clean(tables)
    columns = unique_clean(columns)

    return {
        "classes": tables,
        "properties": columns,
        "all": unique_clean(tables + columns),
    }


def parse_rdf_file(path: str) -> Graph:
    """
    Parse RDF/OWL/Turtle/NTriples with fallback formats.
    """
    suffix = os.path.splitext(path)[1].lower()

    if suffix == ".ttl":
        formats = ["turtle", "n3", "xml", "nt"]
    elif suffix == ".nt":
        formats = ["nt", "turtle", "xml", "n3"]
    elif suffix == ".n3":
        formats = ["n3", "turtle", "xml", "nt"]
    else:
        formats = ["xml", "pretty-xml", "turtle", "n3", "nt"]

    for fmt in formats:
        g = Graph()
        try:
            g.parse(path, format=fmt)
            return g
        except Exception:
            continue

    return Graph()


def build_external_reference(external_path: str) -> Dict[str, List[str]]:
    """
    Build external ontology reference from class/property local names and labels.
    """
    classes = []
    properties = []

    if not os.path.exists(external_path):
        print(f"Missing external ontology path: {external_path}")
        return {"classes": [], "properties": [], "all": []}

    for root, _, files in os.walk(external_path):
        for fname in files:
            if not fname.endswith((".owl", ".rdf", ".ttl", ".nt", ".n3")):
                continue

            fpath = os.path.join(root, fname)
            g = parse_rdf_file(fpath)

            if len(g) == 0:
                continue

            for s in g.subjects(RDF.type, OWL.Class):
                if isinstance(s, BNode):
                    continue

                name = local_name(s)
                if name:
                    classes.append(split_identifier(name))

                for label in g.objects(s, RDFS.label):
                    label_text = clean_text(label)
                    if label_text:
                        classes.append(label_text)

            for prop_type in [OWL.ObjectProperty, OWL.DatatypeProperty, RDF.Property]:
                for s in g.subjects(RDF.type, prop_type):
                    if isinstance(s, BNode):
                        continue

                    name = local_name(s)
                    if name:
                        properties.append(split_identifier(name))

                    for label in g.objects(s, RDFS.label):
                        label_text = clean_text(label)
                        if label_text:
                            properties.append(label_text)

    classes = unique_clean(classes)
    properties = unique_clean(properties)

    return {
        "classes": classes,
        "properties": properties,
        "all": unique_clean(classes + properties),
    }


def build_docs_reference(docs_path: str) -> Dict[str, List[str]]:
    """
    Build documentation reference from sentences and meaningful text chunks.
    """
    docs = []

    if not os.path.exists(docs_path):
        print(f"Missing docs path: {docs_path}")
        return {"all": []}

    for root, _, files in os.walk(docs_path):
        for fname in files:
            fpath = os.path.join(root, fname)

            try:
                if fname.endswith(".docx"):
                    from docx import Document
                    doc = Document(fpath)
                    text = "\n".join(p.text for p in doc.paragraphs)

                elif fname.endswith((".txt", ".md", ".csv", ".json")):
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()

                else:
                    continue

            except Exception as e:
                print(f"Skipped {fpath}: {e}")
                continue

            lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 5]

            for line in lines:
                parts = re.split(r"[.;:!?]\s+", line)
                for part in parts:
                    part = clean_text(part)
                    if 5 <= len(part) <= 300:
                        docs.append(part)

    docs = unique_clean(docs)

    return {"all": docs}


# =========================================================
# SEMANTIC METRICS
# =========================================================

def semantic_prf(
    candidates: List[str],
    references: List[str],
    model: SentenceTransformer,
    threshold: float = SIM_THRESHOLD,
    max_refs: int = 10000,
) -> Dict[str, float]:
    """
    Compute semantic precision, recall, and F1.

    Precision:
      proportion of ontology elements that match at least one reference.

    Recall:
      proportion of reference elements matched by at least one ontology element.

    F1:
      harmonic mean of precision and recall.
    """
    candidates = unique_clean(candidates)
    references = unique_clean(references)

    if not candidates or not references:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "avg_max_sim": 0.0,
        }

    if len(references) > max_refs:
        references = references[:max_refs]

    cand_emb = model.encode(candidates, convert_to_tensor=True, show_progress_bar=False)
    ref_emb = model.encode(references, convert_to_tensor=True, show_progress_bar=False)

    sim = util.cos_sim(cand_emb, ref_emb)

    cand_best = sim.max(dim=1).values
    ref_best = sim.max(dim=0).values

    precision = float((cand_best >= threshold).sum().item()) / len(candidates)
    recall = float((ref_best >= threshold).sum().item()) / len(references)

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "avg_max_sim": round(float(cand_best.mean().item()), 4),
    }


def evaluate_variant(
    variant_name: str,
    owl_path: str,
    schema_ref: Dict[str, List[str]],
    external_ref: Dict[str, List[str]],
    docs_ref: Dict[str, List[str]],
    model: SentenceTransformer,
) -> Dict:
    """
    Evaluate one ontology variant against all three references.
    """
    elems = extract_ontology_class_property_texts(owl_path)

    ontology_all = elems["all"]

    schema_metrics = semantic_prf(ontology_all, schema_ref["all"], model)
    external_metrics = semantic_prf(ontology_all, external_ref["all"], model)
    docs_metrics = semantic_prf(ontology_all, docs_ref["all"], model)

    avg_f1 = round(
        (
            schema_metrics["f1"]
            + external_metrics["f1"]
            + docs_metrics["f1"]
        ) / 3,
        4,
    )

    avg_precision = round(
        (
            schema_metrics["precision"]
            + external_metrics["precision"]
            + docs_metrics["precision"]
        ) / 3,
        4,
    )

    avg_recall = round(
        (
            schema_metrics["recall"]
            + external_metrics["recall"]
            + docs_metrics["recall"]
        ) / 3,
        4,
    )

    return {
        "Variant": variant_name,
        "N_Classes": len(elems["classes"]),
        "N_Properties": len(elems["properties"]),
        "N_Elements": len(ontology_all),

        "Schema_P": schema_metrics["precision"],
        "Schema_R": schema_metrics["recall"],
        "Schema_F1": schema_metrics["f1"],

        "External_P": external_metrics["precision"],
        "External_R": external_metrics["recall"],
        "External_F1": external_metrics["f1"],

        "Docs_P": docs_metrics["precision"],
        "Docs_R": docs_metrics["recall"],
        "Docs_F1": docs_metrics["f1"],

        "Avg_P": avg_precision,
        "Avg_R": avg_recall,
        "Avg_F1": avg_f1,
    }


# =========================================================
# LATEX OUTPUT
# =========================================================

def print_latex_table(df: pd.DataFrame):
    """
    Print compact LaTeX table for the paper.
    """
    cols = [
        "Variant",
        "Schema_F1",
        "External_F1",
        "Docs_F1",
        "Avg_F1",
        "N_Classes",
        "N_Properties",
        "N_Elements",
    ]

    paper_df = df[cols].copy()

    print("\n--- LaTeX Table ---")
    print(r"\begin{tabular}{lccccrrr}")
    print(r"\toprule")
    print(
        r"\textbf{Variant} & \textbf{Schema} & \textbf{External} & "
        r"\textbf{Docs} & \textbf{Avg.} & \textbf{Cls} & "
        r"\textbf{Prop.} & \textbf{Elems} \\"
    )
    print(r"\midrule")

    best_schema = paper_df["Schema_F1"].max()
    best_external = paper_df["External_F1"].max()
    best_docs = paper_df["Docs_F1"].max()
    best_avg = paper_df["Avg_F1"].max()

    for _, row in paper_df.iterrows():
        variant = row["Variant"].replace("_", r"\_")

        def fmt(value, best):
            value = float(value)
            text = f"{value:.4f}"
            if abs(value - best) < 1e-9:
                return r"\textbf{" + text + "}"
            return text

        print(
            f"{variant} & "
            f"{fmt(row['Schema_F1'], best_schema)} & "
            f"{fmt(row['External_F1'], best_external)} & "
            f"{fmt(row['Docs_F1'], best_docs)} & "
            f"{fmt(row['Avg_F1'], best_avg)} & "
            f"{int(row['N_Classes'])} & "
            f"{int(row['N_Properties'])} & "
            f"{int(row['N_Elements'])} \\\\"
        )

    print(r"\bottomrule")
    print(r"\end{tabular}")


# =========================================================
# MAIN
# =========================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Building references...")
    schema_ref = build_schema_reference(SCHEMA_PATH)
    external_ref = build_external_reference(EXTERNAL_ONTOLOGY_PATH)
    docs_ref = build_docs_reference(DOCS_PATH)

    print(f"Schema reference: {len(schema_ref['all'])} elements")
    print(f"External reference: {len(external_ref['all'])} elements")
    print(f"Docs reference: {len(docs_ref['all'])} units")

    rows = []

    for variant_name, owl_path in ABLATION_VARIANTS.items():
        print(f"\nEvaluating {variant_name}")
        row = evaluate_variant(
            variant_name,
            owl_path,
            schema_ref,
            external_ref,
            docs_ref,
            model,
        )
        rows.append(row)

    df = pd.DataFrame(rows)

    output_csv = os.path.join(OUTPUT_DIR, "chinook_ablation_class_property_alignment.csv")
    df.to_csv(output_csv, index=False)

    print("\n--- Results ---")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)
    print(df.to_string(index=False))

    print(f"\nSaved CSV: {output_csv}")

    print_latex_table(df)


if __name__ == "__main__":
    main()