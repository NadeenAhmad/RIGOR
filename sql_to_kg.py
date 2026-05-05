"""
sql_to_kg.py — Real Data KG Population from SQL Dump
=====================================================
Reads the real clinical database SQL dump (MariaDB/MySQL format),
extracts INSERT data for all tables, and populates the RIGOR-enriched
ontologies with real patient instances.

Supports 2 schemas × 3 LLMs = 6 ontologies.

Input:
    clinical_database_21_02_2025.sql   — real patient data dump
    output/<schema>/<model>/enriched_ontology.owl  — RIGOR ontologies

Output (per job):
    kg_output/<schema>/<model>/populated_kg.ttl
    kg_output/<schema>/<model>/stats.json
    kg_output/<schema>/<model>/sparql_results.json

Usage:
    python sql_to_kg.py
    python sql_to_kg.py --schema schema1          # one schema only
    python sql_to_kg.py --model claude            # one model only
"""

import os
import re
import json
import time
import argparse
import logging
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote
from urllib.parse import urlparse
from rdflib import Graph, Namespace, URIRef, RDF, RDFS, OWL, XSD, Literal

try:
    from thefuzz import fuzz
    _FUZZ = True
except ImportError:
    _FUZZ = False

import sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,   # Colab shows stdout more reliably than stderr
)
# Force flush on every log line so Colab doesn't buffer output
logging.getLogger().handlers[0].flush = lambda: sys.stdout.flush()
log = logging.getLogger(__name__)
# line_buffering not needed — flush=True on print() calls handles this in Colab

# =========================================================
# CONFIGURATION  — edit paths to match your layout
# =========================================================

# ── Colab path configuration ─────────────────────────────────────────────────
# All files are uploaded to the same flat working directory in Colab.
# BASE_PATH points to that directory — adjust if your Colab layout differs.
BASE_PATH = "YOUR_BASE_PATH"   # default Colab working directory

OUTPUT_DIR = os.path.join(BASE_PATH, "output/kg_output")


# ── Schema 1: real-world clinical data (SQL dump) ────────────────────────────
#SQL_DUMP = os.path.join(BASE_PATH, "real_data/clinical_database_21_02_2025.sql")

# ── Schema 2: eICU synthetic data (CSV files) ────────────────────────────────
# All CSVs are in the same flat directory as the SQL file.
#CSV_DIR_SCHEMA2 = "YOUR_BASE_PATH/synthetic_data/eicu_crd"

CHINOOK_JSON = os.path.join(BASE_PATH, "real_data/ChinookData.json")

# Schema JSON files (flat directory)
SCHEMAS = {
   # "schema1": os.path.join(BASE_PATH, "sql_schema/schema_rd.json"),
  # "schema2": os.path.join(BASE_PATH, "sql_schema/schema_icu.json"),
    "chinook": os.path.join(BASE_PATH, "sql_schema/schema_chinook.json"),
}

# Data source type per schema: "sql" or "csv"
SCHEMA_DATA_TYPE = {
  # "schema1": "sql",
  # "schema2": "csv",
   "chinook": "json",
   
}

# Ontology files — matching the RIGOR folder structure from the images:
#   RIGOR/real_world/<model>/enriched_ontology.owl   (schema 1)
#   RIGOR/eicu_crd/<model>/enriched_ontology.owl     (schema 2)
ONTOLOGIES = {
   # "schema1": {
   #     "claude":   os.path.join(BASE_PATH, "output", "RIGOR", "real_world", "claude",   "enriched_ontology.owl"),
   #     "mistral":  os.path.join(BASE_PATH, "output", "RIGOR", "real_world", "mistral",  "enriched_ontology.owl"),
   #     "deepseek": os.path.join(BASE_PATH, "output", "RIGOR", "real_world", "deepseek", "enriched_ontology.owl"),
   # },
   # "schema2": {
   #     "claude":   os.path.join(BASE_PATH, "output",  "RIGOR", "eicu_crd",   "claude",   "enriched_ontology.owl"),
    #    "mistral":  os.path.join(BASE_PATH, "output", "RIGOR", "eicu_crd",   "mistral",  "enriched_ontology.owl"),
    #    "deepseek": os.path.join(BASE_PATH, "output", "RIGOR", "eicu_crd",   "deepseek", "enriched_ontology.owl"),
   # },
    

        "chinook": {
        "claude":   os.path.join(BASE_PATH, "output", "RIGOR", "chinook", "claude",   "enriched_ontology.owl"),
        "mistral":  os.path.join(BASE_PATH, "output", "RIGOR", "chinook", "mistral",  "enriched_ontology.owl"),
        "deepseek": os.path.join(BASE_PATH, "output", "RIGOR", "chinook", "deepseek", "enriched_ontology.owl"),
    },
}

ONTOLOGY_BASE = "http://example.org/ontology#"
INSTANCE_BASE = "http://example.org/instance/"

# FK relationships present in the DB but missing from schema JSON.
# Add more here if you discover additional gaps.
IMPLICIT_FKS: Dict[str, List[Tuple[str, str, str]]] = {
    # table: [(fk_column, referenced_table, referenced_pk_column), ...]
    "chemotherapy": [("patient_id", "patient_data", "patient_id")],
}

# Minimum fuzzy-match score (0–100) to accept a class or property match.
# Raise if you get false positives; lower if ontology names diverge from schema.
CLASS_MATCH_THRESHOLD    = 60
PROPERTY_MATCH_THRESHOLD = 55


# =========================================================
# SQL DUMP PARSER  (original logic — kept intact, it works)
# =========================================================
def load_json_tables(json_path: str) -> Dict[str, Dict]:
    """
    Load Chinook JSON data into the same format used by SQL/CSV loaders:
    {table_name: {"columns": [...], "rows": [row_dict, ...]}}
    Expected JSON shape:
      {
        "Artist": [{"ArtistId": 1, "Name": "AC/DC"}, ...],
        "Album":  [{"AlbumId": 1, "Title": "...", "ArtistId": 1}, ...]
      }
    """
    log.info("Loading JSON data: %s", json_path)
    tables: Dict[str, Dict] = {}

    if not os.path.exists(json_path):
        log.error("JSON file not found: %s", json_path)
        return tables

    with open(json_path, "r", encoding="utf-8", errors="replace") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        log.error("Expected top-level JSON object mapping table names to row lists")
        return tables

    for table_name, rows in raw.items():
        if not isinstance(rows, list):
            log.warning("Skipping %s: expected list of rows", table_name)
            continue

        columns = []
        seen = set()
        for row in rows:
            if isinstance(row, dict):
                for col in row.keys():
                    if col not in seen:
                        seen.add(col)
                        columns.append(col)

        cleaned_rows = [r for r in rows if isinstance(r, dict)]

        tables[table_name] = {
            "columns": columns,
            "rows": cleaned_rows,
        }

        log.info(
            "  %-35s %d rows, %d columns",
            table_name,
            len(cleaned_rows),
            len(columns),
        )

    return tables


def parse_sql_dump(sql_path: str) -> Dict[str, Dict]:
    """
    Parse a MariaDB/MySQL SQL dump and extract all INSERT data.
    Returns {table_name: {"columns": [...], "rows": [row_dict, ...]}}
    """
    log.info("Parsing SQL dump: %s", sql_path)
    tables: Dict[str, Dict] = {}

    with open(sql_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Extract column names from CREATE TABLE blocks
    create_pattern = re.compile(
        r"CREATE TABLE.*?`(\w+)`\s*\((.*?)\)\s*ENGINE",
        re.DOTALL | re.IGNORECASE,
    )
    for m in create_pattern.finditer(content):
        table_name = m.group(1)
        col_block  = m.group(2)
        cols = []
        for line in col_block.splitlines():
            line = line.strip()
            cm = re.match(r"`(\w+)`\s+(\w+)", line)
            if cm and not re.match(
                r"(PRIMARY|KEY|UNIQUE|CONSTRAINT|FULLTEXT|INDEX)",
                line.upper(),
            ):
                cols.append(cm.group(1))
        tables[table_name] = {"columns": cols, "rows": []}

    # Extract INSERT INTO data
    insert_pattern = re.compile(
        r"INSERT INTO `(\w+)` \(([^)]+)\) VALUES\s*(.*?);",
        re.DOTALL | re.IGNORECASE,
    )
    for m in insert_pattern.finditer(content):
        table_name = m.group(1)
        col_list   = [c.strip().strip("`") for c in m.group(2).split(",")]
        values_str = m.group(3)
        if table_name not in tables:
            tables[table_name] = {"columns": col_list, "rows": []}
        tables[table_name]["rows"].extend(parse_values(values_str, col_list))

    for name, data in tables.items():
        log.info("  %-35s %d rows, %d columns",
                 name, len(data["rows"]), len(data["columns"]))

    # Patch implicit FKs into the in-memory table metadata
    # (the SQL parser doesn't read FK constraints, that comes from schema JSON)
    return tables


def parse_values(values_str: str, columns: List[str]) -> List[Dict]:
    row_pattern = re.compile(r"\(([^()]*(?:'[^']*'[^()]*)*)\)")
    rows = []
    for rm in row_pattern.finditer(values_str):
        values = split_row_values(rm.group(1))
        if len(values) == len(columns):
            rows.append(dict(zip(columns, values)))
    return rows


def split_row_values(raw: str) -> List[Optional[str]]:
    values, current, in_quote, i = [], "", False, 0
    while i < len(raw):
        ch = raw[i]
        if ch == "'" and not in_quote:
            in_quote = True
            current += ch
        elif ch == "'" and in_quote:
            if i + 1 < len(raw) and raw[i + 1] == "'":
                current += "'"
                i += 2
                continue
            in_quote = False
            current += ch
        elif ch == "," and not in_quote:
            values.append(clean_value(current.strip()))
            current = ""
        else:
            current += ch
        i += 1
    if current.strip():
        values.append(clean_value(current.strip()))
    return values


def clean_value(v: str) -> Optional[str]:
    if v.upper() == "NULL":
        return None
    if v.startswith("'") and v.endswith("'"):
        inner = v[1:-1]
        inner = inner.replace("\\'", "'").replace("\\\\", "\\")
        inner = inner.replace("\\r\\n", " ").replace("\\n", " ").replace("\\r", " ")
        return inner
    return v



# =========================================================
# CSV LOADER  — Synthea synthetic data for schema 2
# =========================================================

def load_csv_folder(csv_dir: str) -> Dict[str, Dict]:
    """
    Load all CSV files from *csv_dir* into the same
    {table_name: {"columns": [...], "rows": [row_dict, ...]}} format
    that parse_sql_dump() produces, so the rest of the pipeline is
    identical regardless of data source.

    Each CSV file is expected to be named <table_name>.csv.
    Synthea generates one file per eICU table.
    """
    import csv as csv_mod
    log.info("Loading CSV folder: %s", csv_dir)
    tables: Dict[str, Dict] = {}

    if not os.path.isdir(csv_dir):
        log.error("CSV directory not found: %s", csv_dir)
        return tables

    # When CSV files share a directory with other files (e.g. Colab flat layout),
    # only load files whose stem matches a table in the schema JSON if provided,
    # otherwise load all CSVs found.
    csv_files = sorted(f for f in os.listdir(csv_dir) if f.endswith(".csv"))
    if not csv_files:
        log.warning("No CSV files found in %s", csv_dir)
        return tables

    for fname in csv_files:
        table_name = fname[:-4]   # strip .csv
        fpath = os.path.join(csv_dir, fname)
        rows = []
        columns = []
        try:
            with open(fpath, encoding="utf-8", errors="replace", newline="") as f:
                reader = csv_mod.DictReader(f)
                columns = list(reader.fieldnames or [])
                for row in reader:
                    # Convert empty strings to None to match SQL NULL behaviour
                    cleaned = {
                        k: (None if v == "" else v)
                        for k, v in row.items()
                    }
                    rows.append(cleaned)
        except Exception as e:
            log.warning("Could not read %s: %s", fpath, e)
            continue

        tables[table_name] = {"columns": columns, "rows": rows}
        log.info("  %-35s %d rows, %d columns",
                 table_name, len(rows), len(columns))

    return tables


# =========================================================
# SCHEMA LOADING
# =========================================================

SQL_TO_XSD = {
    "INTEGER":   XSD.integer,  "INT":      XSD.integer,
    "SMALLINT":  XSD.integer,  "BIGINT":   XSD.integer,
    "TINYINT":   XSD.boolean,  "BOOLEAN":  XSD.boolean,
    "FLOAT":     XSD.float,    "DOUBLE":   XSD.double,
    "DECIMAL":   XSD.decimal,
    "TEXT":      XSD.string,   "VARCHAR":  XSD.string,
    "CHAR":      XSD.string,   "TIMESTAMP": XSD.dateTime,
    "DATETIME":  XSD.dateTime, "DATE":     XSD.date,
}


def get_xsd(sql_type: str):
    base = str(sql_type).upper().split("(")[0].strip()
    return SQL_TO_XSD.get(base, XSD.string)


def to_class_name(name: str) -> str:
    return "".join(p.capitalize() for p in name.split("_"))


def parse_schema(schema: Dict):
    """
    Returns:
        col_types:  {table: {col: xsd_type}}
        pk_index:   {table: pk_col_name}
        fk_index:   {table: {fk_col: (ref_table, ref_col)}}
    """
    col_types, pk_index, fk_index = {}, {}, {}

    for table_name, table_val in schema.items():
        if isinstance(table_val, dict) and "columns" in table_val:
            cols = table_val["columns"]
            fks  = {
                fk["column"]: (fk["references_table"], fk["references_column"])
                for fk in table_val.get("foreign_keys", [])
            }
        else:
            cols = table_val
            fks  = {}

        # Patch implicit / missing FKs
        for col, ref_table, ref_pk in IMPLICIT_FKS.get(table_name, []):
            if col not in fks:
                fks[col] = (ref_table, ref_pk)
                log.info("Patched implicit FK: %s.%s → %s.%s",
                         table_name, col, ref_table, ref_pk)

        col_types[table_name] = {col: get_xsd(dtype) for col, dtype in cols.items()}
        fk_index[table_name]  = fks

        col_names = list(cols.keys())
        candidates = ["id", f"{table_name}_id", f"{table_name.rstrip('s')}_id"]
        pk_index[table_name] = next(
            (c for c in candidates if c in col_names), col_names[0]
        )

    return col_types, pk_index, fk_index


# =========================================================
# ONTOLOGY LOADING  — FIX: Manchester Syntax support
# =========================================================

from urllib.parse import urlparse

def _is_safe_uri(node) -> bool:
    """
    Reject malformed URIRefs that break Turtle serialization.
    """
    if not isinstance(node, URIRef):
        return True

    s = str(node).strip()

    # obvious bad characters / leftovers from broken parsing
    if any(ch in s for ch in [' ', '\n', '\t', '<', '>', '"', "'", ',']):
        return False

    parsed = urlparse(s)
    if not parsed.scheme:
        return False

    return True

def sanitize_graph(g: Graph) -> Graph:
    """
    Remove any triple where subject, predicate, or object is an invalid
    URIRef (contains spaces). These come from OWL restriction nodes that
    RIGOR writes incorrectly into the RDF/XML file. Without this step
    rdflib crashes when serializing to Turtle.
    Returns a clean Graph with invalid triples removed.
    """
    bad = [(s, p, o) for s, p, o in g
           if not _is_safe_uri(s) or not _is_safe_uri(p) or not _is_safe_uri(o)]

    if not bad:
        return g

    clean = Graph()
    for prefix, ns in g.namespaces():
        clean.bind(prefix, ns)
    for s, p, o in g:
        if _is_safe_uri(s) and _is_safe_uri(p) and _is_safe_uri(o):
            clean.add((s, p, o))

    log.info(
        "Sanitized ontology: removed %d invalid URI triples, %d valid triples kept",
        len(bad), len(clean)
    )
    return clean


def load_ontology(onto_path: str) -> Graph:
    """
    Load an OWL ontology from *onto_path*.

    Tries formats in priority order:
      1. RDF/XML  (most common RIGOR .owl output)
      2. Turtle
      3. N3
      4. Manchester Syntax (parsed manually - rdflib has no built-in support)

    After loading by any method, sanitize_graph() strips triples whose
    subject/predicate/object is an invalid URIRef (OWL cardinality restriction
    nodes serialized as strings like "hasPatient exactly 1 Patient").
    Without sanitization rdflib crashes at Turtle serialization time.
    """
    g = Graph()

    # Try rdflib-native formats first
    for fmt in ("xml", "turtle", "n3", "nt"):
        try:
            g.parse(onto_path, format=fmt)
            if len(g) > 0:
                log.info("Loaded ontology via rdflib (%s): %d triples", fmt, len(g))
                return sanitize_graph(g)
        except Exception:
            continue

    # Fall back to our Manchester Syntax parser
    log.info("rdflib formats failed - attempting Manchester Syntax parser")
    try:
        g = parse_manchester_to_graph(onto_path)
        log.info("Loaded ontology via Manchester parser: %d triples", len(g))
        return sanitize_graph(g)
    except Exception as e:
        log.error("Manchester parser also failed: %s", e)
        return Graph()


def parse_manchester_to_graph(path: str) -> Graph:
    """
    Minimal Manchester Syntax → rdflib Graph conversion.
    Extracts Classes, ObjectProperties, DataProperties, and rdfs:labels.
    This is sufficient for build_ontology_index() to work correctly.
    """
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Extract prefixes
    prefixes = {
        "owl":  "http://www.w3.org/2002/07/owl#",
        "rdf":  "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "xsd":  "http://www.w3.org/2001/XMLSchema#",
    }
    for m in re.finditer(r"^Prefix:\s*(\w*):\s*<([^>]+)>", text, re.MULTILINE):
        prefixes[m.group(1)] = m.group(2)

    # Detect base IRI
    base_m = re.search(r"^Ontology:\s*<([^>]+)>", text, re.MULTILINE)
    base_iri = base_m.group(1) if base_m else "http://example.org/ontology#"

    def expand(token: str) -> Optional[str]:
        """
        Expand a prefixed name or full IRI to an absolute URI string.
        Skip complex OWL expressions safely.
        """
        token = token.strip()

        # remove trailing punctuation often left by Manchester syntax parsing
        token = token.rstrip(",;")

        # skip complex expressions
        if " " in token or "(" in token:
            return None

        # full IRI in angle brackets
        if token.startswith("<") and token.endswith(">"):
            return token[1:-1].strip()

        # tolerate malformed <...>, or <...>;
        if token.startswith("<"):
            token = token[1:]
        if token.endswith(">"):
            token = token[:-1]
        token = token.rstrip(",;").strip()

        if token.startswith("http://") or token.startswith("https://"):
            return token

        if ":" in token:
            pfx, local = token.split(":", 1)
            if pfx in prefixes:
                return prefixes[pfx] + local

        return base_iri.rstrip("#/") + "#" + token

    g = Graph()
    ONT  = Namespace(base_iri.rstrip("#/") + "#")
    g.bind("onto", ONT)

    # Split into entity blocks
    block_pat = re.compile(
        r"^(Class|ObjectProperty|DataProperty|AnnotationProperty):\s*(.+?)$",
        re.MULTILINE,
    )
    block_starts = list(block_pat.finditer(text))

    TYPE_MAP = {
        "Class":              OWL.Class,
        "ObjectProperty":     OWL.ObjectProperty,
        "DataProperty":       OWL.DatatypeProperty,
        "AnnotationProperty": OWL.AnnotationProperty,
    }

    for i, m in enumerate(block_starts):
        entity_type  = m.group(1)
        entity_token = m.group(2).strip()
        block_end    = block_starts[i + 1].start() if i + 1 < len(block_starts) else len(text)
        block        = text[m.start():block_end]

        uri = expand(entity_token)
        if uri is None:
            continue
        subj = URIRef(uri)

        owl_type = TYPE_MAP.get(entity_type)
        if owl_type:
            g.add((subj, RDF.type, owl_type))

        # rdfs:label annotations
        for lm in re.finditer(r'rdfs:label\s+"((?:[^"\\]|\\.)*)"', block):
            g.add((subj, RDFS.label, Literal(lm.group(1))))

        # rdfs:comment annotations
        for cm in re.finditer(r'rdfs:comment\s+"((?:[^"\\]|\\.)*)"', block):
            g.add((subj, RDFS.comment, Literal(cm.group(1))))

        # Domain / Range (object & data properties)
        for dm in re.finditer(r"Domain:\s*(.+?)$", block, re.MULTILINE):
            dom_uri = expand(dm.group(1).strip())
            if dom_uri:
                g.add((subj, URIRef("http://www.w3.org/2000/01/rdf-schema#domain"),
                       URIRef(dom_uri)))
        for rm in re.finditer(r"Range:\s*(.+?)$", block, re.MULTILINE):
            rng_uri = expand(rm.group(1).strip())
            if rng_uri:
                g.add((subj, URIRef("http://www.w3.org/2000/01/rdf-schema#range"),
                       URIRef(rng_uri)))

        # SubClassOf
        for sm in re.finditer(r"SubClassOf:\s*(.+?)$", block, re.MULTILINE):
            parent_uri = expand(sm.group(1).strip())
            if parent_uri:
                g.add((subj, RDFS.subClassOf, URIRef(parent_uri)))

    return g


# =========================================================
# ONTOLOGY INTROSPECTION + MATCHING
# =========================================================

def _normalise(s: str) -> str:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    return re.sub(r"[_\-]+", " ", s).lower().strip()


def _similarity(a: str, b: str) -> float:
    na, nb = _normalise(a), _normalise(b)
    if _FUZZ:
        return max(fuzz.ratio(na, nb),
                   fuzz.token_sort_ratio(na, nb),
                   fuzz.partial_ratio(na, nb))
    return SequenceMatcher(None, na, nb).ratio() * 100


def build_ontology_index(g: Graph):
    """
    Index all classes and properties in the ontology by local name and label.
    Returns (class_index, dataprop_index, objprop_index)
    where each index maps lowercase name/label → URIRef.
    """
    class_index, dataprop_index, objprop_index = {}, {}, {}

    for cls in g.subjects(RDF.type, OWL.Class):
        name = str(cls).split("#")[-1].split("/")[-1]
        class_index[name.lower()] = cls
        for lbl in g.objects(cls, RDFS.label):
            class_index[str(lbl).lower()] = cls

    for prop in g.subjects(RDF.type, OWL.DatatypeProperty):
        name = str(prop).split("#")[-1].split("/")[-1]
        dataprop_index[name.lower()] = prop
        for lbl in g.objects(prop, RDFS.label):
            dataprop_index[str(lbl).lower()] = prop

    for prop in g.subjects(RDF.type, OWL.ObjectProperty):
        name = str(prop).split("#")[-1].split("/")[-1]
        objprop_index[name.lower()] = prop
        for lbl in g.objects(prop, RDFS.label):
            objprop_index[str(lbl).lower()] = prop

    return class_index, dataprop_index, objprop_index


def find_class(class_index: Dict, table_name: str) -> Optional[URIRef]:
    """
    Find the ontology class for *table_name*.
    Strategy: exact match first, then fuzzy fallback.
    """
    camel = to_class_name(table_name).lower()
    # 1. Exact match on CamelCase or raw name
    result = class_index.get(camel) or class_index.get(table_name.lower())
    if result:
        return result
    # 2. Fuzzy fallback
    best_uri, best_score = None, 0.0
    for candidate_name, uri in class_index.items():
        score = _similarity(table_name, candidate_name)
        if score > best_score:
            best_score, best_uri = score, uri
    if best_score >= CLASS_MATCH_THRESHOLD:
        log.debug("  Fuzzy class match: %s → score %.1f", table_name, best_score)
        return best_uri
    return None


def find_dataprop(dataprop_index: Dict, col_name: str) -> Optional[URIRef]:
    """Exact then fuzzy match for a data property."""
    result = dataprop_index.get(col_name.lower())
    if result:
        return result
    best_uri, best_score = None, 0.0
    for candidate, uri in dataprop_index.items():
        score = _similarity(col_name, candidate)
        if score > best_score:
            best_score, best_uri = score, uri
    if best_score >= PROPERTY_MATCH_THRESHOLD:
        return best_uri
    return None


def find_objprop(objprop_index: Dict, fk_col: str, ref_table: str) -> URIRef:
    """
    Find the object property for a FK relationship.
    Tries fk_col name and has<RefTable> pattern, then falls back to a
    generated URI so no link is ever silently dropped.
    """
    camel_ref = f"has{to_class_name(ref_table)}".lower()
    result = (
        objprop_index.get(camel_ref)
        or objprop_index.get(fk_col.lower())
    )
    if result:
        return result
    # Fuzzy
    best_uri, best_score = None, 0.0
    for candidate, uri in objprop_index.items():
        score = max(_similarity(fk_col, candidate),
                    _similarity(camel_ref, candidate))
        if score > best_score:
            best_score, best_uri = score, uri
    if best_score >= PROPERTY_MATCH_THRESHOLD and best_uri:
        return best_uri
    # Fallback: mint a URI so the FK link is never lost
    return URIRef(f"{ONTOLOGY_BASE}has{to_class_name(ref_table)}")


def make_uri(table_name: str, row_id) -> URIRef:
    return URIRef(f"{INSTANCE_BASE}{table_name}/{quote(str(row_id), safe='')}")


def cast_literal(value, xsd_type) -> Optional[Literal]:
    if value is None or str(value).strip() == "":
        return None
    v = str(value).strip()
    try:
        if xsd_type == XSD.integer:
            return Literal(int(float(v)), datatype=xsd_type)
        if xsd_type in (XSD.float, XSD.double, XSD.decimal):
            return Literal(float(v), datatype=xsd_type)
        if xsd_type == XSD.boolean:
            return Literal(v in ("1", "true", "True"), datatype=xsd_type)
        if xsd_type in (XSD.date, XSD.dateTime):
            return Literal(v.replace(" ", "T"), datatype=xsd_type)
        return Literal(v, datatype=XSD.string)
    except Exception:
        return Literal(v, datatype=XSD.string)


# =========================================================
# KG POPULATION  (two-pass: data props → FK object props)
# =========================================================

def populate(
    onto_path:  str,
    sql_data:   Dict,
    schema:     Dict,
    model_name: str,
) -> Tuple[Graph, Dict]:

    log.info("Loading ontology (%s)...", model_name)
    onto = load_ontology(onto_path)
    if len(onto) == 0:
        log.error("Ontology loaded but contains 0 triples — check file format.")
        return Graph(), {"error": "empty ontology"}

    kg = Graph()
    kg.bind("onto", Namespace(ONTOLOGY_BASE))
    kg.bind("inst", Namespace(INSTANCE_BASE))
    kg.bind("owl",  OWL)
    kg.bind("rdf",  RDF)
    kg.bind("rdfs", RDFS)
    kg.bind("xsd",  XSD)
    kg += onto   # embed schema triples in the KG

    class_index, dataprop_index, objprop_index = build_ontology_index(onto)
    col_types, pk_index, fk_index = parse_schema(schema)

    stats = {
        "model":             model_name,
        "ontology_triples":  len(onto),
        "tables_populated":  0,
        "rows_populated":    0,
        "triples_added":     0,
        "fk_links":          0,
        "classes_matched":   0,
        "classes_unmatched": [],
        "data_assertions":   0,
    }

    id_registry: Dict[str, Dict[str, URIRef]] = {}

    # ── Pass 1: Individuals + data properties ─────────────────────────── #
    for table_name, col_type_map in col_types.items():
        table_data = sql_data.get(table_name, {})
        rows       = table_data.get("rows", [])
        if not rows:
            log.info("  %-35s no data rows — skipping", table_name)
            continue

        cls_uri = find_class(class_index, table_name)
        if cls_uri is None:
            log.warning("  No class for '%s' — skipping", table_name)
            stats["classes_unmatched"].append(table_name)
            continue

        stats["classes_matched"] += 1
        pk_col  = pk_index.get(table_name, "id")
        fk_cols = fk_index.get(table_name, {})
        id_registry[table_name] = {}
        n_before = len(kg)

        for row in rows:
            pk_val = row.get(pk_col)
            if pk_val is None:
                continue
            ind_uri = make_uri(table_name, pk_val)
            id_registry[table_name][str(pk_val)] = ind_uri

            kg.add((ind_uri, RDF.type, cls_uri))
            kg.add((ind_uri, RDF.type, OWL.NamedIndividual))

            for col_name, xsd_type in col_type_map.items():
                if col_name == pk_col or col_name in fk_cols:
                    continue
                val = row.get(col_name)
                if val is None:
                    continue
                prop_uri = (
                    find_dataprop(dataprop_index, col_name)
                    or URIRef(f"{ONTOLOGY_BASE}{col_name}")   # fallback: preserve data
                )
                lit = cast_literal(val, xsd_type)
                if lit is not None:
                    kg.add((ind_uri, prop_uri, lit))
                    stats["data_assertions"] += 1

        added = len(kg) - n_before
        stats["triples_added"]  += added
        stats["rows_populated"] += len(rows)
        stats["tables_populated"] += 1
        log.info("  %-35s %d rows → %d triples", table_name, len(rows), added)

    # ── Pass 2: FK object properties ──────────────────────────────────── #
    log.info("  Adding FK links...")
    for table_name, fk_cols in fk_index.items():
        if not fk_cols:
            continue
        table_data = sql_data.get(table_name, {})
        rows       = table_data.get("rows", [])
        pk_col     = pk_index.get(table_name, "id")

        for row in rows:
            pk_val  = row.get(pk_col)
            ind_uri = id_registry.get(table_name, {}).get(str(pk_val))
            if ind_uri is None:
                continue

            for fk_col, (ref_table, ref_col) in fk_cols.items():
                fk_val  = row.get(fk_col)
                ref_uri = id_registry.get(ref_table, {}).get(str(fk_val))
                if ref_uri is None:
                    continue
                prop_uri = find_objprop(objprop_index, fk_col, ref_table)
                kg.add((ind_uri, prop_uri, ref_uri))
                stats["fk_links"] += 1

    stats["total_triples"] = len(kg)
    log.info("  Total KG triples: %d", len(kg))
    return kg, stats


# =========================================================
# SPARQL VALIDATION
# =========================================================

SPARQL_QUERIES = {
    "Q1_individuals_per_class": """
        PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl:  <http://www.w3.org/2002/07/owl#>
        SELECT ?class (COUNT(?ind) AS ?count)
        WHERE {
            ?ind rdf:type owl:NamedIndividual .
            ?ind rdf:type ?class .
            FILTER(?class != owl:NamedIndividual)
        }
        GROUP BY ?class
        ORDER BY DESC(?count)
    """,
    "Q2_fk_links_total": """
        PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX owl:  <http://www.w3.org/2002/07/owl#>
        SELECT (COUNT(*) AS ?fk_count)
        WHERE {
            ?s ?p ?o .
            ?p rdf:type owl:ObjectProperty .
            ?s rdf:type owl:NamedIndividual .
            ?o rdf:type owl:NamedIndividual .
        }
    """,
    "Q3_data_assertions_total": """
        PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX owl:  <http://www.w3.org/2002/07/owl#>
        SELECT (COUNT(*) AS ?da_count)
        WHERE {
            ?s ?p ?o .
            ?p rdf:type owl:DatatypeProperty .
            FILTER(isLiteral(?o))
        }
    """,
    "Q4_patients_with_chemotherapy": """
        PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX owl:  <http://www.w3.org/2002/07/owl#>
        PREFIX inst: <http://example.org/instance/>
        SELECT (COUNT(DISTINCT ?patient) AS ?count)
        WHERE {
            ?cx rdf:type owl:NamedIndividual .
            ?cx ?hasPatient ?patient .
            ?patient rdf:type owl:NamedIndividual .
            FILTER(CONTAINS(STR(?cx), "chemotherapy"))
            FILTER(CONTAINS(STR(?patient), "patient_data"))
        }
    """,
    "Q5_radiotherapy_patients": """
        PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX owl:  <http://www.w3.org/2002/07/owl#>
        PREFIX onto: <http://example.org/ontology#>
        SELECT (COUNT(*) AS ?count)
        WHERE {
            ?cx rdf:type owl:NamedIndividual .
            ?cx onto:radiotherapy "true"^^<http://www.w3.org/2001/XMLSchema#boolean> .
            FILTER(CONTAINS(STR(?cx), "chemotherapy"))
        }
    """,
}


def run_sparql_validation(kg: Graph) -> Dict:
    results = {}
    for q_name, query in SPARQL_QUERIES.items():
        try:
            qr = list(kg.query(query))
            if q_name == "Q1_individuals_per_class":
                results[q_name] = [
                    {"class": str(r[0]).split("#")[-1].split("/")[-1],
                     "count": int(r[1])}
                    for r in qr
                ]
            elif q_name in ("Q2_fk_links_total", "Q3_data_assertions_total",
                            "Q4_patients_with_chemotherapy", "Q5_radiotherapy_patients"):
                results[q_name] = int(qr[0][0]) if qr else 0
            else:
                results[q_name] = len(qr)
            log.info("  SPARQL %-40s OK (%s)", q_name, results[q_name])
        except Exception as e:
            results[q_name] = f"error: {e}"
            log.warning("  SPARQL %-40s ERROR — %s", q_name, e)
    return results


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Populate KGs from SQL dump + RIGOR ontologies")
    parser.add_argument("--schema", choices=list(SCHEMAS.keys()),
                        help="Run only this schema (default: all)")
    parser.add_argument("--model",  help="Run only this model (default: all)")
    args, _ = parser.parse_known_args()  # parse_known_args ignores Jupyter/Colab kernel args

    print("=== sql_to_kg.py starting ===", flush=True)
    log.info("=" * 60)
    log.info("KG Population: Real-World (SQL) + Synthetic (CSV)")
    log.info("=" * 60)

    # Pre-load both data sources once; each schema picks the right one below
    #print("Step 1: Loading SQL dump...", flush=True)
    #sql_data = parse_sql_dump(SQL_DUMP)
    #print(f"SQL loaded: {sum(len(v['rows']) for v in sql_data.values())} total rows", flush=True)

    #print("Step 2: Loading CSV folder...", flush=True)
    #csv_data = load_csv_folder(CSV_DIR_SCHEMA2)
    #print(f"CSV loaded: {sum(len(v['rows']) for v in csv_data.values())} total rows", flush=True)

    print("Step 3: Loading Chinook JSON...", flush=True)
    json_data = load_json_tables(CHINOOK_JSON)
    print(f"JSON loaded: {sum(len(v['rows']) for v in json_data.values())} total rows", flush=True)
    all_stats   = []
    total_start = time.time()

    schema_keys = [args.schema] if args.schema else list(SCHEMAS.keys())

    for schema_key in schema_keys:
        schema_path = SCHEMAS[schema_key]
        if not os.path.exists(schema_path):
            log.warning("Schema file not found: %s — skipping %s", schema_path, schema_key)
            continue

        with open(schema_path) as f:
            schema = json.load(f)

        # Route to correct data source
        #data_type = SCHEMA_DATA_TYPE.get(schema_key, "sql")
        #if data_type == "csv":
         #   active_data = csv_data
          #  log.info("Using synthetic CSV data for %s", schema_key)
       # elif data_type == "json":
        active_data = json_data
        log.info("Using Chinook JSON data for %s", schema_key)
       # else:
        #    active_data = sql_data
        #    log.info("Using real-world SQL data for %s", schema_key)

        model_map = ONTOLOGIES[schema_key]
        model_keys = [args.model] if args.model else list(model_map.keys())

        for model_name in model_keys:
            onto_path = model_map.get(model_name)
            if onto_path is None:
                log.warning("Unknown model '%s' for %s", model_name, schema_key)
                continue

            log.info("\n%s", "=" * 55)
            log.info("Schema: %s | Model: %s", schema_key, model_name)
            log.info("=" * 55)

            if not os.path.exists(onto_path):
                log.warning("Ontology not found: %s — skipping", onto_path)
                all_stats.append({
                    "schema": schema_key, "model": model_name,
                    "error": "ontology file missing"
                })
                continue

            start = time.time()
            job_label = f"{schema_key}_{model_name}"
            kg, stats = populate(onto_path, active_data, schema, job_label)
            stats["schema"]      = schema_key
            stats["runtime_sec"] = round(time.time() - start, 1)

            # SPARQL validation skipped — rdflib in-memory SPARQL is too slow
            # for large graphs. Structural stats below are sufficient for the paper.
            log.info("Skipping SPARQL validation (disabled for performance)")
            stats["sparql"] = {"skipped": "disabled for performance on large graphs"}

            # Save outputs
            out_dir = os.path.join(OUTPUT_DIR, schema_key, model_name)
            os.makedirs(out_dir, exist_ok=True)

            kg_path = os.path.join(out_dir, "populated_kg.ttl")
            kg = sanitize_graph(kg)
            kg.serialize(kg_path, format="turtle")
            log.info("KG saved: %s", kg_path)

            with open(os.path.join(out_dir, "stats.json"), "w") as f:
                json.dump(stats, f, indent=2)

            all_stats.append(stats)

    # ── Summary table ──────────────────────────────────────────────────── #
    log.info("\n%s", "=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    print(f"\n{'Job':<22} {'Tables':>7} {'Rows':>7} {'Triples':>10} "
          f"{'FKLinks':>8} {'DataProp':>9} {'Time':>6}")
    print("-" * 70)
    for s in all_stats:
        if "error" in s:
            print(f"{s.get('schema','?')+'_'+s.get('model','?'):<22}  {s['error']}")
            continue
        print(
            f"{s.get('schema','?')+'_'+s.get('model','?'):<22} "
            f"{s.get('tables_populated',0):>7} "
            f"{s.get('rows_populated',0):>7} "
            f"{s.get('total_triples',0):>10,} "
            f"{s.get('fk_links',0):>8,} "
            f"{s.get('data_assertions',0):>9,} "
            f"{s.get('runtime_sec',0):>5.1f}s"
        )

    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    log.info("\nFull summary: %s", summary_path)
    log.info("Total time: %.1fs", time.time() - total_start)


if __name__ == "__main__":
    main()