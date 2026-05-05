"""
RIGOR Pipeline
=================================================

This implementation is designed to match the RIGOR algorithm as described in the paper:

1. FK-guided traversal with parent/reference tables processed before children.
2. Deterministic Direct Mapping is always merged first, guaranteeing schema coverage.
3. Context is retrieved per iteration from the current growing core ontology, documents, and external ontologies.
4. Gen-LLM produces semantic enrichment deltas, not schema coverage from scratch.
5. Judge-LLM validates/corrects deltas before merge.
6. Parsed graph is validated deterministically after parsing so the final OWL matches the checked artifact.
7. Required RIGOR constructs are represented/preserved where practical in RDFLib.

Set OPENROUTER_API_KEY before running:
    Windows PowerShell: $env:OPENROUTER_API_KEY="your_key_here"
    Linux/macOS       : export OPENROUTER_API_KEY="your_key_here"
"""


from __future__ import annotations
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import json

import re
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import chardet
#import faiss
import numpy as np
from openai import OpenAI
from rdflib import BNode, Graph, Literal, Namespace, RDF, RDFS, OWL, XSD, URIRef
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# =========================================================
# CONFIGURATION
# =========================================================

BASE_PATH = Path("YOUR_BASE_PATH")

SCHEMA_PATH = "/sql_schema/schema_chinook.json"
DOCS_PATH = "/documents_chinook"
ONTOLOGY_PATH = "/external_ontologies_chinook"
CORE_OWL_PATH = BASE_PATH / "core_ontology" / "core.owl"
OUTPUT_PATH = "output/claude/enriched_ontology.owl"
DIRECT_MAPPINGS_DIR = "output/direct_mappings"

ONTOLOGY_IRI = "http://example.org/ontology"
BASE = Namespace(f"{ONTOLOGY_IRI}#")
PROV = Namespace("http://www.w3.org/ns/prov#")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

MAX_JUDGE_RETRIES = 2
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
SENT_MODEL: Optional[SentenceTransformer] = None

OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "http://localhost")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "RIGOR-SemanticEnrichment")
MODELS = {
    "claude":   "anthropic/claude-opus-4-6",
    "mistral":  "mistralai/mistral-small-24b-instruct-2501",
    "deepseek": "deepseek/deepseek-chat",
}


DEFAULT_MODEL = "anthropic/claude-opus-4-6"

ETL_PREFIXES = ("s_",)
ETL_EXACT = {"s_ColLineage", "s_Generation", "s_GUID", "s_Lineage"}

SQL_TO_XSD = {
    "INTEGER": XSD.integer,
    "INT": XSD.integer,
    "SMALLINT": XSD.integer,
    "BIGINT": XSD.integer,
    "TINYINT": XSD.boolean,
    "BOOLEAN": XSD.boolean,
    "BOOL": XSD.boolean,
    "FLOAT": XSD.float,
    "DOUBLE": XSD.double,
    "DECIMAL": XSD.decimal,
    "NUMERIC": XSD.decimal,
    "TEXT": XSD.string,
    "VARCHAR": XSD.string,
    "CHAR": XSD.string,
    "CLOB": XSD.string,
    "TIMESTAMP": XSD.dateTime,
    "DATETIME": XSD.dateTime,
    "DATE": XSD.date,
    "TIME": XSD.time,
    "BLOB": XSD.base64Binary,
}

XSD_NAME_TO_URI = {
    "string": XSD.string,
    "integer": XSD.integer,
    "int": XSD.integer,
    "float": XSD.float,
    "boolean": XSD.boolean,
    "bool": XSD.boolean,
    "datetime": XSD.dateTime,
    "dateTime": XSD.dateTime,
    "date": XSD.date,
    "time": XSD.time,
    "decimal": XSD.decimal,
    "double": XSD.double,
    "base64Binary": XSD.base64Binary,
}

# =========================================================
# DATA MODEL
# =========================================================

@dataclass(frozen=True)
class ForeignKey:
    column: str
    references_table: str
    references_column: str

@dataclass
class Column:
    name: str
    raw_type: Any
    xsd_type: URIRef
    nullable: Optional[bool] = None
    unique: bool = False
    primary_key: bool = False
    source: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Table:
    name: str
    columns: Dict[str, Column]
    foreign_keys: List[ForeignKey]

Schema = Dict[str, Table]

# =========================================================
# GENERAL HELPERS
# =========================================================

def ensure_namespaces(g: Graph) -> None:
    g.bind("", BASE)
    g.bind("owl", OWL)
    g.bind("xsd", XSD)
    g.bind("rdfs", RDFS)
    g.bind("prov", PROV)
    g.bind("skos", SKOS)


def uri(local_name: str) -> URIRef:
    return BASE[safe_local_name(local_name)]


def safe_local_name(name: str) -> str:
    """Make a string safe enough for use as a local IRI name."""
    s = str(name).strip()
    s = re.sub(r"[^A-Za-z0-9_\-]", "_", s)
    s = re.sub(r"_+", "_", s)
    return s or "Unnamed"


def to_class_name(name: str) -> str:
    parts = re.split(r"[_\s\-]+", str(name).strip())
    return "".join(p[:1].upper() + p[1:] for p in parts if p) or "UnnamedClass"


def humanize_identifier(name: str) -> str:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name))
    s = s.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", s).strip()


def is_etl_column(col_name: str) -> bool:
    return col_name in ETL_EXACT or any(col_name.startswith(p) for p in ETL_PREFIXES)


def sql_type_base(sql_type: Any) -> str:
    if isinstance(sql_type, dict):
        value = sql_type.get("type") or sql_type.get("datatype") or sql_type.get("data_type") or sql_type.get("sql_type") or "TEXT"
    else:
        value = sql_type
    return str(value).upper().split("(")[0].strip()


def get_xsd_type(sql_type: Any) -> URIRef:
    return SQL_TO_XSD.get(sql_type_base(sql_type), XSD.string)


def fix_date_type(col_name: str, xsd_type: URIRef) -> URIRef:
    lower = col_name.lower()
    if xsd_type in {XSD.float, XSD.double, XSD.decimal, XSD.string}:
        if any(tok in lower for tok in ["datetime", "timestamp", "created_at", "updated_at"]):
            return XSD.dateTime
        if "date" in lower:
            return XSD.date
        if lower.endswith("time") or "_time" in lower:
            return XSD.time
    return xsd_type


def xsd_prefixed(x: URIRef) -> str:
    for name, u in XSD_NAME_TO_URI.items():
        if u == x:
            return f"xsd:{name}"
    return "xsd:string"


def parse_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "yes", "1", "not null", "non-null"}
    return bool(value)


def normalize_column(col_name: str, col_value: Any) -> Column:
    meta = col_value if isinstance(col_value, dict) else {}
    xsd = fix_date_type(col_name, get_xsd_type(col_value))
    primary = bool(meta.get("primary_key") or meta.get("pk") or meta.get("is_primary_key"))
    unique = bool(meta.get("unique") or meta.get("is_unique"))
    nullable = parse_bool(meta.get("nullable"))
    if meta.get("not_null") is True or meta.get("required") is True:
        nullable = False
    return Column(
        name=col_name,
        raw_type=col_value,
        xsd_type=xsd,
        nullable=nullable,
        unique=unique,
        primary_key=primary,
        source=dict(meta),
    )


def load_schema_from_json(path: Path | str) -> Schema:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    schema: Schema = {}
    for table_name, table_value in raw.items():
        if isinstance(table_value, dict) and "columns" in table_value:
            raw_columns = table_value.get("columns", {})
            raw_fks = table_value.get("foreign_keys", [])
        else:
            raw_columns = table_value
            raw_fks = []

        columns: Dict[str, Column] = {}
        for col_name, col_value in raw_columns.items():
            if is_etl_column(col_name):
                continue
            columns[col_name] = normalize_column(col_name, col_value)

        foreign_keys: List[ForeignKey] = []
        for fk in raw_fks:
            col = fk.get("column")
            ref_table = fk.get("references_table")
            ref_col = fk.get("references_column")
            if col and ref_table and ref_col and not is_etl_column(col):
                foreign_keys.append(ForeignKey(col, ref_table, ref_col))

        schema[table_name] = Table(table_name, columns, foreign_keys)
    #Enable FK inference to enrich schema when FK metadata is missing, which is common in practice. Disable if you prefer to rely strictly on provided metadata.
    #infer_missing_primary_keys(schema) 
    #infer_missing_foreign_keys(schema)
    return schema


def infer_missing_primary_keys(schema: Schema) -> None:
    for table in schema.values():
        if any(c.primary_key for c in table.columns.values()):
            continue
        candidates = ["id", f"{table.name}_id", f"{table.name.rstrip('s')}_id"]
        for cand in candidates:
            if cand in table.columns:
                table.columns[cand].primary_key = True
                table.columns[cand].nullable = False
                break


def table_primary_key(table: Table) -> Optional[str]:
    for c in table.columns.values():
        if c.primary_key:
            return c.name
    return None


def infer_missing_foreign_keys(schema: Schema) -> None:
    table_names = set(schema)
    pk_index = {t: table_primary_key(tbl) or "id" for t, tbl in schema.items()}
    for table in schema.values():
        existing_fk_cols = {fk.column for fk in table.foreign_keys}
        for col in table.columns.values():
            if col.name in existing_fk_cols or col.primary_key:
                continue
            if col.xsd_type != XSD.integer:
                continue
            parent = None
            if col.name.endswith("_id"):
                base = col.name[:-3]
                if base in table_names and base != table.name:
                    parent = base
            if parent is None and col.name.endswith("id") and not col.name.endswith("_id"):
                base = col.name[:-2]
                if base in table_names and base != table.name:
                    parent = base
            if parent is None and col.name in table_names and col.name != table.name:
                parent = col.name
            if parent:
                table.foreign_keys.append(ForeignKey(col.name, parent, pk_index[parent]))


def cleaned_schema_json_for_table(table_name: str, schema: Schema) -> str:
    table = schema[table_name]
    relevant = {table_name} | {fk.references_table for fk in table.foreign_keys if fk.references_table in schema}
    data: Dict[str, Any] = {}
    for t in relevant:
        tbl = schema[t]
        data[t] = {
            "columns": {
                c.name: {
                    "xsd_type": xsd_prefixed(c.xsd_type),
                    "nullable": c.nullable,
                    "unique": c.unique,
                    "primary_key": c.primary_key,
                }
                for c in tbl.columns.values()
            },
            "foreign_keys": [fk.__dict__ for fk in tbl.foreign_keys],
        }
    return json.dumps(data, indent=2)


def format_foreign_keys(table_name: str, schema: Schema) -> str:
    fks = schema[table_name].foreign_keys
    if not fks:
        return "None"
    return "\n".join(
        f"  - Column '{fk.column}' references {fk.references_table}.{fk.references_column}"
        for fk in fks
    )

# =========================================================
# FK-GUIDED TRAVERSAL
# =========================================================

def traverse_schema_fk_order(schema: Schema) -> List[str]:
    """
    Parent/reference tables before child/dependent tables.

    Edge direction: referenced parent -> referencing child.
    Roots are tables with no outgoing FKs, i.e., tables that do not depend on any parent.
    This matches RIGOR's rationale: classes for parent tables exist before child tables refer to them.
    """
    children: Dict[str, Set[str]] = defaultdict(set)
    indegree: Dict[str, int] = {t: 0 for t in schema}

    for child_name, table in schema.items():
        for fk in table.foreign_keys:
            parent = fk.references_table
            if parent in schema and parent != child_name:
                if child_name not in children[parent]:
                    children[parent].add(child_name)
                    indegree[child_name] += 1

    roots = sorted([t for t, deg in indegree.items() if deg == 0])
    queue = deque(roots if roots else sorted(schema.keys()))
    seen: Set[str] = set()
    order: List[str] = []

    while queue:
        t = queue.popleft()
        if t in seen:
            continue
        seen.add(t)
        order.append(t)
        for child in sorted(children[t]):
            indegree[child] -= 1
            if indegree[child] <= 0:
                queue.append(child)

    # Cycles or unresolved components are appended deterministically.
    for t in sorted(schema.keys()):
        if t not in seen:
            order.append(t)

    return order

# =========================================================
# DETERMINISTIC DIRECT MAPPING
# =========================================================



def data_property_name(table_name: str, col_name: str) -> str:
    """
    Table-scoped data property names prevent domain collisions for repeated
    column names such as Name, Title, Email, Address, City, Country, etc.
    """
    return safe_local_name(f"{table_name}_{col_name}")

def object_property_name_for_fk(fk: ForeignKey, source_table: Optional[str] = None) -> str:
    """
    Generate source-table-scoped object property names to avoid domain collisions.

    Examples:
      Album.ArtistId          -> Album_hasArtist
      Track.AlbumId           -> Track_hasAlbum
      PlaylistTrack.TrackId   -> PlaylistTrack_hasTrack
      InvoiceLine.TrackId     -> InvoiceLine_hasTrack
      Employee.ReportsTo      -> Employee_ReportsTo
    """
    if source_table and fk.references_table == source_table:
        return safe_local_name(f"{source_table}_{fk.column}")

    if fk.column.lower().endswith("id"):
        base = fk.column[:-2]
        if base:
            return safe_local_name(f"{source_table}_has{to_class_name(base)}")

    return safe_local_name(f"{source_table}_has{to_class_name(fk.references_table)}")

def add_annotation(g: Graph, subject: URIRef, label: str, comment: str, source_iri: str) -> None:
    g.add((subject, RDFS.label, Literal(label, lang="en")))
    g.add((subject, RDFS.comment, Literal(comment, lang="en")))
    g.add((subject, PROV.wasDerivedFrom, URIRef(source_iri)))


def add_existential_restriction(g: Graph, class_uri: URIRef, prop_uri: URIRef, filler: URIRef) -> None:
    """Represent C SubClassOf: p some filler."""
    restriction = BNode()
    g.add((restriction, RDF.type, OWL.Restriction))
    g.add((restriction, OWL.onProperty, prop_uri))
    g.add((restriction, OWL.someValuesFrom, filler))
    g.add((class_uri, RDFS.subClassOf, restriction))


def add_universal_restriction(g: Graph, class_uri: URIRef, prop_uri: URIRef, filler: URIRef) -> None:
    """Represent C SubClassOf: p only filler."""
    restriction = BNode()
    g.add((restriction, RDF.type, OWL.Restriction))
    g.add((restriction, OWL.onProperty, prop_uri))
    g.add((restriction, OWL.allValuesFrom, filler))
    g.add((class_uri, RDFS.subClassOf, restriction))


def add_haskey(g: Graph, class_uri: URIRef, prop_uris: Sequence[URIRef]) -> None:
    if not prop_uris:
        return
    # RDF list for owl:hasKey.
    collection = BNode()
    g.add((class_uri, OWL.hasKey, collection))
    current = collection
    for i, p in enumerate(prop_uris):
        g.add((current, RDF.first, p))
        if i == len(prop_uris) - 1:
            g.add((current, RDF.rest, RDF.nil))
        else:
            nxt = BNode()
            g.add((current, RDF.rest, nxt))
            current = nxt


def direct_mapping_graph_for_table(table_name: str, schema: Schema) -> Graph:
    """
    Deterministic RIGOR DirectMap(clean(r)).

    Guarantees:
    - table -> owl:Class
    - non-FK columns -> owl:DatatypeProperty
    - FK columns -> owl:ObjectProperty
    - ETL columns are absent because schema was cleaned at load time
    - date/time corrections are applied before this function
    - universal restrictions are added for every mapped property
    - existential restrictions are added for primary-key, unique, or not-null columns
    """
    g = Graph()
    ensure_namespaces(g)
    table = schema[table_name]
    class_name = to_class_name(table_name)
    class_uri = uri(class_name)

    g.add((class_uri, RDF.type, OWL.Class))
    add_annotation(
        g,
        class_uri,
        humanize_identifier(class_name),
        f"Class generated from source table '{table_name}'.",
        f"{ONTOLOGY_IRI}/provenance/{table_name}",
    )

    fk_by_col = {fk.column: fk for fk in table.foreign_keys}
    key_props: List[URIRef] = []

    for col in table.columns.values():
        if col.name in fk_by_col:
            fk = fk_by_col[col.name]
            if fk.references_table not in schema:
                continue
            prop_name = object_property_name_for_fk(fk, table_name)
            prop_uri = uri(prop_name)
            range_uri = uri(to_class_name(fk.references_table))
            g.add((range_uri, RDF.type, OWL.Class))
            g.add((prop_uri, RDF.type, OWL.ObjectProperty))
            g.add((prop_uri, RDFS.domain, class_uri))
            g.add((prop_uri, RDFS.range, range_uri))
            relationship_comment = (
                f"Object property generated from foreign key '{table_name}.{col.name}' "
                f"referencing '{fk.references_table}.{fk.references_column}'."
            )

            if fk.references_table == table_name:
                relationship_comment += (
                    " This is an explicit recursive self-referential relationship within "
                    f"the {to_class_name(table_name)} class."
                )

            add_annotation(
                g,
                prop_uri,
                humanize_identifier(prop_name),
                relationship_comment,
                f"{ONTOLOGY_IRI}/provenance/{table_name}/{col.name}",
            )
            add_universal_restriction(g, class_uri, prop_uri, range_uri)
            if col.primary_key or col.unique or col.nullable is False:
                add_existential_restriction(g, class_uri, prop_uri, range_uri)
            if col.primary_key:
                key_props.append(prop_uri)
        else:
            prop_name = data_property_name(table_name, col.name)
            prop_uri = uri(prop_name)
            g.add((prop_uri, RDF.type, OWL.DatatypeProperty))
            g.add((prop_uri, RDFS.domain, class_uri))
            g.add((prop_uri, RDFS.range, col.xsd_type))
            unit_or_encoding_note = ""
            if col.xsd_type == XSD.integer and re.search(r"(stage|score|status|grade|class|category|type)$", col.name, re.I):
                unit_or_encoding_note = " This integer-valued property may encode a categorical or ordinal concept; consult source documentation for value meanings."
            add_annotation(
                g,
                prop_uri,
                humanize_identifier(col.name),
                f"Data property generated from source column '{table_name}.{col.name}' with range {xsd_prefixed(col.xsd_type)}.{unit_or_encoding_note}",
                f"{ONTOLOGY_IRI}/provenance/{table_name}/{col.name}",
            )
            add_universal_restriction(g, class_uri, prop_uri, col.xsd_type)
            if col.primary_key or col.unique or col.nullable is False:
                add_existential_restriction(g, class_uri, prop_uri, col.xsd_type)
            if col.primary_key:
                key_props.append(prop_uri)

    add_haskey(g, class_uri, key_props)
    return g


def render_direct_mapping_manchester(table_name: str, schema: Schema) -> str:
    table = schema[table_name]
    class_name = to_class_name(table_name)
    lines = [
        f"Prefix: : <{ONTOLOGY_IRI}#>",
        "Prefix: xsd: <http://www.w3.org/2001/XMLSchema#>",
        "Prefix: rdfs: <http://www.w3.org/2000/01/rdf-schema#>",
        "Prefix: owl: <http://www.w3.org/2002/07/owl#>",
        "Prefix: prov: <http://www.w3.org/ns/prov#>",
        "",
        f"Ontology: <{ONTOLOGY_IRI}>",
        "",
        f"Class: {class_name}",
        "  Annotations:",
        f"    rdfs:label \"{humanize_identifier(class_name)}\"@en,",
        f"    rdfs:comment \"Class generated from source table '{table_name}'.\"@en,",
        f"    prov:wasDerivedFrom <{ONTOLOGY_IRI}/provenance/{table_name}>",
    ]

    fk_by_col = {fk.column: fk for fk in table.foreign_keys}
    pk_props = []
    subclass_restrictions = []

    for col in table.columns.values():
        if col.name in fk_by_col:
            fk = fk_by_col[col.name]
            prop = object_property_name_for_fk(fk, table_name)
            rng = to_class_name(fk.references_table)

            comment = (
                f"Foreign-key relationship from {table_name}.{col.name} "
                f"to {fk.references_table}.{fk.references_column}."
            )
            if fk.references_table == table_name:
                comment += (
                    f" This is an explicit recursive self-referential relationship "
                    f"within {to_class_name(table_name)}."
                )

            lines += [
                "",
                f"ObjectProperty: {prop}",
                "  Annotations:",
                f"    rdfs:label \"{humanize_identifier(prop)}\"@en,",
                f"    rdfs:comment \"{comment}\"@en,",
                f"    prov:wasDerivedFrom <{ONTOLOGY_IRI}/provenance/{table_name}/{col.name}>",
                f"  Domain: {class_name}",
                f"  Range: {rng}",
            ]

            subclass_restrictions.append(f"{prop} only {rng}")
            if col.primary_key or col.unique or col.nullable is False:
                subclass_restrictions.append(f"{prop} some {rng}")
            if col.primary_key:
                pk_props.append(prop)

        else:
            prop = data_property_name(table_name, col.name)
            rng = xsd_prefixed(col.xsd_type)

            comment = f"Column {table_name}.{col.name}; datatype {rng}."
            if col.xsd_type == XSD.integer and re.search(
                r"(stage|score|status|grade|class|category|type)$",
                col.name,
                re.I,
            ):
                comment += (
                    " This integer-valued property may encode a categorical "
                    "or ordinal concept; consult source documentation for value meanings."
                )

            lines += [
                "",
                f"DataProperty: {prop}",
                "  Annotations:",
                f"    rdfs:label \"{humanize_identifier(prop)}\"@en,",
                f"    rdfs:comment \"{comment}\"@en,",
                f"    prov:wasDerivedFrom <{ONTOLOGY_IRI}/provenance/{table_name}/{col.name}>",
                f"  Domain: {class_name}",
                f"  Range: {rng}",
            ]

            subclass_restrictions.append(f"{prop} only {rng}")
            if col.primary_key or col.unique or col.nullable is False:
                subclass_restrictions.append(f"{prop} some {rng}")
            if col.primary_key:
                pk_props.append(prop)

    if pk_props or subclass_restrictions:
        lines += ["", f"Class: {class_name}"]
        if pk_props:
            lines.append(f"  HasKey: {' '.join(pk_props)}")
        for r in subclass_restrictions:
            lines.append(f"  SubClassOf: {r}")

    return "\n".join(lines)
# =========================================================
# DOCUMENT AND ONTOLOGY RETRIEVAL
# =========================================================

def load_text_documents(doc_folder: Path | str) -> Dict[str, str]:
    folder = Path(doc_folder)
    docs: Dict[str, str] = {}
    if not folder.exists():
        print(f"  Warning: docs folder not found: {folder}")
        return docs

    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        rel_name = str(path.relative_to(folder))
        try:
            suffix = path.suffix.lower()
            if suffix in {".txt", ".md", ".csv", ".json", ".sql"}:
                raw = path.read_bytes()
                enc = chardet.detect(raw).get("encoding") or "utf-8"
                docs[rel_name] = path.read_text(encoding=enc, errors="replace")
            elif suffix == ".docx":
                try:
                    import docx
                    docs[rel_name] = "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
                except ImportError:
                    print(f"  Skipping {rel_name} because python-docx is not installed")
            elif suffix == ".pdf":
                try:
                    import PyPDF2
                    with open(path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        docs[rel_name] = "\n".join(page.extract_text() or "" for page in reader.pages)
                except ImportError:
                    print(f"  Skipping {rel_name} because PyPDF2 is not installed")
        except Exception as e:
            print(f"  Skipping {rel_name}: {e}")

    print(f"  Loaded {len(docs)} documents")
    return docs


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def embed_texts(texts: List[str], model_name: str = EMBED_MODEL_NAME) -> np.ndarray:
    global SENT_MODEL

    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    if SENT_MODEL is None:
        print("   Loading SentenceTransformer on CPU...")
        SENT_MODEL = SentenceTransformer(model_name, device="cpu")

    vectors = SENT_MODEL.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
        device="cpu",
        batch_size=16,
    )

    return np.asarray(vectors, dtype=np.float32)

def build_faiss_index_from_embeddings(embeddings: np.ndarray):
    if embeddings.size == 0:
        return None

    import faiss  # lazy import avoids macOS OpenMP conflict with torch/sentence-transformers

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


def retrieve_top_k(query: str, index: Any, texts: Sequence[str], k: int = 3) -> List[str]:
    if index is None or not texts:
        return []
    q_emb = embed_texts([query]).astype(np.float32)
    _, indices = index.search(q_emb, min(k, len(texts)))
    return [texts[i] for i in indices[0] if 0 <= i < len(texts)]


def load_external_ontologies(onto_folder: str) -> List[str]:
    """
    Load external ontologies and extract class/property text chunks
    for semantic retrieval via FAISS.
    Supports .owl, .rdf, .ttl, .nt, and .n3.
    """
    chunks = []

    if not os.path.exists(onto_folder):
        print(f"  Warning: ontology folder not found: {onto_folder}")
        return chunks

    for root, _, files in os.walk(onto_folder):
        for filename in files:
            if not filename.endswith((".owl", ".rdf", ".ttl", ".nt", ".n3")):
                continue

            path = os.path.join(root, filename)
            before = len(chunks)

            try:
                g = Graph()
                parsed = False
                last_error = None

                suffix = os.path.splitext(filename)[1].lower()

                if suffix == ".ttl":
                    formats = ("turtle", "n3", "xml", "application/rdf+xml", "nt")
                elif suffix == ".nt":
                    formats = ("nt", "turtle", "xml", "application/rdf+xml", "n3")
                elif suffix == ".n3":
                    formats = ("n3", "turtle", "xml", "application/rdf+xml", "nt")
                else:
                    formats = ("xml", "application/rdf+xml", "pretty-xml", "turtle", "n3", "nt")

                for fmt in formats:
                    try:
                        g.parse(path, format=fmt)
                        parsed = True
                        print(f"  Parsed {filename} as {fmt}")
                        break
                    except Exception as e:
                        last_error = e

                if not parsed:
                    raise ValueError(
                        f"Could not parse ontology with supported RDF formats. "
                        f"Last error: {last_error}"
                    )

                for cls in g.subjects(RDF.type, OWL.Class):
                    name = str(cls).split("#")[-1].split("/")[-1]
                    lbl = next(g.objects(cls, RDFS.label), None)
                    cmt = next(g.objects(cls, RDFS.comment), None)

                    if name:
                        text = f"[{filename}] Class: {name} IRI: {cls}"
                        if lbl:
                            text += f" — label: {lbl}"
                        if cmt:
                            text += f" — comment: {str(cmt)[:200]}"
                        chunks.append(text)

                props = (
                    list(g.subjects(RDF.type, OWL.ObjectProperty))
                    + list(g.subjects(RDF.type, OWL.DatatypeProperty))
                    + list(g.subjects(RDF.type, RDF.Property))
                )

                for prop in props:
                    name = str(prop).split("#")[-1].split("/")[-1]
                    lbl = next(g.objects(prop, RDFS.label), None)
                    cmt = next(g.objects(prop, RDFS.comment), None)

                    if name:
                        text = f"[{filename}] Property: {name} IRI: {prop}"
                        if lbl:
                            text += f" — label: {lbl}"
                        if cmt:
                            text += f" — comment: {str(cmt)[:200]}"
                        chunks.append(text)

                added = len(chunks) - before
                print(f"  {filename}: added {added} chunks")

            except Exception as e:
                print(f"  Skipped {filename}: {e}")

    print(f"  Loaded {len(chunks)} external ontology chunks")
    return chunks


def build_core_ontology_chunks(core: Graph) -> List[str]:
    chunks: List[str] = []
    for cls in core.subjects(RDF.type, OWL.Class):
        if isinstance(cls, BNode):
            continue
        name = str(cls).split("#")[-1].split("/")[-1]
        labels = list(core.objects(cls, RDFS.label))[:2]
        comments = list(core.objects(cls, RDFS.comment))[:1]
        text = f"Class: {name} IRI: {cls}"
        if labels:
            text += " Label: " + "; ".join(map(str, labels))
        if comments:
            text += " Comment: " + str(comments[0])[:200]
        chunks.append(text)

    for prop in list(core.subjects(RDF.type, OWL.DatatypeProperty)) + list(core.subjects(RDF.type, OWL.ObjectProperty)):
        if isinstance(prop, BNode):
            continue
        name = str(prop).split("#")[-1].split("/")[-1]
        labels = list(core.objects(prop, RDFS.label))[:2]
        ranges = list(core.objects(prop, RDFS.range))[:1]
        text = f"Property: {name} IRI: {prop}"
        if labels:
            text += " Label: " + "; ".join(map(str, labels))
        if ranges:
            text += f" Range: {ranges[0]}"
        chunks.append(text)
    return chunks


def build_retrieval_query(table_name: str, schema: Schema) -> str:
    table = schema[table_name]
    col_names = " ".join(table.columns.keys())
    fk_targets = " ".join(fk.references_table for fk in table.foreign_keys)
    return f"{table_name} {to_class_name(table_name)} {col_names} {fk_targets}".strip()

# =========================================================
# OPENROUTER CLIENTS
# =========================================================

class OpenRouterLLM:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        self.model = model
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 10000, retries: int = 3) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_headers={"HTTP-Referer": OPENROUTER_SITE_URL, "X-Title": OPENROUTER_APP_NAME},
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You generate OWL 2 ontology fragments in Manchester Syntax. "
                                "Return only valid Manchester Syntax and no explanation."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as e:
                last_error = e
                if attempt < retries:
                    sleep_seconds = 2 ** (attempt - 1)
                    print(f"  Retry {attempt}/{retries} after error: {e}")
                    time.sleep(sleep_seconds)
                else:
                    print(f"  ERROR: {last_error}")
        return ""

class OntologyLLM:
    def __init__(self, client: OpenRouterLLM):
        self.client = client

    def generate(
        self,
        table_name: str,
        direct_mapping: str,
        schema_str: str,
        foreign_keys_str: str,
        documents: str,
        core_context: str,
        external_context: str,
        correction_hint: str = "",
    ) -> str:
        prompt = self._build_prompt(
            table_name,
            direct_mapping,
            schema_str,
            foreign_keys_str,
            documents,
            core_context,
            external_context,
            correction_hint,
        )
        return self.client.generate(prompt, temperature=0.2, max_tokens=10000)

    def _build_prompt(
        self,
        table_name: str,
        direct_mapping: str,
        schema_str: str,
        foreign_keys_str: str,
        documents: str,
        core_context: str,
        external_context: str,
        correction_hint: str,
    ) -> str:
        documents = documents[:5000] + "..." if len(documents) > 5000 else documents
        return f"""You are receiving a deterministic OWL direct mapping for database table '{table_name}'.
Your role is semantic enrichment, not schema reconstruction.
The direct mapping is already merged into the core ontology before your delta is applied.
Therefore, your output should refine, annotate, align, or add justified axioms while preserving coverage.
Do not invent source columns.
Do not remove deterministic mappings unless the reason is explicit and valid.

[DIRECT MAPPING FOR THIS TABLE]
{direct_mapping}

[CLEANED DATABASE SCHEMA: CURRENT TABLE + FK TARGETS]
{schema_str}

[FOREIGN KEY CONSTRAINTS FOR '{table_name}']
{foreign_keys_str}

[RETRIEVED DOCUMENTATION]
{documents or 'None'}

[RETRIEVED CURRENT CORE ONTOLOGY CONTEXT]
{core_context or 'None'}

[RETRIEVED EXTERNAL ONTOLOGY HINTS]
{external_context or 'None'}

[CORRECTION INSTRUCTIONS FROM PREVIOUS JUDGE PASS]
{correction_hint or 'None'}

[MANDATORY TASKS]
1. Add or improve rdfs:label and rdfs:comment for every class and property in the direct mapping.
2. Preserve FK modeling as owl:ObjectProperty. Do not output a DataProperty for an FK column.
3. Do not output ETL properties beginning with s_.
4. Add SubClassOf axioms where natural specialization exists.
5. Add comments documenting categorical or ordinal integer encodings when applicable.
6. Add DisjointWith axioms between sibling classes only when logically safe.
7. Add or preserve existential restrictions for not-null, unique, and primary-key properties.
8. Add external alignment using owl:equivalentClass, rdfs:subClassOf, or skos:exactMatch only when supported by retrieved external ontology hints.
9. Add prov:wasDerivedFrom annotations for every newly introduced or modified class/property.

[STRICT CONSTRAINTS]
- A URI must not be both owl:ObjectProperty and owl:DatatypeProperty.
- Every property must have exactly one rdfs:domain and exactly one rdfs:range.
- DataProperty ranges must be XSD datatypes.
- ObjectProperty ranges must be classes.
- Do not create self-referential ObjectProperties unless explicitly meaningful.
- Do not rename deterministic properties unless you also preserve the original property as equivalent or subproperty with provenance.
- Output only Manchester Syntax.

[OUTPUT]
"""

class JudgeLLM:
    def __init__(self, client: OpenRouterLLM):
        self.client = client

    def judge(
        self,
        table_name: str,
        direct_mapping: str,
        delta_ontology: str,
        schema_context: str,
        foreign_keys_str: str,
        core_context: str,
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(table_name, direct_mapping, delta_ontology, schema_context, foreign_keys_str, core_context)
        response = self.client.generate(prompt, temperature=0.1, max_tokens=8000)
        return self._parse_response(response)

    def _build_prompt(self, table_name: str, direct_mapping: str, delta_ontology: str, schema_context: str, foreign_keys_str: str, core_context: str) -> str:
        return f"""You are an ontology validation expert. Evaluate the delta ontology fragment generated for table '{table_name}'.
The deterministic Direct Mapping has already been merged into the core before this delta is applied.
Your task is to ensure the delta does not break RIGOR's guarantees and adds valid semantic enrichment.

[DIRECT MAPPING]
{direct_mapping}

[DELTA ONTOLOGY]
{delta_ontology}

[CLEANED SCHEMA]
{schema_context}

[FOREIGN KEY CONSTRAINTS]
{foreign_keys_str}

[CURRENT CORE CONTEXT]
{core_context}

[VALIDATION CRITERIA]
Critical:
1. No FK column may appear as a DataProperty if it is modeled as an ObjectProperty.
2. No URI may be both owl:ObjectProperty and owl:DatatypeProperty.
3. Every property must have exactly one domain and one range.
4. DataProperty ranges must be valid XSD datatypes; date/time properties must not be xsd:string or xsd:float.
5. ObjectProperty domain and range must not be identical unless documented as a meaningful recursive relationship.
6. The delta must not remove or contradict deterministic direct mapping coverage.
7. Required primary-key/not-null/unique existential restrictions must be preserved or strengthened.

Important:
8. All class/property labels and comments should be present and in English.
9. Encoded categorical integers should have encoding comments or modeled enumerations.
10. Junction tables should be treated as relationships when appropriate.
11. Core concepts should be reused, subclassed, or aligned rather than duplicated.
12. Disjointness should be used only when logically safe.

Minor:
13. External alignments should be added where supported by retrieved context.
14. Provenance should be present for every new or modified element.

[OUTPUT FORMAT]
Return exactly this structure and no extra text:
Decision: APPROVED | REJECTED | APPROVED_WITH_CORRECTIONS
Critical issues (must fix before merging): [list each issue on a new line, or "none"]
Important issues (should fix before merging): [list each issue on a new line, or "none"]
Minor issues (can fix in a later pass): [list each issue on a new line, or "none"]
Corrected fragment: [full corrected Manchester Syntax if Decision is not APPROVED, otherwise "N/A"]
"""

    def _parse_response(self, response: str) -> Dict[str, Any]:
        result = {"decision": "REJECTED", "critical": [], "important": [], "minor": [], "corrected": None}
        if not response:
            result["critical"] = ["Judge-LLM returned empty response"]
            return result

        m = re.search(r"Decision:\s*(APPROVED_WITH_CORRECTIONS|APPROVED|REJECTED)", response, re.I)
        if m:
            result["decision"] = m.group(1).upper()

        def section(title: str, next_titles: str) -> List[str]:
            pat = rf"{title}.*?:\s*(.*?)(?=\n(?:{next_titles})|\Z)"
            sm = re.search(pat, response, re.I | re.S)
            if not sm:
                return []
            text = sm.group(1).strip()
            if text.lower() in {"none", "n/a", ""}:
                return []
            return [ln.strip("- *\t ") for ln in text.splitlines() if ln.strip() and ln.strip().lower() != "none"]

        result["critical"] = section("Critical issues", "Important issues|Minor issues|Corrected fragment")
        result["important"] = section("Important issues", "Minor issues|Corrected fragment")
        result["minor"] = section("Minor issues", "Corrected fragment")

        m2 = re.search(r"Corrected fragment:\s*(.*)\Z", response, re.I | re.S)
        if m2:
            corrected = m2.group(1).strip()
            if corrected.lower() not in {"n/a", "none", ""}:
                result["corrected"] = corrected
        return result

# =========================================================
# MANCHESTER DELTA PARSER
# =========================================================

def strip_fences(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("```"):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines)


def split_manchester_blocks(text: str) -> List[List[str]]:
    text = strip_fences(text)
    blocks: List[List[str]] = []
    cur: List[str] = []
    starters = ("Class:", "DataProperty:", "ObjectProperty:", "AnnotationProperty:")
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if any(line.startswith(s) for s in starters):
            if cur:
                blocks.append(cur)
            cur = [line]
        elif cur:
            cur.append(line)
    if cur:
        blocks.append(cur)
    return blocks


def parse_prefixed_or_uri(token: str, default_base: Namespace = BASE) -> URIRef:
    t = token.strip().strip(",").strip()
    if t.startswith("<") and t.endswith(">"):
        return URIRef(t[1:-1])
    if t in {"owl:Thing", "Thing"}:
        return OWL.Thing
    if t.startswith("xsd:"):
        name = t.split(":", 1)[1]
        return XSD_NAME_TO_URI.get(name, XSD.string)
    if t.startswith("owl:"):
        return OWL[t.split(":", 1)[1]]
    if t.startswith("rdfs:"):
        return RDFS[t.split(":", 1)[1]]
    if t.startswith("skos:"):
        return SKOS[t.split(":", 1)[1]]
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", t):
        return URIRef(t)
    return default_base[safe_local_name(t)]


def parse_annotation_line(line: str) -> Optional[Tuple[URIRef, Literal | URIRef]]:
    # Handles rdfs:label "..."@en, rdfs:comment "..."@en, prov:wasDerivedFrom <...>, skos:exactMatch <...>
    clean = line.rstrip(",").strip()
    m_lit = re.match(r"(rdfs:label|rdfs:comment)\s+\"(.*?)\"(?:@(\w+))?", clean, re.I)
    if m_lit:
        pred = RDFS.label if m_lit.group(1).lower() == "rdfs:label" else RDFS.comment
        return pred, Literal(m_lit.group(2), lang=m_lit.group(3) or "en")
    m_iri = re.match(r"(prov:wasDerivedFrom|skos:exactMatch|owl:equivalentClass)\s+(.+)", clean, re.I)
    if m_iri:
        p = m_iri.group(1).lower()
        pred = PROV.wasDerivedFrom if p == "prov:wasderivedfrom" else SKOS.exactMatch if p == "skos:exactmatch" else OWL.equivalentClass
        return pred, parse_prefixed_or_uri(m_iri.group(2))
    return None


def parse_restriction(expr: str, class_uri: URIRef) -> Optional[Tuple[URIRef, URIRef, URIRef]]:
    # property some/only filler
    m = re.match(r"([A-Za-z0-9_\-]+)\s+(some|only)\s+(.+)", expr.strip())
    if not m:
        return None
    prop = uri(m.group(1))
    kind = OWL.someValuesFrom if m.group(2) == "some" else OWL.allValuesFrom
    filler = parse_prefixed_or_uri(m.group(3))
    return prop, kind, filler


def parse_llm_ontology(llm_output: str) -> Graph:
    """
    Tolerant parser for the Manchester subset used by RIGOR prompts.
    It preserves labels/comments/provenance, domains/ranges, subclass links,
    existential/universal restrictions, disjointness, external alignments, and HasKey.
    """
    g = Graph()
    ensure_namespaces(g)
    blocks = split_manchester_blocks(llm_output)

    for block in blocks:
        header = block[0]
        rest = block[1:]
        try:
            kind, name = header.split(":", 1)
            kind = kind.strip()
            name = name.strip()
            subject = uri(name)

            if kind == "Class":
                g.add((subject, RDF.type, OWL.Class))
                in_annotations = False
                for line in rest:
                    if line.startswith("Annotations:"):
                        in_annotations = True
                        continue
                    if line.startswith(("SubClassOf:", "DisjointWith:", "EquivalentTo:", "HasKey:")):
                        in_annotations = False
                    if in_annotations:
                        ann = parse_annotation_line(line)
                        if ann:
                            g.add((subject, ann[0], ann[1]))
                        continue
                    if line.startswith("SubClassOf:"):
                        expr = line.split(":", 1)[1].strip()
                        restr = parse_restriction(expr, subject)
                        if restr:
                            prop, restriction_pred, filler = restr
                            b = BNode()
                            g.add((b, RDF.type, OWL.Restriction))
                            g.add((b, OWL.onProperty, prop))
                            g.add((b, restriction_pred, filler))
                            g.add((subject, RDFS.subClassOf, b))
                        else:
                            g.add((subject, RDFS.subClassOf, parse_prefixed_or_uri(expr)))
                    elif line.startswith("DisjointWith:"):
                        target = parse_prefixed_or_uri(line.split(":", 1)[1].strip())
                        g.add((subject, OWL.disjointWith, target))
                    elif line.startswith("EquivalentTo:"):
                        target = parse_prefixed_or_uri(line.split(":", 1)[1].strip())
                        g.add((subject, OWL.equivalentClass, target))
                    elif line.startswith("HasKey:"):
                        props = [uri(p) for p in line.split(":", 1)[1].split()]
                        add_haskey(g, subject, props)

            elif kind in {"DataProperty", "ObjectProperty"}:
                rdf_type = OWL.DatatypeProperty if kind == "DataProperty" else OWL.ObjectProperty
                g.add((subject, RDF.type, rdf_type))
                in_annotations = False
                for line in rest:
                    if line.startswith("Annotations:"):
                        in_annotations = True
                        continue
                    if line.startswith(("Domain:", "Range:", "SubPropertyOf:", "EquivalentTo:")):
                        in_annotations = False
                    if in_annotations:
                        ann = parse_annotation_line(line)
                        if ann:
                            g.add((subject, ann[0], ann[1]))
                        continue
                    if line.startswith("Domain:"):
                        g.add((subject, RDFS.domain, parse_prefixed_or_uri(line.split(":", 1)[1].strip())))
                    elif line.startswith("Range:"):
                        rng = line.split(":", 1)[1].strip()
                        if kind == "DataProperty":
                            if rng.startswith("xsd:"):
                                g.add((subject, RDFS.range, parse_prefixed_or_uri(rng)))
                            else:
                                # Prevent invalid non-XSD datatype ranges from entering the graph.
                                g.add((subject, RDFS.range, XSD.string))
                                g.add((subject, RDFS.comment, Literal(f"Parser corrected non-XSD datatype range '{rng}' to xsd:string.", lang="en")))
                        else:
                            g.add((subject, RDFS.range, parse_prefixed_or_uri(rng)))
                    elif line.startswith("SubPropertyOf:"):
                        g.add((subject, RDFS.subPropertyOf, parse_prefixed_or_uri(line.split(":", 1)[1].strip())))
                    elif line.startswith("EquivalentTo:"):
                        g.add((subject, OWL.equivalentProperty, parse_prefixed_or_uri(line.split(":", 1)[1].strip())))
        except Exception as e:
            print(f"  Parser skipped block '{header[:80]}': {e}")

    return g

# =========================================================
# DETERMINISTIC VALIDATION
# =========================================================

@dataclass
class ValidationReport:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


def validate_graph_against_rigor(g: Graph, schema: Schema) -> ValidationReport:
    report = ValidationReport(ok=True)

    # Type consistency: no property URI may be both object and datatype property.
    obj_props = set(g.subjects(RDF.type, OWL.ObjectProperty))
    data_props = set(g.subjects(RDF.type, OWL.DatatypeProperty))
    for p in sorted(obj_props & data_props, key=str):
        report.add_error(f"Property {p} is both owl:ObjectProperty and owl:DatatypeProperty")

    # Exactly one domain/range per named property.
    for p in sorted(obj_props | data_props, key=str):
        domains = list(g.objects(p, RDFS.domain))
        ranges = list(g.objects(p, RDFS.range))
        if len(domains) != 1:
            report.add_error(f"Property {p} has {len(domains)} domains, expected exactly 1")
        if len(ranges) != 1:
            report.add_error(f"Property {p} has {len(ranges)} ranges, expected exactly 1")
        if p in data_props and ranges and not str(ranges[0]).startswith(str(XSD)):
            report.add_error(f"Datatype property {p} has non-XSD range {ranges[0]}")
        if p in obj_props and domains and ranges and domains[0] == ranges[0]:
            comments = " ".join(str(c).lower() for c in g.objects(p, RDFS.comment))
            if "recursive" not in comments and "self" not in comments:
                report.add_error(f"Object property {p} is self-referential without explicit recursive justification")

    # Schema coverage and FK correctness from deterministic mapping.
    for table_name, table in schema.items():
        class_uri = uri(to_class_name(table_name))
        if (class_uri, RDF.type, OWL.Class) not in g:
            report.add_error(f"Missing class for table {table_name}")

        fk_cols = {fk.column: fk for fk in table.foreign_keys}
        for col in table.columns.values():
            if is_etl_column(col.name):
                continue
            if col.name in fk_cols:
                fk = fk_cols[col.name]
                op = uri(object_property_name_for_fk(fk, table_name))
                if (op, RDF.type, OWL.ObjectProperty) not in g:
                    report.add_error(f"Missing ObjectProperty for FK {table_name}.{col.name} -> {fk.references_table}.{fk.references_column}")
                dp = uri(data_property_name(table_name, col.name))
                if (dp, RDF.type, OWL.DatatypeProperty) in g:
                    report.add_error(f"FK column {table_name}.{col.name} also appears as DatatypeProperty {dp}")
            else:
                dp = uri(data_property_name(table_name, col.name))
                if (dp, RDF.type, OWL.DatatypeProperty) not in g:
                    report.add_error(f"Missing DatatypeProperty for non-FK column {table_name}.{col.name}")

    # Annotation/provenance completeness warnings for named classes/properties.
    for elem in set(g.subjects(RDF.type, OWL.Class)) | obj_props | data_props:
        if isinstance(elem, BNode):
            continue
        if not list(g.objects(elem, RDFS.label)):
            report.add_warning(f"Missing rdfs:label for {elem}")
        if not list(g.objects(elem, RDFS.comment)):
            report.add_warning(f"Missing rdfs:comment for {elem}")
        if not list(g.objects(elem, PROV.wasDerivedFrom)):
            report.add_warning(f"Missing prov:wasDerivedFrom for {elem}")

    return report


def merge_graph_strict(core: Graph, delta: Graph) -> None:
    """Merge delta, replacing duplicate labels/comments from same subject only by union semantics."""
    core += delta

# =========================================================
# MAIN PIPELINE
# =========================================================

def run_semantic_enrichment(
    schema_path: Path | str = SCHEMA_PATH,
    docs_path: Path | str = DOCS_PATH,
    ontology_path: Path | str = ONTOLOGY_PATH,
    core_owl_path: Path | str = CORE_OWL_PATH,
    output_path: Path | str = OUTPUT_PATH,
    direct_mappings_dir: Path | str = DIRECT_MAPPINGS_DIR,
    model: str = DEFAULT_MODEL,
    use_llm: bool = True,
) -> None:
    print("=" * 70)
    print("RIGOR")
    print("=" * 70)

    output_path = Path(output_path)
    direct_mappings_dir = Path(direct_mappings_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    direct_mappings_dir.mkdir(parents=True, exist_ok=True)

    print("\n1. Loading and cleaning schema...")
    schema = load_schema_from_json(schema_path)
    print(f"   {len(schema)} tables after schema load")

    print("\n2. Initialising local embedding model...")
    _ = embed_texts(["warmup"])
    print(f"   Model: {EMBED_MODEL_NAME}")

    print("\n3. Building document FAISS index...")
    docs = load_text_documents(docs_path)
    doc_chunks: List[str] = []
    for filename, content in docs.items():
        for chunk in chunk_text(content):
            doc_chunks.append(f"[{filename}]\n{chunk}")
    doc_index = build_faiss_index_from_embeddings(embed_texts(doc_chunks)) if doc_chunks else None
    print(f"   Indexed {len(doc_chunks)} document chunks")

    print("\n4. Building external ontology FAISS index...")
    onto_chunks = load_external_ontologies(ontology_path)
    onto_index = build_faiss_index_from_embeddings(embed_texts(onto_chunks)) if onto_chunks else None
    print(f"   Indexed {len(onto_chunks)} ontology chunks")

    print("\n5. Loading seed core ontology...")
    core = Graph()
    ensure_namespaces(core)
    core_owl_path = Path(core_owl_path)
    if core_owl_path.exists():
        try:
            core.parse(str(core_owl_path), format="xml")
            ensure_namespaces(core)
            print(f"   Loaded {len(core)} seed triples from {core_owl_path}")
        except Exception as e:
            print(f"   Warning: failed to load seed core ontology: {e}")
    else:
        print("   No seed core ontology found; starting with an empty core")

    gen_llm: Optional[OntologyLLM] = None
    judge_llm: Optional[JudgeLLM] = None
    if use_llm:
        print("\n6. Initialising LLM clients...")
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("   OPENROUTER_API_KEY is not set; continuing in deterministic Direct Mapping only mode")
            use_llm = False
        else:
            llm_client = OpenRouterLLM(api_key, model=model)
            gen_llm = OntologyLLM(llm_client)
            judge_llm = JudgeLLM(llm_client)
            print(f"   Model: {model}")

    print("\n7. Computing FK-guided traversal order...")
    traversal_order = traverse_schema_fk_order(schema)
    for i, t in enumerate(traversal_order, 1):
        print(f"   {i}. {t}")

    print("\n8. Processing tables...")
    for idx, table_name in enumerate(traversal_order, start=1):
        print(f"\n{'=' * 60}")
        print(f"[{idx}/{len(traversal_order)}] {table_name}")
        print(f"{'=' * 60}")

        try:
            retrieval_query = build_retrieval_query(table_name, schema)
            print(f"   Retrieval query: {retrieval_query}")

            # A. Deterministic Direct Mapping: merged first to guarantee coverage.
            dm_graph = direct_mapping_graph_for_table(table_name, schema)
            merge_graph_strict(core, dm_graph)
            dm_text = render_direct_mapping_manchester(table_name, schema)
            dm_file = direct_mappings_dir / f"{table_name}_direct.owl"
            dm_file.write_text(dm_text, encoding="utf-8")
            print(f"   Direct mapping merged and saved: {dm_file}")

            # B. Retrieve context from current O_t, docs, and external ontologies.
            core_chunks = build_core_ontology_chunks(core)
            core_index = build_faiss_index_from_embeddings(embed_texts(core_chunks)) if core_chunks else None
            core_ctx = retrieve_top_k(retrieval_query, core_index, core_chunks, k=3) if core_index is not None else []
            docs_ctx = retrieve_top_k(retrieval_query, doc_index, doc_chunks, k=3) if doc_index is not None else []
            onto_ctx = retrieve_top_k(retrieval_query, onto_index, onto_chunks, k=3) if onto_index is not None else []
            print(f"   Retrieved {len(core_ctx)} core chunks, {len(docs_ctx)} doc chunks, {len(onto_ctx)} external chunks")

            schema_str = cleaned_schema_json_for_table(table_name, schema)
            fk_str = format_foreign_keys(table_name, schema)
            core_text = "\n\n".join(core_ctx) if core_ctx else "None"
            doc_text = "\n\n".join(docs_ctx) if docs_ctx else "None"
            ext_text = "\n\n".join(onto_ctx) if onto_ctx else "None"

            
            # C. Optional LLM enrichment, Judge validation, deterministic validation, and retry.
            if use_llm and gen_llm and judge_llm:
                correction_hint = ""
                accepted_delta_graph = None

                for attempt in range(1, MAX_JUDGE_RETRIES + 2):
                    print(f"   Gen-LLM enrichment attempt {attempt}...")

                    delta_raw = gen_llm.generate(
                        table_name=table_name,
                        direct_mapping=dm_text,
                        schema_str=schema_str,
                        foreign_keys_str=fk_str,
                        documents=doc_text,
                        core_context=core_text,
                        external_context=ext_text,
                        correction_hint=correction_hint,
                    )

                    if not delta_raw:
                        correction_hint = (
                            "The previous Gen-LLM response was empty. "
                            "Return a valid Manchester Syntax delta ontology."
                        )
                        continue

                    print(f"   Judge-LLM validation attempt {attempt}...")
                    verdict = judge_llm.judge(
                        table_name=table_name,
                        direct_mapping=dm_text,
                        delta_ontology=delta_raw,
                        schema_context=schema_str,
                        foreign_keys_str=fk_str,
                        core_context=core_text,
                    )

                    print(f"   Decision: {verdict['decision']}")

                    if verdict["critical"]:
                        for issue in verdict["critical"]:
                            print(f"     Critical: {issue}")

                    # Judge rejected: send Judge critical issues back to Gen-LLM.
                    if verdict["decision"] == "REJECTED":
                        correction_hint = (
                            "The previous attempt was rejected by the Judge-LLM. "
                            "Fix ALL of these critical issues:\n"
                            + "\n".join(f"- {x}" for x in verdict["critical"])
                        )
                        continue

                    # Judge approved or corrected: choose the fragment to parse.
                    accepted_text = (
                        verdict["corrected"]
                        if verdict["decision"] == "APPROVED_WITH_CORRECTIONS" and verdict["corrected"]
                        else delta_raw
                    )

                    # Parse and run deterministic graph validation before merge.
                    delta_graph = parse_llm_ontology(accepted_text)

                    candidate = Graph()
                    ensure_namespaces(candidate)
                    candidate += core
                    candidate += delta_graph

                    # Important: validate only tables processed so far, not future tables.
                    processed_schema = {
                        t: schema[t]
                        for t in traversal_order[:idx]
                    }

                    report = validate_graph_against_rigor(candidate, processed_schema)

                    if report.ok:
                        accepted_delta_graph = delta_graph
                        print(f"   Deterministic graph validation passed")
                        break

                    # Deterministic validation failed: send those errors back to Gen-LLM.
                    print("   Deterministic graph validation failed:")
                    for err in report.errors[:20]:
                        print(f"     - {err}")

                    correction_hint = (
                        "The previous fragment passed or was corrected by the Judge-LLM, "
                        "but failed deterministic RDF graph validation after parsing. "
                        "Fix ALL of these structural errors:\n"
                        + "\n".join(f"- {x}" for x in report.errors[:20])
                        + "\nReturn only corrected Manchester Syntax."
                    )

                if accepted_delta_graph is not None:
                    merge_graph_strict(core, accepted_delta_graph)
                    print(f"   Delta merged: {len(accepted_delta_graph)} triples")
                else:
                    print("   No valid semantic delta after retries; deterministic mapping retained")
            table_report = validate_graph_against_rigor(core, {table_name: schema[table_name]})
            if not table_report.ok:
                raise RuntimeError("Post-table validation failed:\n" + "\n".join(table_report.errors))
            if table_report.warnings:
                print(f"   Warnings: {len(table_report.warnings)} annotation/provenance warnings")

            n_cls = len(set(core.subjects(RDF.type, OWL.Class)))
            n_dp = len(set(core.subjects(RDF.type, OWL.DatatypeProperty)))
            n_op = len(set(core.subjects(RDF.type, OWL.ObjectProperty)))
            print(f"   Core now has {n_cls} classes, {n_dp} data properties, {n_op} object properties")

        except Exception as e:
            print(f"   ERROR processing {table_name}: {e}")
            raise

    print("\n9. Final deterministic RIGOR validation...")
    final_report = validate_graph_against_rigor(core, schema)
    if not final_report.ok:
        print("   Final validation failed:")
        for err in final_report.errors:
            print(f"     - {err}")
        raise RuntimeError("Final ontology failed deterministic RIGOR validation")
    if final_report.warnings:
        print(f"   Final validation passed with {len(final_report.warnings)} warnings")
        for warn in final_report.warnings[:20]:
            print(f"     Warning: {warn}")
        if len(final_report.warnings) > 20:
            print(f"     ... {len(final_report.warnings) - 20} more warnings")
    else:
        print("   Final validation passed with no warnings")

    print("\n10. Saving final ontology...")
    ensure_namespaces(core)
    core.add((URIRef(ONTOLOGY_IRI), RDF.type, OWL.Ontology))
    core.add((URIRef(ONTOLOGY_IRI), PROV.generatedAtTime, Literal(datetime.now(timezone.utc).isoformat(), datatype=XSD.dateTime)))
    core.serialize(str(output_path), format="xml")
    print(f"\nDone -> {output_path} ({len(core)} triples)")
    print("=" * 70)

# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    run_semantic_enrichment()
