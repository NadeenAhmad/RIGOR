"""
Direct Mapping Generator
Converts a SQL schema JSON to OWL 2 Manchester Syntax.

Improvements over original:
  - ETL / system artefact columns are filtered out before generation
  - Date / time columns that are typed as FLOAT or TEXT are corrected
    to xsd:date or xsd:dateTime
  - Self-referential ObjectProperties are suppressed (lookup tables)
  - Provenance individuals included
  - Per-table Manchester files can also be generated (used by app.py)

Set INPUT_JSON and OUTPUT_FILE below, then:
    python mapping.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

# =========================================================
# CONFIGURE THESE
# =========================================================

BASE_PATH    = "YOUR_BASE_PATH"

INPUT_JSON   = f"{BASE_PATH}/sql_schema/schema_chinook.json"
OUTPUT_FILE  = f"{BASE_PATH}/output/direct_mapping_chinook.owl"
ONTOLOGY_IRI = "http://example.org/ontology"

# =========================================================
# ETL / SYSTEM ARTEFACT FILTER
# Columns matching these rules have no clinical meaning and
# should not appear in the ontology.  Extend as needed when
# onboarding a new schema.
# =========================================================

ETL_PREFIXES = ("s_",)
ETL_EXACT    = {"s_ColLineage", "s_Generation", "s_GUID", "s_Lineage"}

def is_etl_column(col_name: str) -> bool:
    """Return True if the column is a database/ETL system artefact."""
    return col_name in ETL_EXACT or any(col_name.startswith(p) for p in ETL_PREFIXES)

# =========================================================
# SQL TO XSD TYPE MAP
# =========================================================

SQL_TO_XSD = {
    "INTEGER":   "xsd:integer",
    "INT":       "xsd:integer",
    "SMALLINT":  "xsd:integer",
    "BIGINT":    "xsd:integer",
    "TINYINT":   "xsd:boolean",
    "BOOLEAN":   "xsd:boolean",
    "BOOL":      "xsd:boolean",
    "FLOAT":     "xsd:float",
    "DOUBLE":    "xsd:double",
    "DECIMAL":   "xsd:decimal",
    "NUMERIC":   "xsd:decimal",
    "TEXT":      "xsd:string",
    "VARCHAR":   "xsd:string",
    "CHAR":      "xsd:string",
    "CLOB":      "xsd:string",
    "TIMESTAMP": "xsd:dateTime",
    "DATETIME":  "xsd:dateTime",
    "DATE":      "xsd:date",
    "TIME":      "xsd:time",
    "BLOB":      "xsd:base64Binary",
}

def get_xsd_type(sql_type: str) -> str:
    base = str(sql_type).upper().split("(")[0].strip()
    return SQL_TO_XSD.get(base, "xsd:string")

def fix_date_type(col_name: str, xsd_type: str) -> str:
    """
    Correct date/time columns that were stored as FLOAT or TEXT in the
    database schema.  Any column whose name contains 'date', 'time',
    'timestamp', or 'created_at' is promoted to xsd:date or xsd:dateTime.
    """
    lower = col_name.lower()
    if xsd_type in ("xsd:float", "xsd:string"):
        if "datetime" in lower or "timestamp" in lower or lower == "created_at":
            return "xsd:dateTime"
        if "date" in lower or "time" in lower:
            return "xsd:date"
    return xsd_type

def to_class_name(table_name: str) -> str:
    """'patient_data' -> 'PatientData'"""
    return "".join(part.capitalize() for part in table_name.split("_"))

# =========================================================
# SCHEMA PARSING
# =========================================================

def parse_table(table_name, table_value):
    """
    Returns:
        columns     : { col_name: xsd_type_string }  (ETL columns removed)
        foreign_keys: { col_name: (referenced_table, referenced_col) }
    """
    columns      = {}
    foreign_keys = {}

    if "columns" in table_value and isinstance(table_value["columns"], dict):
        raw_cols = table_value["columns"]
        for col, dtype in raw_cols.items():
            if is_etl_column(col):
                continue                          # ← drop ETL artefacts
            xsd = get_xsd_type(dtype)
            xsd = fix_date_type(col, xsd)         # ← correct date types
            columns[col] = xsd

        for fk in table_value.get("foreign_keys", []):
            col       = fk.get("column")
            ref_table = fk.get("references_table")
            ref_col   = fk.get("references_column")
            if col and ref_table and ref_col:
                foreign_keys[col] = (ref_table, ref_col)
    else:
        # Flat format — apply same filters
        for col, dtype in table_value.items():
            if is_etl_column(col):
                continue
            xsd = get_xsd_type(dtype)
            xsd = fix_date_type(col, xsd)
            columns[col] = xsd

    return columns, foreign_keys

def detect_primary_key(table_name, columns):
    """
    Detect PK from column names:
      1. 'id'
      2. '<table>_id'
      3. '<table_singularized>_id'
      4. First integer column
      5. First column (fallback)
    """
    col_names = list(columns.keys())

    if "id" in col_names:
        return "id"
    for candidate in [f"{table_name}_id", f"{table_name.rstrip('s')}_id"]:
        if candidate in col_names:
            return candidate
    for col, dtype in columns.items():
        if dtype == "xsd:integer":
            return col
    return col_names[0]

def infer_foreign_keys(table_name, columns, all_table_names, pk_index):
    """
    Fallback FK inference for schemas without an explicit FK list.
    Uses naming convention rules only.
    """
    own_pk = pk_index[table_name]
    fks    = {}

    for col, dtype in columns.items():
        # TINYINT columns are boolean flags — never FKs
        if dtype == "xsd:boolean":
            continue
        if dtype != "xsd:integer":
            continue
        if col == own_pk:
            continue

        parent = None

        if col.endswith("_id"):
            base = col[:-3]
            if base in all_table_names and base != table_name:
                parent = base

        if parent is None and col.endswith("id") and not col.endswith("_id"):
            base = col[:-2]
            if base in all_table_names and base != table_name:
                parent = base

        if parent is None and col in all_table_names and col != table_name:
            parent = col

        if parent:
            fks[col] = (parent, pk_index.get(parent, f"{parent}_id"))

    return fks

def is_self_referential(col, ref_table, table_name):
    """
    Return True if the FK would create a self-referential ObjectProperty
    (domain == range), which is always semantically incorrect for lookup
    tables.  Such properties are suppressed; RIGOR will generate the
    correct version.
    """
    return to_class_name(ref_table) == to_class_name(table_name)

# =========================================================
# GENERATE MANCHESTER SYNTAX
# =========================================================

def generate_manchester(schema, ontology_iri):
    timestamp = datetime.now(timezone.utc).isoformat()

    # Parse all tables
    parsed = {}
    for table_name, table_value in schema.items():
        columns, fks = parse_table(table_name, table_value)
        parsed[table_name] = {"columns": columns, "fks": fks}

    pk_index = {
        t: detect_primary_key(t, parsed[t]["columns"])
        for t in parsed
    }

    # Infer missing FKs
    all_table_names = set(parsed.keys())
    for table_name, data in parsed.items():
        if not data["fks"]:
            data["fks"] = infer_foreign_keys(
                table_name, data["columns"], all_table_names, pk_index
            )

    lines = [
    f"Prefix: : <{ontology_iri}#>",
    "Prefix: xsd: <http://www.w3.org/2001/XMLSchema#>",
    "Prefix: prov: <http://www.w3.org/ns/prov#>",
    "Prefix: owl: <http://www.w3.org/2002/07/owl#>",
    "",
        f"Ontology: <{ontology_iri}>",
        "",
        "Annotations:",
        f'    prov:generatedAtTime "{timestamp}"^^xsd:dateTime,',
        "    prov:wasGeneratedBy :DirectMappingActivity",
        "",
    ]

    for table_name, data in parsed.items():
        class_name = to_class_name(table_name)
        columns    = data["columns"]
        fks        = data["fks"]
        pk         = pk_index[table_name]

        lines.append(f"Class: :{class_name}")
        lines.append(f"    HasKey: :{pk}")
        lines.append("")

        for col, (ref_table, ref_col) in fks.items():
            # Suppress self-referential ObjectProperties
            if is_self_referential(col, ref_table, table_name):
                continue
            prop_name   = f"has{to_class_name(ref_table)}"
            range_class = to_class_name(ref_table)
            lines += [
f"ObjectProperty: :{prop_name}",
f"    Domain: :{class_name}",
f"    Range:  {range_class}",
                "",
            ]

        for col, xsd_type in columns.items():
            if col == pk:
                continue
            if col in fks:
                continue
            lines += [
f"DataProperty: :{col}",
f"    Domain: :{class_name}",
                f"    Range:  :{xsd_type}",
                "",
            ]

    lines += [
"Individual: :DirectMappingActivity",
"    Types: prov:Activity",
"    Facts:",
"        prov:used :SchemaJSON,",
"        prov:wasAssociatedWith :OntologyGeneratorScript",
"",
"Individual: :SchemaJSON",
"    Types: prov:Entity",
"",
"Individual: :OntologyGeneratorScript",
"    Types: prov:SoftwareAgent",
"",
    ]

    return "\n".join(lines)

# =========================================================
# SUMMARY
# =========================================================

def print_summary(schema):
    parsed = {}
    for table_name, table_value in schema.items():
        columns, fks = parse_table(table_name, table_value)
        parsed[table_name] = {"columns": columns, "fks": fks}

    pk_index        = {t: detect_primary_key(t, parsed[t]["columns"]) for t in parsed}
    all_table_names = set(parsed.keys())

    for table_name, data in parsed.items():
        if not data["fks"]:
            data["fks"] = infer_foreign_keys(
                table_name, data["columns"], all_table_names, pk_index
            )

    total_obj  = sum(
        len([c for c, (rt, _) in d["fks"].items()
             if not is_self_referential(c, rt, t)])
        for t, d in parsed.items()
    )
    total_data = sum(
        len([c for c in d["columns"]
             if c != pk_index[t] and c not in d["fks"]])
        for t, d in parsed.items()
    )
    total_etl_removed = sum(
        len([c for c in schema[t].get("columns", schema[t]).keys()
             if is_etl_column(c)])
        for t in schema
    )

    print("\n  Schema Analysis")
    print("  " + "=" * 60)
    for table_name, data in parsed.items():
        pk  = pk_index[table_name]
        fks = data["fks"]
        print(f"\n  Class : {to_class_name(table_name)}  (PK: {pk})")
        for col, (ref_table, ref_col) in fks.items():
            if is_self_referential(col, ref_table, table_name):
                print(f"    [SUPPRESSED self-referential] ObjectProperty: has{to_class_name(ref_table)}")
            else:
                print(f"    ObjectProperty : has{to_class_name(ref_table)}"
                      f"  [{col} -> {ref_table}.{ref_col}]")
        n_data = len([c for c in data["columns"] if c != pk and c not in fks])
        print(f"    DataProperties : {n_data}")

    print(f"\n  Tables                  : {len(parsed)}")
    print(f"  Object properties       : {total_obj}")
    print(f"  Data properties         : {total_data}")
    print(f"  ETL columns removed     : {total_etl_removed}")
    print()

# =========================================================
# MAIN
# =========================================================

def main():
    input_path = Path(INPUT_JSON)

    if not input_path.exists():
        print(f"\n  ERROR: File not found:\n    {input_path}")
        print("  Please update INPUT_JSON at the top of this script.\n")
        return

    print(f"\n  Loading: {input_path.name}")
    with open(input_path, encoding="utf-8") as f:
        schema = json.load(f)
    print(f"  Tables found: {len(schema)}")

    print_summary(schema)

    owl_text = generate_manchester(schema, ONTOLOGY_IRI)

    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(owl_text)
    print(f"  Ontology written to: {output_path.resolve()}\n")


if __name__ == "__main__":
    main()
