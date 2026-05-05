"""
ontology_checker.py — Consistency and Pitfall Checker for RIGOR Ontologies

Runs two external tools on every ontology:
  1. HermiT reasoner (via JAR) — logical consistency check
  2. OOPS! REST API           — ontology pitfall detection

Results are merged into the existing evaluation JSON files produced by
eval.py WITHOUT overwriting any other keys. Adds/updates only:
  - hermit_consistency : {consistent, output, stderr, error}
  - oops_pitfalls      : [{code, name, importance, count, description}, ...]
  - oops_raw_counts    : {critical: N, important: N, minor: N}

Prerequisites:
  - Java installed (java on PATH)
  - HermiT.jar downloaded — set path in HERMIT_JAR below
    Download: https://github.com/owlcs/HermiT/releases
  - Internet access for OOPS! API calls
  - eval.py must have been run first so the JSON files exist
    (script will create them if missing, just with checker results)

Usage:
  python ontology_checker.py
"""

import os
import json
import subprocess
import tempfile
import requests
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rdflib import Graph, URIRef, Namespace, RDF, OWL
from rdflib.namespace import RDFS

# =========================================================
# CONFIGURATION — update paths as needed
# =========================================================

BASE_PATH      = "YOUR_BASE_PATH"
EVAL_DIR       = f"{BASE_PATH}/evaluation"
HERMIT_JAR     = f"{BASE_PATH}/jars/HermiT.jar"   # download if not present
JAVA_EXE       = "java"
HERMIT_TIMEOUT = 180    # seconds per ontology (increase for large ontologies)
OOPS_ENDPOINT  = "https://oops.linkeddata.es/rest"
OOPS_TIMEOUT   = 300    # OOPS can be slow for large ontologies

# Ontologies to check — must match the ONTOLOGIES dict in eval.py
ONTOLOGIES = {
   # "real_world": {
       # "direct_mapping": f"{BASE_PATH}/output/direct_mapping/direct_mapping_rd.owl",
      #  "baseline_claude":    f"{BASE_PATH}/output/baseline/real_world/claude/baseline_ontology.owl",
       # "baseline_mistral":   f"{BASE_PATH}/output/baseline/real_world/mistral/baseline_ontology.owl",
      #  "baseline_deepseek":  f"{BASE_PATH}/output/baseline/real_world/deepseek/baseline_ontology.owl",
      #  "non_iterative_claude":   f"{BASE_PATH}/output/non_iterative/real_world/claude/non_iterative_ontology.owl",
       # "non_iterative_mistral":  f"{BASE_PATH}/output/non_iterative/real_world/mistral/non_iterative_ontology.owl",
       # "non_iterative_deepseek": f"{BASE_PATH}/output/non_iterative/real_world/deepseek/non_iterative_ontology.owl",
      #  "RIGOR_claude":   f"{BASE_PATH}/output/RIGOR/real_world/claude/enriched_ontology.owl",
     #   "RIGOR_mistral":  f"{BASE_PATH}/output/RIGOR/real_world/mistral/enriched_ontology.owl",
    # "RIGOR_deepseek": f"{BASE_PATH}/output/RIGOR/real_world/deepseek/enriched_ontology.owl",

   # },
      #  "eicu_crd": {
        #"direct_mapping":    f"{BASE_PATH}/output/direct_mapping/direct_mapping_icu.owl",
        #"baseline_claude":   f"{BASE_PATH}/output/baseline/eicu_crd/claude/baseline_ontology.owl",
        #"baseline_mistral":  f"{BASE_PATH}/output/baseline/eicu_crd/mistral/baseline_ontology.owl",
        #"baseline_deepseek": f"{BASE_PATH}/output/baseline/eicu_crd/deepseek/baseline_ontology.owl",
        #"non_iterative_claude":   f"{BASE_PATH}/output/non_iterative/eicu_crd/claude/non_iterative_ontology.owl",
        #"non_iterative_mistral":  f"{BASE_PATH}/output/non_iterative/eicu_crd/mistral/non_iterative_ontology.owl",
        #"non_iterative_deepseek": f"{BASE_PATH}/output/non_iterative/eicu_crd/deepseek/non_iterative_ontology.owl",
      #  "RIGOR_claude":   f"{BASE_PATH}/output/RIGOR/eicu_crd/claude/enriched_ontology.owl",
      #  "RIGOR_mistral":  f"{BASE_PATH}/output/RIGOR/eicu_crd/mistral/enriched_ontology.owl",
      #  "RIGOR_deepseek": f"{BASE_PATH}/output/RIGOR/eicu_crd/deepseek/enriched_ontology.owl",
    #} 
    
        "chinook": {
        "direct_mapping":    f"{BASE_PATH}/output/direct_mapping/direct_mapping_chinook.owl",
        "baseline_claude":   f"{BASE_PATH}/output/baseline/chinook/claude/baseline_ontology.owl",
        "baseline_mistral":  f"{BASE_PATH}/output/baseline/chinook/mistral/baseline_ontology.owl",
        "baseline_deepseek": f"{BASE_PATH}/output/baseline/chinook/deepseek/baseline_ontology.owl",
        "non_iterative_claude":   f"{BASE_PATH}/output/non_iterative/chinook/claude/non_iterative_ontology.owl",
        "non_iterative_mistral":  f"{BASE_PATH}/output/non_iterative/chinook/mistral/non_iterative_ontology.owl",
        "non_iterative_deepseek": f"{BASE_PATH}/output/non_iterative/chinook/deepseek/non_iterative_ontology.owl",
        "RIGOR_claude":   f"{BASE_PATH}/output/RIGOR/chinook/claude/enriched_ontology.owl",
        "RIGOR_mistral":  f"{BASE_PATH}/output/RIGOR/chinook/mistral/enriched_ontology.owl",
        "RIGOR_deepseek": f"{BASE_PATH}/output/RIGOR/chinook/deepseek/enriched_ontology.owl",
    }
}

# =========================================================
# SYNTAX CHECK — rdflib parsing
# =========================================================

def check_syntax(owl_path: str) -> Dict:
    if not os.path.exists(owl_path):
        return {
            "syntax_valid":  False,
            "parse_format":  None,
            "triple_count":  0,
            "error_message": f"File not found: {owl_path}",
            "issues":        [],
        }

    try:
        with open(owl_path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except Exception as e:
        return {
            "syntax_valid":  False,
            "parse_format":  None,
            "triple_count":  0,
            "error_message": str(e),
            "issues":        [],
        }

    issues = []

    sparql_lines = [
        line.strip() for line in raw.splitlines()
        if re.search(r"\?[A-Za-z]", line) and not line.strip().startswith("#")
    ]
    if sparql_lines:
        issues.append({
            "type":    "sparql_variables",
            "count":   len(sparql_lines),
            "example": sparql_lines[0][:120],
        })

    angle_bracket_uris = re.findall(r"http[s]?://\S+#<http[s]?://[^>]+>", raw)
    if angle_bracket_uris:
        issues.append({
            "type":    "angle_bracket_uris",
            "count":   len(angle_bracket_uris),
            "example": angle_bracket_uris[0][:120],
        })

    prefix_counts = {}
    for m in re.finditer(r"@prefix\s+(\S+):", raw):
        p = m.group(1)
        prefix_counts[p] = prefix_counts.get(p, 0) + 1
    dup_prefixes = {p: c for p, c in prefix_counts.items() if c > 1}
    if dup_prefixes:
        issues.append({
            "type":    "duplicate_prefixes",
            "count":   len(dup_prefixes),
            "example": list(dup_prefixes.keys())[:5],
        })

    if "```" in raw:
        issues.append({
            "type":  "markdown_fences",
            "count": raw.count("```"),
        })

    g, parse_format, parse_error = load_graph_with_manchester_fallback(owl_path)

    return {
        "syntax_valid":  g is not None,
        "parse_format":  parse_format,
        "triple_count":  len(g) if g is not None else 0,
        "error_message": "" if g is not None else (parse_error or "Unknown parse error"),
        "issues":        issues,
    }


# =========================================================
# HERMIT CONSISTENCY CHECK
# =========================================================
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
    g.bind("prov", prov)

    XSD_TYPES = {
        "xsd:string": RDFS.Literal,
        "xsd:integer": URIRef(str(Namespace("http://www.w3.org/2001/XMLSchema#")) + "integer"),
        "xsd:float": URIRef(str(Namespace("http://www.w3.org/2001/XMLSchema#")) + "float"),
        "xsd:boolean": URIRef(str(Namespace("http://www.w3.org/2001/XMLSchema#")) + "boolean"),
        "xsd:date": URIRef(str(Namespace("http://www.w3.org/2001/XMLSchema#")) + "date"),
        "xsd:dateTime": URIRef(str(Namespace("http://www.w3.org/2001/XMLSchema#")) + "dateTime"),
        "xsd:time": URIRef(str(Namespace("http://www.w3.org/2001/XMLSchema#")) + "time"),
        "xsd:decimal": URIRef(str(Namespace("http://www.w3.org/2001/XMLSchema#")) + "decimal"),
        "xsd:double": URIRef(str(Namespace("http://www.w3.org/2001/XMLSchema#")) + "double"),
        "xsd:base64Binary": URIRef(str(Namespace("http://www.w3.org/2001/XMLSchema#")) + "base64Binary"),
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

    def local_name(value: str) -> str:
        value = value.strip()
        if value.startswith(":"):
            return value[1:]
        return value

    for block in blocks:
        header = block[0]
        rest = block[1:]

        def get_value(prefix: str):
            for ln in rest:
                if ln.startswith(prefix):
                    return ln[len(prefix):].strip()
            return None

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

def load_graph_with_manchester_fallback(owl_path: str) -> Tuple[Optional[Graph], Optional[str], Optional[str]]:
    """
    Returns (graph, parse_format, error_message)
    parse_format is one of: xml, turtle, n3, manchester
    """
    if not os.path.exists(owl_path):
        return None, None, f"File not found: {owl_path}"

    try:
        with open(owl_path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except Exception as e:
        return None, None, str(e)

    stripped = raw.lstrip()

    # Manchester fallback
    if stripped.startswith("Prefix:") or stripped.startswith("Ontology:") or "Class:" in stripped:
        try:
            g = parse_manchester_direct_mapping(raw)
            return g, "manchester", None
        except Exception as e:
            return None, None, f"Manchester parsing failed: {e}"

    # Standard RDF formats
    for fmt in ("xml", "turtle", "n3"):
        try:
            g = Graph()
            g.parse(owl_path, format=fmt)
            return g, fmt, None
        except Exception:
            continue

    return None, None, "Could not parse ontology with xml/turtle/n3 or Manchester fallback"


def check_hermit(owl_path: str) -> Dict:
    """
    Run HermiT reasoner on an OWL file and return consistency result.
    HermiT exits with code 0 and prints 'is satisfiable' if consistent.
    Returns:
        {consistent: bool|None, output: str, stderr: str, error: str}
    """
    if not os.path.exists(HERMIT_JAR):
        return {
            "consistent": None,
            "output": "",
            "stderr": "",
            "error": f"HermiT JAR not found: {HERMIT_JAR}. "
                     f"Download from https://github.com/owlcs/HermiT/releases",
        }

    if not os.path.exists(owl_path):
        return {"consistent": None, "output": "", "stderr": "",
                "error": f"Ontology file not found: {owl_path}"}

    # HermiT needs the file as a proper IRI
    file_iri = Path(owl_path).resolve().as_uri()

    cmd = [
        JAVA_EXE,
        "-cp", HERMIT_JAR,
        "org.semanticweb.HermiT.cli.CommandLine",
        "--ignoreUnsupportedDatatypes",
        "-k",              # consistency check only
        file_iri,
    ]

    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=HERMIT_TIMEOUT,
        )
        out  = (p.stdout or "").strip()
        err  = (p.stderr or "").strip()
        blob = (out + "\n" + err).lower()

        if "inconsistentontologyexception" in blob or "inconsistent ontology" in blob:
            consistent = False
        elif "is satisfiable" in blob:
            consistent = True
        elif "unsatisfiable" in blob or "is not satisfiable" in blob:
            consistent = False
        elif p.returncode == 0:
            consistent = True   # HermiT returned 0 — treat as consistent
        else:
            consistent = None   # Unknown — parse error or other failure

        return {
            "consistent":   consistent,
            "return_code":  p.returncode,
            "output":       out[:2000],   # truncate to keep JSON small
            "stderr":       err[:2000],
            "error":        "",
        }

    except subprocess.TimeoutExpired:
        return {
            "consistent": None,
            "output": "",
            "stderr": "",
            "error": f"HermiT timed out after {HERMIT_TIMEOUT}s",
        }
    except Exception as e:
        return {"consistent": None, "output": "", "stderr": "", "error": str(e)}


# =========================================================
# OOPS! PITFALL DETECTION
# =========================================================

OOPS_NS = Namespace("http://oops.linkeddata.es/def#")

def load_owl_as_rdfxml(owl_path: str) -> Optional[str]:
    """
    Load ontology (including Manchester direct mapping files) and serialize as RDF/XML for OOPS.
    """
    g, parse_format, error = load_graph_with_manchester_fallback(owl_path)
    if g is None:
        return None
    return g.serialize(format="xml")


def build_oops_request(rdfxml: str) -> str:
    safe = rdfxml.replace("]]>", "]]]]><![CDATA[>")
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<OOPSRequest>
  <OntologyUrl></OntologyUrl>
  <OntologyContent><![CDATA[{safe}]]></OntologyContent>
  <Pitfalls></Pitfalls>
  <OutputFormat>TURTLE</OutputFormat>
</OOPSRequest>"""


def call_oops(rdfxml: str) -> Optional[str]:
    """POST ontology to OOPS REST API. Returns raw response text or None."""
    xml_req = build_oops_request(rdfxml)
    try:
        r = requests.post(
            OOPS_ENDPOINT,
            data=xml_req.encode("utf-8"),
            headers={
                "Content-Type": "application/xml",
                "Accept": "text/turtle, application/rdf+xml;q=0.9, */*;q=0.1",
            },
            timeout=OOPS_TIMEOUT,
        )
        if not r.ok:
            print(f"    OOPS HTTP {r.status_code}: {r.text[:200]}")
            return None
        return r.text
    except requests.exceptions.Timeout:
        print(f"    OOPS timed out after {OOPS_TIMEOUT}s")
        return None
    except Exception as e:
        print(f"    OOPS request failed: {e}")
        return None


def parse_oops_response(response_text: str) -> List[Dict]:
    """
    Parse OOPS response graph and extract pitfalls.
    Returns a list of pitfall dicts with code, name, importance, count, description.
    """
    g = Graph()
    try:
        g.parse(data=response_text, format="turtle")
    except Exception:
        try:
            g.parse(data=response_text, format="xml")
        except Exception as e:
            print(f"    Could not parse OOPS response: {e}")
            return []

    def first_lit(node, pred) -> str:
        for o in g.objects(node, pred):
            return str(o)
        return ""

    pitfalls = []
    for node in g.subjects(RDF.type, OOPS_NS.pitfall):
        code        = first_lit(node, OOPS_NS.hasCode).strip()
        name        = first_lit(node, OOPS_NS.hasName).strip()
        importance  = first_lit(node, OOPS_NS.hasImportanceLevel).strip()
        count       = first_lit(node, OOPS_NS.hasNumberAffectedElements).strip()
        description = first_lit(node, OOPS_NS.hasDescription).strip()

        pitfalls.append({
            "code":        code,
            "name":        name,
            "importance":  importance,
            "count":       int(count) if count.isdigit() else count,
            "description": description,
        })

    # Sort: critical first, then important, then minor
    importance_order = {"critical": 0, "important": 1, "minor": 2}
    pitfalls.sort(key=lambda p: importance_order.get(p["importance"].lower(), 3))

    return pitfalls


def check_oops(owl_path: str) -> Dict:
    """
    Run OOPS! pitfall detection on an OWL file.
    Returns:
        {pitfalls: [...], raw_counts: {critical, important, minor}, error: str}
    """
    if not os.path.exists(owl_path):
        return {"pitfalls": [], "raw_counts": {}, "error": f"File not found: {owl_path}"}

    rdfxml = load_owl_as_rdfxml(owl_path)
    if rdfxml is None:
        return {"pitfalls": [], "raw_counts": {}, "error": "Could not parse ontology"}

    response_text = call_oops(rdfxml)
    if response_text is None:
        return {"pitfalls": [], "raw_counts": {}, "error": "OOPS API call failed"}

    pitfalls = parse_oops_response(response_text)

    # Count by importance level
    raw_counts = {"critical": 0, "important": 0, "minor": 0}
    for p in pitfalls:
        level = p["importance"].lower()
        if level in raw_counts:
            raw_counts[level] += 1

    return {
        "pitfalls":   pitfalls,
        "raw_counts": raw_counts,
        "error":      "",
    }


# =========================================================
# JSON FILE UPDATE — safe merge, never destroys existing keys
# =========================================================

def load_eval_json(json_path: str) -> Dict:
    """Load existing evaluation JSON or return empty dict."""
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"    Warning: could not load {json_path}: {e}")
    return {}


def save_eval_json(json_path: str, data: Dict) -> None:
    """Save JSON to file, creating parent directories if needed."""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# =========================================================
# MAIN
# =========================================================

def main():
    os.makedirs(EVAL_DIR, exist_ok=True)

    # Check HermiT JAR availability upfront
    hermit_available = os.path.exists(HERMIT_JAR)
    if not hermit_available:
        print(f"WARNING: HermiT JAR not found at {HERMIT_JAR}")
        print("  Download from: https://github.com/owlcs/HermiT/releases")
        print("  Consistency checks will be skipped.\n")

    total    = sum(len(v) for v in ONTOLOGIES.values())
    current  = 0

    for db_name, ontologies in ONTOLOGIES.items():
        print(f"\n{'='*60}")
        print(f"Database: {db_name}")
        print(f"{'='*60}")

        for label, owl_path in ontologies.items():
            current += 1
            print(f"\n[{current}/{total}] {label}")
            print(f"  File: {owl_path}")

            if not os.path.exists(owl_path):
                print("  Skipped — file not found")
                continue

            json_path = os.path.join(EVAL_DIR, f"{db_name}_{label}.json")
            eval_data = load_eval_json(json_path)

            # ── 1. Syntax check (rdflib) ─────────────────────────
            print("  Running syntax check...")
            syntax_result = check_syntax(owl_path)
            status_str = "VALID" if syntax_result["syntax_valid"] else "INVALID"
            print(f"  Syntax: {status_str} "
                  f"(format: {syntax_result['parse_format'] or 'none'}, "
                  f"triples: {syntax_result['triple_count']})")
            if syntax_result["issues"]:
                for issue in syntax_result["issues"]:
                    print(f"    Issue: {issue['type']} (count: {issue.get('count', '?')})")

            # ── 2. HermiT consistency check ───────────────────────
            if hermit_available:
                print("  Running HermiT consistency check...")
                hermit_result = check_hermit(owl_path)
                status = {True: "CONSISTENT", False: "INCONSISTENT",
                          None: "UNKNOWN"}.get(hermit_result["consistent"], "UNKNOWN")
                print(f"  HermiT result: {status}")
                if hermit_result.get("error"):
                    print(f"  HermiT error: {hermit_result['error']}")
            else:
                hermit_result = {
                    "consistent": None,
                    "output": "",
                    "stderr": "",
                    "error": "HermiT JAR not available",
                }

            # ── 3. OOPS! pitfall detection ────────────────────────
            print("  Running OOPS! pitfall detection...")
            oops_result = check_oops(owl_path)
            if oops_result.get("error"):
                print(f"  OOPS error: {oops_result['error']}")
            else:
                counts = oops_result["raw_counts"]
                print(f"  OOPS pitfalls: "
                      f"{counts.get('critical',0)} critical, "
                      f"{counts.get('important',0)} important, "
                      f"{counts.get('minor',0)} minor")

            # ── 4. Merge into existing JSON ───────────────────────
            eval_data["syntax_check"]       = syntax_result
            eval_data["hermit_consistency"] = hermit_result
            eval_data["oops_pitfalls"]      = oops_result.get("pitfalls", [])
            eval_data["oops_raw_counts"]    = oops_result.get("raw_counts", {})

            # Ensure label and path are set (in case JSON was empty)
            eval_data.setdefault("label",    label)
            eval_data.setdefault("owl_path", owl_path)

            save_eval_json(json_path, eval_data)
            print(f"  Updated: {json_path}")

    # ── Summary table ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Label':<30} {'Syntax':>8} {'Consistent':>12} {'Critical':>9} {'Important':>10} {'Minor':>6}")
    print("-" * 78)

    for db_name, ontologies in ONTOLOGIES.items():
        for label in ontologies:
            json_path = os.path.join(EVAL_DIR, f"{db_name}_{label}.json")
            if not os.path.exists(json_path):
                continue
            data      = load_eval_json(json_path)
            hermit    = data.get("hermit_consistency", {})
            counts    = data.get("oops_raw_counts", {})
            consistent = {True: "YES", False: "NO",
                         None: "?"}.get(hermit.get("consistent"), "?")
            syntax    = data.get("syntax_check", {})
            syn_valid = "OK" if syntax.get("syntax_valid") else "FAIL"
            print(
                f"{label:<30} {syn_valid:>8} {consistent:>12} "
                f"{counts.get('critical',0):>9} "
                f"{counts.get('important',0):>10} "
                f"{counts.get('minor',0):>6}"
            )

    print(f"\nAll results saved to: {EVAL_DIR}")
    print("Each ontology's JSON file now contains:")
    print("  syntax_check       — {syntax_valid, parse_format, triple_count, error_message, issues}")
    print("  hermit_consistency — {consistent: bool, output, stderr, error}")
    print("  oops_pitfalls      — [{code, name, importance, count, description}, ...]")
    print("  oops_raw_counts    — {critical: N, important: N, minor: N}")


if __name__ == "__main__":
    main()