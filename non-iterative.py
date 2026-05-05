"""
non-iterative.py — RIGOR Non-Iterative Ontology Generation

Generates ontologies in a SINGLE PASS providing the LLM with:
  - Full database schema
  - External ontologies (via FAISS retrieval)
  - Documentation context (via FAISS retrieval)

No iterative refinement, no Judge-LLM, no growing core ontology.
This isolates the contribution of RIGOR's iterative + Judge approach.

Runs for both schemas x 3 LLMs = 6 outputs.

Usage:
    export OPENROUTER_API_KEY="your_key_here"
    python3 non-iterative.py
"""

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
#import faiss
import chardet
import numpy as np

from openai import OpenAI
from rdflib import Graph, RDF, OWL, RDFS, XSD, Namespace, Literal
from sentence_transformers import SentenceTransformer
# =========================================================
# CONFIGURATION
# =========================================================

BASE_PATH = "YOUR_BASE_PATH"

SCHEMAS = {
   # "real_world": f"{BASE_PATH}/sql_schema/schema_rd.json",
    #"eicu_crd":   f"{BASE_PATH}/sql_schema/schema_icu.json",
     "chinook": f"{BASE_PATH}/sql_schema/schema_chinook.json",
}

MODELS = {
    "claude":   "anthropic/claude-opus-4-6",
    "mistral":  "mistralai/mistral-small-24b-instruct-2501",
    "deepseek": "deepseek/deepseek-chat",
}

DOCS_PATH          = f"{BASE_PATH}/documents_chinook"
EXTERNAL_ONTO_PATH = f"{BASE_PATH}/external_ontologies_chinook"
OUTPUT_BASE        = f"{BASE_PATH}/output/non_iterative"
ONTOLOGY_IRI       = "http://example.org/ontology"

# Local embeddings for FAISS retrieval.
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
SENT_MODEL = None

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "http://localhost")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "RIGOR-NonIterative")

# =========================================================
# CLIENT SETUP
# =========================================================

def get_openrouter_client() -> OpenAI:
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. "
            "Run: export OPENROUTER_API_KEY='your_key_here'"
        )

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

# =========================================================
# SCHEMA LOADING
# =========================================================

def load_schema(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_columns(table_value):
    if isinstance(table_value, dict) and "columns" in table_value:
        return table_value["columns"]
    return table_value if isinstance(table_value, dict) else {}

def format_schema_str(schema):
    """Format the full schema as a compact string for the prompt."""
    lines = []
    for table_name, table_val in schema.items():
        cols = get_columns(table_val)
        fks = table_val.get("foreign_keys", []) if isinstance(table_val, dict) else []
        lines.append(f"Table: {table_name}")
        for col, dtype in cols.items():
            lines.append(f"  - {col}: {dtype}")
        for fk in fks:
            lines.append(
                f"  FK: {fk.get('column')} -> "
                f"{fk.get('references_table')}.{fk.get('references_column')}"
            )
    return "\n".join(lines)

# =========================================================
# DOCUMENT RETRIEVAL
# =========================================================


def load_documents(doc_folder):
    docs = {}
    if not os.path.exists(doc_folder):
        print(f"  Warning: docs folder not found: {doc_folder}")
        return docs

    for root, _, files in os.walk(doc_folder):
        for filename in files:
            if not filename.endswith((".txt", ".md", ".docx")):
                continue

            path = os.path.join(root, filename)
            rel_name = os.path.relpath(path, doc_folder)

            try:
                if filename.endswith((".txt", ".md")):
                    with open(path, "rb") as f:
                        raw = f.read()
                    enc = chardet.detect(raw).get("encoding") or "utf-8"
                    with open(path, "r", encoding=enc, errors="replace") as f:
                        docs[rel_name] = f.read()

                elif filename.endswith(".docx"):
                    import docx
                    docs[rel_name] = "\n".join(
                        p.text for p in docx.Document(path).paragraphs
                    )

            except Exception as e:
                print(f"  Skipping {rel_name}: {e}")

    print(f"  Loaded {len(docs)} documents")
    return docs

def chunk_text(text, max_chars=1200, overlap=150):
    """Simple character-based chunking for retrieval."""
    if not text:
        return []

    chunks = []
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


def embed_texts(texts, model=EMBED_MODEL_NAME, batch_size=64):
    """
    Embed texts locally using all-MiniLM-L6-v2, matching the RIGOR retrieval setup.
    """
    global SENT_MODEL

    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    if SENT_MODEL is None:
        print("  Loading SentenceTransformer on CPU...")
        SENT_MODEL = SentenceTransformer(model, device="cpu")

    vectors = SENT_MODEL.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
        batch_size=batch_size,
        device="cpu",
    )

    return np.asarray(vectors, dtype=np.float32)

def build_faiss_index_from_embeddings(embeddings):
    if embeddings.size == 0:
        return None

    import faiss  # lazy import avoids macOS native-library conflicts

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index

def retrieve_top_k(query, index, texts, k=3):
    if index is None or not texts:
        return []

    q_emb = embed_texts([query]).astype(np.float32)
    _, indices = index.search(q_emb, min(k, len(texts)))
    return [texts[i] for i in indices[0] if 0 <= i < len(texts)]

# =========================================================
# EXTERNAL ONTOLOGY LOADING
# =========================================================

def load_external_ontologies(onto_folder):
    chunks = []
    if not os.path.exists(onto_folder):
        print(f"  Warning: ontology folder not found: {onto_folder}")
        return chunks

    for root, _, files in os.walk(onto_folder):
        for fname in files:
            if not fname.endswith((".owl", ".rdf", ".ttl", ".nt", ".n3")):
                continue

            path = os.path.join(root, fname)

            try:
                g = Graph()
                parsed = False
                last_error = None

                suffix = os.path.splitext(fname)[1].lower()
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
                        print(f"  Parsed {fname} as {fmt}")
                        break
                    except Exception as e:
                        last_error = e

                if not parsed:
                    raise ValueError(f"Could not parse ontology with supported formats. Last error: {last_error}")

                before = len(chunks)

                for cls in g.subjects(RDF.type, OWL.Class):
                    name = str(cls).split("#")[-1].split("/")[-1]
                    lbl = next(g.objects(cls, RDFS.label), None)
                    cmt = next(g.objects(cls, RDFS.comment), None)
                    if name:
                        text = f"[{fname}] Class: {name} IRI: {cls}"
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
                        text = f"[{fname}] Property: {name} IRI: {prop}"
                        if lbl:
                            text += f" — label: {lbl}"
                        if cmt:
                            text += f" — comment: {str(cmt)[:200]}"
                        chunks.append(text)

                print(f"  {fname}: added {len(chunks) - before} chunks")

            except Exception as e:
                print(f"  Skipped {fname}: {e}")

    print(f"  Loaded {len(chunks)} external ontology chunks")
    return chunks

# =========================================================
# PROMPT
# =========================================================

def build_prompt(table_name, columns, fks, schema_str, docs_context, onto_context):
    col_lines = "\n".join(f"  - {col}: {dtype}" for col, dtype in columns.items())
    fk_lines = "\n".join(
        f"  - {fk.get('column')} -> {fk.get('references_table')}.{fk.get('references_column')}"
        for fk in fks
    ) or "  None"

    docs_text = "\n\n".join(docs_context[:3]) if docs_context else "None"
    onto_text = "\n".join(onto_context[:20]) if onto_context else "None"

    return f"""Generate an OWL 2 ontology fragment in Manchester Syntax for the database table '{table_name}'.

[SCHEMA — CURRENT TABLE]
{col_lines}

[FOREIGN KEYS]
{fk_lines}

[FULL DATABASE SCHEMA CONTEXT]
{schema_str}

[RELEVANT DOCUMENTATION]
{docs_text}

[RELEVANT EXTERNAL ONTOLOGY CONCEPTS]
{onto_text}

[INSTRUCTIONS]
- Create an OWL Class for this table
- For every FK column: create an owl:ObjectProperty linking to the referenced class
- For every non-FK column: create a owl:DatatypeProperty with correct xsd: range
- Add rdfs:label and rdfs:comment to every class and property
- Every property MUST have exactly one Domain and one Range
- Do NOT add a Judge-LLM refinement step — output the ontology directly
- Output ONLY valid Manchester Syntax, nothing else

[OUTPUT]"""

# =========================================================
# OUTPUT PARSER — same as baseline
# =========================================================

def parse_manchester_to_rdf(llm_output, base_ns):
    g = Graph()
    g.bind("owl", OWL)
    g.bind("xsd", XSD)
    g.bind("rdfs", RDFS)
    g.bind("", base_ns)

    XSD_MAP = {
        "string": XSD.string,
        "integer": XSD.integer,
        "float": XSD.float,
        "boolean": XSD.boolean,
        "datetime": XSD.dateTime,
        "date": XSD.date,
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

        def find_labels(kw):
            results = []
            for ln in rest:
                if ln.lower().strip().startswith(kw):
                    m = re.search(r'"([^"]+)"(?:@(\w+))?', ln)
                    if m:
                        results.append((m.group(1), m.group(2) or "en"))
            return results

        try:
            low = hdr.lower()
            cur_elem = None

            if low.startswith("class:"):
                name = hdr.split(":", 1)[1].strip()
                uri = base_ns[name]
                g.add((uri, RDF.type, OWL.Class))
                cur_elem = uri

            elif low.startswith("dataproperty:"):
                name = hdr.split(":", 1)[1].strip()
                dom, rng = fv("domain"), fv("range")
                if dom and rng:
                    uri = base_ns[name]
                    g.add((uri, RDF.type, OWL.DatatypeProperty))
                    g.add((uri, RDFS.domain, base_ns[dom]))
                    rng_clean = rng.lower().replace("xsd:", "")
                    g.add((uri, RDFS.range, XSD_MAP.get(rng_clean, XSD.string)))
                    cur_elem = uri

            elif low.startswith("objectproperty:"):
                name = hdr.split(":", 1)[1].strip()
                dom, rng = fv("domain"), fv("range")
                if dom and rng:
                    uri = base_ns[name]
                    g.add((uri, RDF.type, OWL.ObjectProperty))
                    g.add((uri, RDFS.domain, base_ns[dom]))
                    g.add((uri, RDFS.range, base_ns[rng]))
                    cur_elem = uri

            if cur_elem:
                for val, lang in find_labels("rdfs:label"):
                    g.add((cur_elem, RDFS.label, Literal(val, lang=lang)))
                for val, lang in find_labels("rdfs:comment"):
                    g.add((cur_elem, RDFS.comment, Literal(val, lang=lang)))

        except Exception:
            continue

    return g

# =========================================================
# OPENROUTER CHAT CALL
# =========================================================

def call_llm(client, model_name, prompt, retries=3):
    """Call OpenRouter API and return generated text."""
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                temperature=0.2,
                max_tokens=10000,
                extra_headers={
                    "HTTP-Referer": OPENROUTER_SITE_URL,
                    "X-Title": OPENROUTER_APP_NAME,
                },
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

            content = response.choices[0].message.content
            return (content or "").strip()

        except Exception as e:
            last_error = e
            if attempt < retries:
                sleep_seconds = 2 ** (attempt - 1)
                print(f"    Retry {attempt}/{retries} after error: {e}")
                time.sleep(sleep_seconds)
            else:
                print(f"    API call failed: {last_error}")
                return ""


def to_class_name(name):
    return "".join(p[:1].upper() + p[1:] for p in re.split(r"[_\s\-]+", str(name)) if p)


def build_retrieval_query(table_name, table_value):
    columns = get_columns(table_value)
    fks = table_value.get("foreign_keys", []) if isinstance(table_value, dict) else []

    col_names = " ".join(columns.keys())
    fk_targets = " ".join(
        fk.get("references_table", "")
        for fk in fks
        if fk.get("references_table")
    )

    return f"{table_name} {to_class_name(table_name)} {col_names} {fk_targets}".strip()
# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 60)
    print("Non-Iterative Ontology Generation")
    print("=" * 60)

    try:
        client = get_openrouter_client()
    except Exception as e:
        print(f"ERROR: {e}")
        return

    print("\nLoading documents...")
    docs = load_documents(DOCS_PATH)

    doc_chunks = []
    for filename, content in docs.items():
        for chunk in chunk_text(content):
            doc_chunks.append(f"[{filename}]\n{chunk}")

    if doc_chunks:
        print("Building document embedding index...")
        doc_embeddings = embed_texts(doc_chunks)
        doc_index = build_faiss_index_from_embeddings(doc_embeddings)
    else:
        doc_index = None

    print("Loading external ontologies...")
    onto_chunks = load_external_ontologies(EXTERNAL_ONTO_PATH)

    if onto_chunks:
        print("Building ontology embedding index...")
        onto_embeddings = embed_texts(onto_chunks)
        onto_index = build_faiss_index_from_embeddings(onto_embeddings)
    else:
        onto_index = None

    for model_label, model_name in MODELS.items():
        print(f"\nModel: {model_name}")

        for db_name, schema_path in SCHEMAS.items():
            print(f"\n  Schema: {db_name}")

            if not os.path.exists(schema_path):
                print(f"  Not found: {schema_path} — skipping")
                continue

            schema = load_schema(schema_path)
            schema_str = format_schema_str(schema)
            base_ns = Namespace(f"{ONTOLOGY_IRI}#")

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
                fks = table_value.get("foreign_keys", []) if isinstance(table_value, dict) else []

                if not columns:
                    continue

                retrieval_query = build_retrieval_query(table_name, table_value)

                docs_ctx = retrieve_top_k(retrieval_query, doc_index, doc_chunks, k=3) \
                    if doc_index is not None else []

                onto_ctx = retrieve_top_k(retrieval_query, onto_index, onto_chunks, k=3) \
                    if onto_index is not None else []

                prompt = build_prompt(
                    table_name, columns, fks, schema_str, docs_ctx, onto_ctx
                )

                generated = call_llm(client, model_name, prompt).strip()
                if not generated:
                    print("    LLM returned empty response")

                raw_path = os.path.join(raw_dir, f"{table_name}_raw.txt")
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(generated)

                if generated:
                    delta = parse_manchester_to_rdf(generated, base_ns)
                    core_graph += delta

                    n_cls = len(list(delta.subjects(RDF.type, OWL.Class)))
                    n_dp = len(list(delta.subjects(RDF.type, OWL.DatatypeProperty)))
                    n_op = len(list(delta.subjects(RDF.type, OWL.ObjectProperty)))

                    print(f"    Parsed: {n_cls} classes, {n_dp} dataprops, {n_op} objectprops")

            out_path = os.path.join(out_dir, "non_iterative_ontology.owl")
            core_graph.serialize(out_path, format="xml")
            print(f"  Saved: {out_path} ({len(core_graph)} triples)")

    print("\nNon-iterative generation complete.")


if __name__ == "__main__":
    main()