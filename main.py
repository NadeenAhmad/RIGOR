import os
import json
from owlready2 import get_ontology, Thing, ObjectProperty, DataProperty, types
from typing import Dict, List, Generator
from owlready2 import DataProperty, ObjectProperty, Ontology, get_namespace
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import warnings
from tqdm import tqdm  

# Configure Owlready2 to be more tolerant
warnings.filterwarnings("ignore", category=UserWarning, module="owlready2")

def load_schema_from_json(json_path):
    with open(json_path, 'r') as f:
        schema = json.load(f)
    return schema

def create_document_retriever(docs_path, chunk_size=1000, chunk_overlap=200):
    """Process documents and return query-based chunk retriever"""
    try:
        print("Loading documents...")
        loader = DirectoryLoader(docs_path, glob="**/*.docx")
        documents = loader.load()
        
        if not documents:
            raise ValueError("No documents found in the specified path")
        
        print(f"Loaded {len(documents)} documents. Splitting into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)

        if not chunks:
            raise ValueError("No text chunks were generated from documents")
            
        print("Creating embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        print("Building document vector store...")
        try:
            # First try the preferred method
            doc_vector_store = FAISS.from_documents(chunks, embeddings)
        except Exception as e:
            print(f"Using fallback method due to: {str(e)}")
            texts = [doc.page_content for doc in chunks]
            metadatas = [doc.metadata for doc in chunks]
            doc_vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        
        def retrieve_docs(query, k=3):
            """Return relevant document chunks"""
            try:
                docs = doc_vector_store.similarity_search(query, k=k)
                return [doc.page_content for doc in docs]
            except Exception as e:
                print(f"Retrieval error: {str(e)}")
                return ["No relevant documents found"]
        
        return retrieve_docs
    
    except Exception as e:
        print(f"Document processing failed: {str(e)}")
        raise

def safe_get_name(entity):
    """Safely get name of an ontology entity"""
    try:
        return entity.name if hasattr(entity, 'name') else str(entity)
    except:
        return str(entity)

def create_ontology_processor(ontology_folder):
    """Process a folder of OWL ontologies with robust error handling"""
    ontology_chunks = []
    processed_files = 0
    
    print(f"Processing ontologies in {ontology_folder}...")
    for filename in tqdm(os.listdir(ontology_folder)):
        if not filename.endswith(".owl"):
            continue
            
        owl_path = os.path.join(ontology_folder, filename)
        try:
            # Load without strict parameter
            onto = get_ontology(owl_path).load()
            processed_files += 1
            
            # Process classes
            for cls in onto.classes():
                if cls == Thing:
                    continue
                    
                try:
                    cls_name = safe_get_name(cls)
                    
                    # Handle superclasses
                    for super_cls in cls.is_a:
                        if super_cls != Thing:
                            super_name = safe_get_name(super_cls)
                            ontology_chunks.append(
                                f"{cls_name} is_a {super_name}: {cls_name} is a subclass of {super_name}"
                            )
                    
                    # Handle properties
                    for prop in onto.object_properties():
                        try:
                            prop_name = safe_get_name(prop)
                            
                            # Handle domain and range
                            if hasattr(prop, 'domain') and cls in prop.domain:
                                for range_entity in prop.range:
                                    range_name = safe_get_name(range_entity)
                                    ontology_chunks.append(
                                        f"{cls_name} {prop_name} {range_name}: {cls_name} relates to {range_name} via {prop_name}"
                                    )
                        except Exception as prop_e:
                            continue
                except Exception as prop_e:
                            continue
                            
            for prop in onto.object_properties():
                try:
                    prop_name = safe_get_name(prop)
                    
                    for domain in getattr(prop, 'domain', []):
                        for range_ in getattr(prop, 'range', []):
                            domain_name = safe_get_name(domain)
                            range_name = safe_get_name(range_)
                            if domain_name and range_name:
                                ontology_chunks.append(
                                    f"{domain_name} {prop_name} {range_name}: {domain_name} can relate to {range_name} via {prop_name}"
                                )
                except Exception as prop_e:
                    continue
                    
        except Exception as e:
            print(f"\nSkipped {filename}: {str(e)}")
            continue
    
    if not ontology_chunks:
        raise ValueError(f"No valid ontology data processed from {processed_files} files")
    
    print("Creating ontology embeddings...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        # Create index in batches if needed
        batch_size = 1000
        if len(ontology_chunks) > batch_size:
            print(f"Processing {len(ontology_chunks)} chunks in batches...")
            from itertools import islice
            vector_store = None
            
            for i in range(0, len(ontology_chunks), batch_size):
                batch = ontology_chunks[i:i + batch_size]
                if vector_store is None:
                    vector_store = FAISS.from_texts(batch, embeddings)
                else:
                    vector_store.add_texts(batch)
        else:
            vector_store = FAISS.from_texts(ontology_chunks, embeddings)
    except Exception as e:
        print(f"Failed to create ontology index: {str(e)}")
        raise
    
    def retrieve_ontology(query, k=2):
        try:
            docs = vector_store.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            print(f"Ontology retrieval error: {str(e)}")
            return ["No relevant ontology concepts found"]
    
    return retrieve_ontology


def traverse_database_schema(schema: Dict, root_table: str = None) -> Generator[Dict, None, List[str]]:
    """
    Iteratively traverse database schema using foreign keys when available,
    and continue to unconnected tables when necessary.
    Yields schema context at each step and returns final traversal order.
    """
    processed = set()
    queue = []
    traversal_order = []
    full_schema = schema.copy()

    # Start from root table or pick the first table arbitrarily
    if not root_table:
        root_table = next(iter(full_schema.keys()))
    queue.append(root_table)

    while queue or len(processed) < len(full_schema):
        if not queue:
            # All reachable tables processed, fallback to an unvisited table
            unvisited = set(full_schema.keys()) - processed
            if unvisited:
                queue.append(next(iter(unvisited)))

        current_table = queue.pop(0)
        print("Current Table:", current_table)
        if current_table in processed:
            continue

        processed.add(current_table)
        traversal_order.append(current_table)

        # Create reduced schema snapshot for current step
        reduced_schema = {
            table: details
            for table, details in full_schema.items()
            if table not in processed
        }

        schema_context = {
            "current_table": current_table,
            "schema_snapshot": reduced_schema,
            "foreign_keys": full_schema[current_table].get("foreign_keys", []),
            "processed_tables": list(processed)
        }

        yield schema_context

        # Enqueue referenced tables via foreign keys
        for fk in full_schema[current_table].get("foreign_keys", []):
            ref_table = fk.get("references_table")
            if ref_table not in processed and ref_table not in queue:
                queue.append(ref_table)

    return traversal_order
    
def build_core_ontology(
    schema_context: Dict, 
    existing_ontology: Ontology = None, 
    llm_generated: Ontology = None
) -> Ontology:
    """Build and merge ontology incrementally based on database schema traversal.
    Args:
        schema_context: Database schema information.
        existing_ontology: Existing ontology to merge with (optional).
        llm_generated: LLM-generated ontology to incorporate (optional).
    Returns:
        Merged ontology combining schema, existing, and LLM-generated components.
    """
    # Initialize core ontology (fresh or from existing)
    core_onto = get_ontology("http://core.onto/") if existing_ontology is None else existing_ontology

    with core_onto:
        # Process current table from schema
        current_table = schema_context["current_table"]
        table_details = schema_context["schema_snapshot"].get(current_table, {})

        # Create or retrieve class for the current table
        table_class = getattr(core_onto, current_table) if hasattr(core_onto, current_table) \
            else types.new_class(current_table, (Thing,))

        # Add data properties (non-foreign key columns)
        for column in table_details.get("columns", []):
            if not any(fk["column"] == column for fk in table_details.get("foreign_keys", [])):
                prop_name = f"has_{column}"
                if not hasattr(core_onto, prop_name):
                    prop_cls = types.new_class(prop_name, (DataProperty,))
                    prop_cls.domain = [table_class]
                    prop_cls.range = [str]

        # Add object properties (foreign keys)
        for fk in table_details.get("foreign_keys", []):
            ref_table = fk["references_table"]
            prop_name = f"relates_to_{ref_table}"

            ref_class = getattr(core_onto, ref_table) if hasattr(core_onto, ref_table) \
                else types.new_class(ref_table, (Thing,))

            if not hasattr(core_onto, prop_name):
                prop_cls = types.new_class(prop_name, (ObjectProperty,))
                prop_cls.domain = [table_class]
                prop_cls.range = [ref_class]

        # Merge with existing ontology if provided
        if existing_ontology:
            core_onto.imported_ontologies.append(existing_ontology)

        # Incorporate LLM-generated ontology if provided
        if llm_generated:
            core_onto.imported_ontologies.append(llm_generated)
            # Alternatively: deep merge classes/properties if needed
            # merge_ontologies(core_onto, llm_generated)

    return core_onto

def format_schema_prompt(schema_context: Dict, full_schema: Dict) -> str:
    """Format schema context into LLM prompt including full current table schema and its foreign keys"""
    current_table = schema_context["current_table"]
    current_columns = full_schema.get(current_table, {}).get("columns", [])
    current_schema_str = f"{current_table}: Columns {', '.join(current_columns)}"

    available_tables = "\n".join([
        f"Table {name}: Columns {', '.join(details.get('columns', []))}"
        for name, details in schema_context["schema_snapshot"].items()
    ]) or "None"

    # Safely build foreign key info
    fks = "\n".join([
        f"{current_table}.{fk.get('column', 'UNKNOWN')} -> {fk.get('references_table', 'UNKNOWN')}.{fk.get('references_column', 'UNKNOWN')}"
        for fk in schema_context.get("foreign_keys", [])
    ]) or "None"

    return f"""Database Schema Context:
Current Focus Table: {current_table}
Processed Tables: {', '.join(schema_context['processed_tables'])}

Current Table Schema:
{current_schema_str}

Foreign Key Relationships (from {current_table}):
{fks}

Available Tables in Snapshot:
{available_tables}
"""