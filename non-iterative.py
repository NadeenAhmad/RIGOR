import os
import faiss
import chardet
import json
import torch
from relbench.datasets import get_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login

# Login to Hugging Face using access token
login("Your_HuggingFace_API_Access_Token")

# Load SQL schema from a JSON file
def load_schema_from_json(json_path):
    with open(json_path, 'r') as f:
        schema = json.load(f)
    return schema

# Read documentation files (.txt or .docx) from a folder with correct encoding
def read_documentation(doc_folder):
    docs = {}
    for filename in os.listdir(doc_folder):
        if filename.endswith(".txt") or filename.endswith(".docx"):
            path = os.path.join(doc_folder, filename)
            with open(path, 'rb') as f:
                raw_data = f.read()
            detected = chardet.detect(raw_data)
            encoding = detected.get('encoding', 'utf-8')
            with open(path, 'r', encoding=encoding, errors='replace') as f:
                docs[filename] = f.read()
    return docs

# Build FAISS index from documentation using sentence embeddings
def build_doc_index(docs, embedding_model):
    texts = list(docs.values())
    filenames = list(docs.keys())
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings, filenames, texts

# Retrieve top-k most similar documents given a query (e.g., table name)
def retrieve_docs(query, embedding_model, index, filenames, texts, top_k=3):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for idx in indices[0]:
        results.append((filenames[idx], texts[idx]))
    return results

# Generate RDF triples from table schema and documentation context using a language model
def generate_triples(table_name, table_schema, llm, save_dir, core_ontology, gold_ontology, docs_context):
    prompt = f"Given the following SQL table schema for table '{table_name}':\n"
    for col, dtype in table_schema.items():
        prompt += f"- {col}: {dtype}\n"
    prompt += "\nDocumentation context:\n"
    for filename, in docs_context:
        prompt += f"File: {filename}\n{docs_context}\n"
    prompt += "\nCore Ontology (in Turtle format):\n"
    prompt += core_ontology
    prompt += "\nGold Ontology (in OWL format):\n"
    prompt += gold_ontology
    prompt += ("\n\nBased on the above core ontology, gold standard ontology from domain, generate RDF triples in Turtle format mapping the SQL table and its columns "
               "to the provided ontology. Only output valid RDF Triples in Turtle format and nothing else.")

    print(len(prompt))  # Optional: print prompt length for debugging

    # Generate output using LLM pipeline
    result = llm(prompt, max_new_tokens=2048, do_sample=True, temperature=0.7)
    output_text = result[0]['generated_text']
    generated_part = output_text[len(prompt):].strip()

    # Save generated triples to file
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{table_name}_onto.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(generated_part)
    
    return generated_part

# Main function: loads data, builds index, runs triple generation loop
def main():
    # Load SQL schema
    path_json = "path_to_table_schema_in_json"
    schema = load_schema_from_json(path_json)
    print("Extracted SQL schema:", schema)

    # Read documentation files
    doc_folder = "path_to_document_folder"
    docs = read_documentation(doc_folder)
    print(f"Read {len(docs)} documentation files.")

    # Load ontologies
    with open("path_to_core_ontology", "r") as f:
        core_ontology = f.read()
    with open("path_to_gold_ontology", "r") as g:
        gold_ontology = g.read()

    # Load embedding model and build FAISS index
    embedding_model = SentenceTransformer('all-MiniLM-L6-V2')
    index, embeddings, filenames, texts = build_doc_index(docs, embedding_model)

    # Load LLM model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",  
        trust_remote_code=True,
        device_map="auto",
        token=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        trust_remote_code=True,
        token=True
    )

    model.config.sliding_window = None

    # Create text-generation pipeline
    llm = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    processed_tables = set()

    # Iterate through each table in schema and generate RDF triples
    for table_name, table_schema in schema.items():
        if table_name in processed_tables:
            continue  # Skip if already processed

        # Retrieve top relevant documentation context
        docs_context = retrieve_docs(table_name, embedding_model, index, filenames, texts, top_k=3)
        print("The docs context are:", docs_context)

        # Directory to save output
        save_dir = "./non-iterative_llama"

        # Generate triples and print output
        output = generate_triples(table_name, table_schema, llm, save_dir, core_ontology, gold_ontology, docs_context)
        print(f"\nGenerated triples for table '{table_name}':\n{output}\n")

# Run main if this script is executed
if __name__ == "__main__":
    main()
