import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
login("Your_HuggingFace_API_Access_Token")

def load_schema_from_json(json_path):
    with open(json_path, 'r') as f:
        schema = json.load(f)
    return schema

def generate_triples(table_name, table_schema, llm, save_dir, core_ontology, gold_ontology, docs_context):
    prompt = f"Given the following SQL table schema for table '{table_name}':\n"
    for col, dtype in table_schema.items():
        prompt += f"- {col}: {dtype}\n"

    prompt += ("\n\nBased on the above table schema, generate RDF triples in Turtle format mapping the SQL table and its columns"
               "to the provided ontology. Only output valid RDF Triples in Turtle format and nothing else.")

    result = llm(prompt, max_new_tokens=2048, do_sample=True, temperature=0.7)
    output_text = result[0]['generated_text']
    onto_doc = output_text
    generated_part = output_text[len(prompt):].strip()
    
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{table_name}_onto.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(generated_part)
    
    return generated_part


def main():
    path_json="path_to_table_schema_in_json"
    schema = load_schema_from_json(path_json)
    print("Extracted SQL schema:", schema)

    model = AutoModelForCausalLM.from_pretrained(
        "Model_name", #meta-llama/Llama-3.1-8B-Instruct was used in our research
        trust_remote_code=True,
        device_map="auto",
        token=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Model_name", #meta-llama/Llama-3.1-8B-Instruct was used in our research
        trust_remote_code=True,
        token=True
    )

    model.config.sliding_window = None
    llm = pipeline("text-generation",model=model,tokenizer=tokenizer, device_map="auto")
    processed_tables = set()

    for table_name, table_schema in schema.items():
        if table_name in processed_tables:
            continue

        save_dir = "./generated_triples_baseline"
        output = generate_triples(table_name, table_schema, llm, save_dir)

        print(f"\nGenerated triples for table '{table_name}':\n{output}\n")


if __name__ == "__main__":
    main()