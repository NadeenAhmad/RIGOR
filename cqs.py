import os
from huggingface_hub import login
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

login("Your_HuggingFace API Access Token")
def load_schema_from_json(json_path):
    with open(json_path, 'r') as f:
        schema = json.load(f)
    return schema

def generate_cqs(table_name, table_schema, llm, save_dir="Directory_to_save_the_genreated_CQs"):
    prompt = f"Given the SQL table schema for table '{table_name}':\n"
    for col, dtype in table_schema.items():
        prompt += f"- {col}: {dtype}\n"

    prompt += (
        f"\n\nGenerate 5 competency questions (CQs) that this table's ontology should answer. "
        "For each question, also provide:\n"
        "- A short answer explaining how the ontology would answer it using the table schema.\n"
        "Think step-by-step based on the data and relationships.\n\n"
        f"### Competency Questions for Table '{table_name}'\n"
        "**1. [Question]**\n"
        "- **Answer**: [Explanation]\n"
        "Continue similarly for 5 questions.\n"
        "Do not include any additional commentary or explanation outside the format."
    )


    # Generate result using the language model
    result = llm(prompt, max_new_tokens=512, temperature=0.5)
    cq_doc = result[0]['generated_text']

    # Step 2: Save to file
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{table_name}_cqs.txt")
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(cq_doc)
    
    return cq_doc


def main():
    path_json="path_to_schema_in_json"
    schema = load_schema_from_json(path_json)

    model = AutoModelForCausalLM.from_pretrained(
        "Model_name",  #mistralai/Mistral-Small-24B-Instruct-2501 was used to generate the CQs
        trust_remote_code=True,
        device_map="auto",
        token=True,
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Model_name", #mistralai/Mistral-Small-24B-Instruct-2501 was used to generate the CQs
        trust_remote_code=True,
        token=True,
        torch_dtype=torch.float16
    )

    llm = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    processed_tables = set()

    for table_name, table_schema in schema.items():
        if table_name in processed_tables:
            continue

        output = generate_cqs(table_name, table_schema,llm)

if __name__ == "__main__":
    main()