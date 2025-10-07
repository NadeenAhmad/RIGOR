import os
import json
import re
from transformers import pipeline
from rdflib import Graph, URIRef, RDF, OWL, RDFS
import textwrap
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
login("Hugging_Face_Token")
# Configuration
COMPETENCY_QUESTIONS_DIR = "path_to_cqs_directory"
ONTOLOGY_FILE = "path_to_generated_ontology"
SCHEMA_FILE = "path_to_schema_file"
OUTPUT_FILE = "mistral_icu.json"
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"  # or any other model
MAX_CHUNK_SIZE = 2048  # characters for ontology chunks

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL,
    device_map="auto",
    torch_dtype=torch.float16
)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    temperature=0.1,
)

def load_competency_questions():
    questions = []
    for file in os.listdir(COMPETENCY_QUESTIONS_DIR):
        if file.endswith(".txt"):
            with open(os.path.join(COMPETENCY_QUESTIONS_DIR, file), 'r') as f:
                content = f.read().split("\n\n")
                questions.append({
                    "question": content[0].strip(),
                    "answer": content[1].strip() if len(content) > 1 else ""
                })
    return questions

def split_ontology(ontology_text):
    """Split ontology into manageable chunks"""
    chunks = textwrap.wrap(ontology_text, width=MAX_CHUNK_SIZE, 
                          break_long_words=False, replace_whitespace=False)
    return chunks
def load_ontology_chunks():
    g = Graph()
    g.parse(ONTOLOGY_FILE, format="xml")  # Assuming OWL/XML format
    
    chunks = []
    
    # Process classes
    for cls in g.subjects(predicate=RDF.type, object=OWL.Class):
        class_name = g.qname(cls)
        chunk = f"Class: {class_name}\n"
        properties = []
        
        # Get properties where this class is in the domain
        for prop in g.subjects(predicate=RDFS.domain, object=cls):
            prop_name = g.qname(prop)
            prop_info = f"- {prop_name}: "
            
            # Get range
            ranges = []
            for rng in g.objects(prop, RDFS.range):
                if isinstance(rng, URIRef):
                    ranges.append(g.qname(rng))
            if ranges:
                prop_info += f"Range={', '.join(ranges)}"
            
            properties.append(prop_info)
        
        if properties:
            chunk += "Properties:\n" + "\n".join(properties)
            chunks.append(chunk)
    
    # Process object properties
    for prop in g.subjects(predicate=RDF.type, object=OWL.ObjectProperty):
        prop_name = g.qname(prop)
        chunk = f"Object Property: {prop_name}\n"
        info = []
        
        # Get domains
        domains = []
        for dom in g.objects(prop, RDFS.domain):
            if isinstance(dom, URIRef):
                domains.append(g.qname(dom))
        if domains:
            info.append(f"Domain: {', '.join(domains)}")
        
        # Get ranges
        ranges = []
        for rng in g.objects(prop, RDFS.range):
            if isinstance(rng, URIRef):
                ranges.append(g.qname(rng))
        if ranges:
            info.append(f"Range: {', '.join(ranges)}")
        
        if info:
            chunk += "\n".join(info)
            chunks.append(chunk)
    
    return chunks


def load_schema():
    with open(SCHEMA_FILE, 'r') as f:
        return json.load(f)

def evaluate_metric(metric, context):
    # Llama-3.1 specific prompt template
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an ontology evaluation expert. Analyze this ontology fragment and provide a numerical score (0-5) for {metric}.
Criteria: {get_metric_criteria(metric)}
Return ONLY a numerical score between 0.0-5.0 with one decimal place<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Competency Questions:
{context['questions']}

Ontology Fragment:
{context['ontology_chunk']}

Database Schema:
{context['schema']}

Format your response as: "Score: X.X"<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
Score: """

    response = pipe(
        prompt,
        max_new_tokens=10,
        return_full_text=False,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )[0]['generated_text']

    # Debugging output
    print(f"Raw response for {metric}: {response}")
    
    # Enhanced parsing with multiple fallbacks
    try:
        # Try multiple possible patterns
        patterns = [
            r"Score:\s*([0-5]\.\d)",
            r"\b([0-5]\.\d)\b",
            r"([0-5]\.[0-9])/5",
            r"final score:\s*([0-5]\.\d)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                return max(0.0, min(5.0, round(score, 2)))
        
        print(f"No valid score found in: {response}")
        return 0.0
    except Exception as e:
        print(f"Parse error: {str(e)} in response: {response}")
        return 0.0

def get_metric_criteria(metric):
    criteria = {
        "accuracy": "1. Correctness of domain representation\n2. Alignment with competency questions",
        "completeness": "1. Coverage of required concepts\n2. Presence of necessary relationships",
        "conciseness": "1. Absence of redundancy\n2. Minimal complexity",
        "adaptability": "1. Extensibility\n2. Modular design\n3. Clear naming",
        "clarity": "1. Readability\n2. Unambiguous definitions\n3. Documentation quality",
        "consistency": "1. Logical coherence\n2. Valid relationships\n3. Proper inheritance"
    }
    return criteria.get(metric, "")

def main():
    questions = load_competency_questions()
    ontology_chunks = load_ontology_chunks()
    schema = json.dumps(load_schema())

    metrics = ["accuracy", "completeness", "conciseness", 
               "adaptability", "clarity", "consistency"]
    
    results = {metric: [] for metric in metrics}

    # Evaluate each chunk
    for chunk in ontology_chunks:
        context = {
            "questions": questions,
            "ontology_chunk": chunk,
            "schema": schema
        }
        
        for metric in metrics:
            score = evaluate_metric(metric, context)
            results[metric].append(score)

    # Calculate average scores
    final_scores = {metric: round(sum(scores)/len(scores), 2) 
                   for metric, scores in results.items()}

    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump({
            "chunk_scores": results,
            "final_scores": final_scores
        }, f, indent=2)

    print(f"Evaluation completed. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()