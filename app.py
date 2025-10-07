from main import *
import datetime


login("Huggingface_API_code")

def prompt_llm(schema: Dict, document_retriever, ontology_processor, llm):
    """Iterative RAG process for ontology development"""
    # Initialize core ontology
    core_ontology = Graph()
    core_ontology.parse("path_to_core_ontology", format="xml")

    core_ontology.bind("owl", OWL)
    core_ontology.bind("xsd", XSD)
    core_ontology.bind("", Namespace("http://example.org/ontology#"))
    
    # Traverse schema with foreign-key guided traversal
    traversal_gen = traverse_database_schema(schema)
    
    for schema_context in traversal_gen:
        current_table = schema_context["current_table"]
        
        # 1. RAG Retrieval using table name as query
        relevant_docs = document_retriever(current_table)
        existing_ontology_knowledge = ontology_processor(current_table)
        
        # 2. Prepare LLM input
        schema_prompt = format_schema_prompt(schema_context, schema)
        llm_input = {
            "table_name": current_table,
            "schema_context": schema_prompt,
            "documents": "\n".join(relevant_docs),
            "existing_ontology": "\n".join(existing_ontology_knowledge)
        }
        temp_graph = Graph()
        temp_graph.namespace_manager = core_ontology.namespace_manager
        
        # 3. Generate new ontology elements with LLM
        llm_response = llm.generate(llm_input)
        
        # 4. Build/update core ontology
        temp_graph = parse_llm_ontology(
            llm_output=llm_response,
            core_graph=core_ontology
        )

        core_ontology += temp_graph

        # 5. Track progress
        print(f"Processed {current_table}")
        print(f"Remaining tables: {len(schema_context['schema_snapshot'])}")
        print(f"Core_Ontology size: {len(list(core_ontology.subjects(RDF.type, OWL.Class)))} classes\n")
        print(f"New_Ontology size: {len(list(temp_graph.subjects(RDF.type, OWL.Class)))} classes\n")
        
    
    return core_ontology

def clean_llm_output(llm_output: str) -> str:
    cleaned_lines = []
    for line in llm_output.split('\n'):
        line = line.strip()
        if not line:
            continue
        if any(line.lower().startswith(x) for x in ["class:", "dataproperty:", "objectproperty:", "annotations:"]):
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def parse_llm_ontology(llm_output: str, core_graph: Graph) -> Graph:
    """Convert LLM output to RDFLib graph and merge with core ontology"""
    temp_graph = Graph()
    temp_graph.namespace_manager = core_graph.namespace_manager  
    
    # Define common namespaces
    owl = Namespace("http://www.w3.org/2002/07/owl#")
    xsd = Namespace("http://www.w3.org/2001/XMLSchema#")
    base = Namespace("http://example.org/ontology#")
    prov = Namespace("http://www.w3.org/ns/prov#")
    
    # Extended XSD type mapping
    XSD_TYPES = {
        'string': XSD.string,
        'integer': XSD.integer,
        'float': XSD.float,
        'boolean': XSD.boolean,
        'datetime': XSD.dateTime,
        'date': XSD.date,
        'time': XSD.time
    }

    current_element = None 

    for line in llm_output.split('\n'):
        line = line.strip()
        if not line or line.startswith(('[', '###')):
            continue
            
        try:
            # Class declaration
            if line.lower().startswith('class:'):
                class_name = line.split(':', 1)[1].strip()
                class_uri = base[class_name]
                temp_graph.add((class_uri, RDF.type, OWL.Class))
                current_element = class_uri
                
            # DataProperty declaration
            elif line.lower().startswith('dataproperty:'):
                parts = re.split(r'\s+', line[len('dataproperty:'):].strip())
                prop_name = parts[0]
                domain_class = parts[parts.index('domain')+1]
                range_type = parts[parts.index('range')+1].lower()
                
                prop_uri = base[prop_name]
                domain_uri = base[domain_class]
                
                # Add property declaration
                temp_graph.add((prop_uri, RDF.type, OWL.DatatypeProperty))
                temp_graph.add((prop_uri, RDFS.domain, domain_uri))
                temp_graph.add((prop_uri, RDFS.range, XSD_TYPES.get(range_type, XSD.string)))
                current_element = prop_uri
                
            # ObjectProperty declaration
            elif line.lower().startswith('objectproperty:'):
                parts = re.split(r'\s+', line[len('objectproperty:'):].strip())
                prop_name = parts[0]
                domain_class = parts[parts.index('domain')+1]
                range_class = parts[parts.index('range')+1]
                
                prop_uri = base[prop_name]
                domain_uri = base[domain_class]
                range_uri = base[range_class]
                
                # Add property declaration
                temp_graph.add((prop_uri, RDF.type, OWL.ObjectProperty))
                temp_graph.add((prop_uri, RDFS.domain, domain_uri))
                temp_graph.add((prop_uri, RDFS.range, range_uri))
                current_element = prop_uri
                
            # Provenance handling
            elif 'prov:wasderivedfrom' in line.lower():
                if current_element:
                    prov_uri = re.search(r'<(.*?)>', line)
                    if prov_uri:
                        temp_graph.add((current_element, prov.wasDerivedFrom, URIRef(prov_uri.group(1))))
                    else:
                        print("NO VALID PROVENANCE URI FOUND")
                else:
                    print("NO CURRENT ELEMENT FOR PROVENANCE")
                
        except Exception as e:
            print(f"Skipped line: {line} | Error: {str(e)}")
            continue
    
    # Merge the temporary graph into core ontology
    core_graph += temp_graph
    
    print(f"Merged ontology stats:")
    print(f" - Classes: {len(list(core_graph.subjects(RDF.type, OWL.Class)))}")
    print(f" - DataProperties: {len(list(core_graph.subjects(RDF.type, OWL.DatatypeProperty)))}")
    print(f" - ObjectProperties: {len(list(core_graph.subjects(RDF.type, OWL.ObjectProperty)))}\n")
    
    return core_graph

# 1. Initialize Hugging Face model
def setup_huggingface_model(model_name="meta-llama/Llama-3.1-8B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        max_new_tokens=1000, 
        do_sample=True,
        temperature=0.7
    )
    
    return HuggingFacePipeline(pipeline=hf_pipeline)

# 2. Custom LLM wrapper for ontology generation
class OntologyLLM:
    def __init__(self, hf_pipeline):
        self.pipeline = hf_pipeline
    
    def generate(self, input_data: Dict) -> str:
        prompt = self._format_ontology_prompt(input_data)
        response = self.pipeline(prompt)
        response = response.replace(prompt, "").strip()
        response = clean_llm_output(response)
        return response
    
    def _format_ontology_prompt(self, data: Dict) -> str:
        return f"""Generate ontology elements with provenance annotations for database table '{data['table_name']}' based on:

[CONTEXT]
### Database Schema
{data['schema_context']}

### Relevant Documents
{data['documents']}

### Existing Ontology Knowledge
{data['existing_ontology']}

[STRICT INSTRUCTIONS]
1. TYPE CONSISTENCY: A URI must not be declared as more than one of the owl:Class, owl:ObjectProperty and owl:DatatypeProperty 
2. PROPERTY EXCLUSIVITY: Each property URI MUST be either owl:ObjectProperty or owl:DatatypeProperty and a property URI MUST NOT be simultaneously declared as both.
3. NO TYPE CONFLICTS: Each URI MUST denote a unique entity i.e reuse of the same URI across distinct categories (Class vs. Property vs. Individual) is STRICTLY forbidden.
4. STANDARD DATATYPES: All owl:DatatypeProperty ranges MUST be restricted to XML Schema datatypes (xsd:string, xsd:integer, xsd:decimal, etc.) 
5. UNION FOR MULTIPLE DOMAIN/RANGE: If a property applies to multiple domains or multiple ranges, these MUST be expressed using a single axiom with owl:unionOf.
6. DOMAIN AND RANGE REQUIREMENT: Every owl:ObjectProperty and owl:DatatypeProperty MUST explicitly declare both: at least one rdfs:domain at least one rdfs:range.
All the above mentioned rules are important to apply
[EXAMPLE FORMAT]
----------------------------------------
Class: {data['table_name']} 
Annotations: prov:wasDerivedFrom <http://example.org/provenance/{data['table_name']}>

DataProperty: has_column_name
  domain {data['table_name']}
  range xsd:string
Annotations: prov:wasDerivedFrom <http://example.org/provenance/{data['table_name']}/column_name>

ObjectProperty: relates_to_table
  domain {data['table_name']}
  range RelatedTable
Annotations: prov:wasDerivedFrom <http://example.org/provenance/{data['table_name']}/fk_column>
----------------------------------------

Only output Manchester Syntax and nothing else and follow the instructions strictly.

[OUTPUT]
"""
    
# 3. Full execution pipeline
def run_full_pipeline(schema_path: str, docs_path: str, ontology_path: str):
    # Load components
    schema = load_schema_from_json(schema_path)
    doc_retriever = create_document_retriever(docs_path)
    ontology_proc = create_ontology_processor(ontology_path)
    
    # Initialize models
    hf_pipe = setup_huggingface_model()
    llm = OntologyLLM(hf_pipe)
    
    # Run iterative process
    final_ontology = prompt_llm(
        schema=schema,
        document_retriever=doc_retriever,
        ontology_processor=ontology_proc,
        llm=llm
    )
    
    # Save results
    final_ontology.serialize("llama_icu.owl", format="xml")
    print("Ontology generation complete!")

# 4. Example execution
if __name__ == "__main__":
    run_full_pipeline(
        schema_path="path_to_schema",
        docs_path="path_to_documents",
        ontology_path="path_to_ontology"
    ) 