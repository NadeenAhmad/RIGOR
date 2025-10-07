

!pip install sentence-transformers scikit-learn matplotlib seaborn

!pip install rdflib

from rdflib import Graph, OWL, RDFS
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- Step 1: Load Ontology Classes ---
def extract_classes_from_owl(owl_file_path):
    g = Graph()
    g.parse(owl_file_path, format="xml")
    classes = set()
    for s, p, o in g:
        if o in [OWL.Class, RDFS.Class]:
            classes.add(str(s).split("#")[-1].strip())
    return sorted(classes)

# --- Step 2: Load Column Names from First Table ---
def extract_columns_of_first_table(json_file_path):
    with open(json_file_path, "r") as f:
        schema = json.load(f)
    first_table = sorted(schema.keys())[0]
    column_names = list(schema[first_table].get("columns", {}).keys())
    return column_names, first_table

# --- Step 3: Semantic Similarity Evaluation (Class → Columns) ---
def compute_similarity_class_to_columns(ontology_classes, column_names, threshold=0.65, top_n=3):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    class_embeddings = model.encode(ontology_classes)
    column_embeddings = model.encode(column_names)

    sim_matrix = cosine_similarity(class_embeddings, column_embeddings)
    sim_df = pd.DataFrame(sim_matrix, index=ontology_classes, columns=column_names)

    # Extract only top N similar columns per class
    reduced_matches = {
        cls: sim_df.loc[cls].nlargest(top_n)
        for cls in ontology_classes
    }
    reduced_df = pd.concat(reduced_matches, axis=1).T

    # Coverage stats
    max_sim_per_class = sim_matrix.max(axis=1)
    covered_classes = np.sum(max_sim_per_class >= threshold)
    coverage_rate = 100 * covered_classes / len(ontology_classes)

    # Debug output
    print("\n--- Class-to-Column Similarity ---")
    for cls, sim in zip(ontology_classes, max_sim_per_class):
        mark = "matched" if sim >= threshold else "unmatched"
        #print(f"{cls:30} | Max Similarity: {sim:.2f} {mark}")

    return reduced_df, coverage_rate, covered_classes

# --- Step 4: Heatmap Plot ---
# --- Step 4: Heatmap Plot ---
def plot_heatmap(df, table_name, threshold, output_path="heatmap.jpg"):
    plt.figure(figsize=(12, len(df) * 0.4))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title(f"Top Column Matches for Ontology Classes (Threshold = {threshold})")
    plt.xlabel(f"Top Matching Columns from '{table_name}'")
    plt.ylabel("Ontology Classes")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save heatmap as JPG
    plt.savefig(output_path, format="jpg", dpi=300)
    plt.show()

# --- Step 5: Run the Pipeline ---
owl_file = "chemotherapy.owl"         # <-- update this
json_file = "schema_rd.json"          # <-- update this
similarity_threshold = 0.55
top_n_matches = 3

# Load and run
ontology_classes = extract_classes_from_owl(owl_file)
column_names, table_name = extract_columns_of_first_table(json_file)

reduced_matrix, coverage, matched_classes = compute_similarity_class_to_columns(
    ontology_classes, column_names, threshold=similarity_threshold, top_n=top_n_matches
)

# Report results
print(f"\n Class Coverage Rate by Columns in '{table_name}': {coverage:.2f}%")
print(f" {matched_classes} out of {len(ontology_classes)} classes are semantically represented by table columns")

# Plot heatmap

# Plot and save heatmap
plot_heatmap(reduced_matrix, table_name, similarity_threshold, output_path="class_column_similarity.jpg")

