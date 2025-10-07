# RIGOR FRAMEWORK


This repository presents a structured pipeline for mapping clinical SQL schemas to domain ontologies using LLMs using RAG in a iterative manner. The pipeline consists of several core modules and evaluation strategies.

---

## 1. Baseline

The **baseline** method involves direct RDF triple generation from SQL table schemas without leveraging any external documentation or iterative refinement. This serves as a control to assess the impact of incorporating additional context and reasoning steps in later stages.

## Example Usage

```bash
python baseline.py 
```

## 2. Non-Iterative Approach

The **non-iterative pipeline** enhances the baseline by using table-level documentation and domain ontologies as context for a single-shot RDF triple generation. It uses an embedding model (e.g., `all-MiniLM-L6-V2`) to retrieve relevant documentation, which is then passed with the schema and ontology to an LLM (e.g., LLaMA-3.1-8B-Instruct) for triple generation.
## Example Usage

```bash
python non-iterative.py 
```

## 4. RIGOR Framework

**RIGOR (Reasoning-based Iterative Generation for Ontological Representation)** is an extension of the non-iterative pipeline. It introduces an iterative loop and predefined patterns guide the refinement of RDF outputs until satisfactory alignment is achieved.

```bash
python app.py 
```

## 3. Competency Questions (CQs) Generation

This module automatically generates **Competency Questions (CQs)** for each SQL table based on its structure and semantic context. These questions help validate the alignment between the generated ontology and the intended schema semantics, and serve as a bridge for expert review or formal evaluation.
```bash
python cqs.py 
```

## 5. LLM-As-Judge

The **LLM-As-Judge** module uses a separate LLM (distinct from the generator) to evaluate the quality of generated RDF triples.
This enables a low-cost, scalable alternative to human evaluation.
```bash
python eval.py 
```

## 6. Semantic Evaluation

The **Semantic Evaluation** of the ontology refers to the semantic covergae of the ontology classes to 
the columns of the table schema.
```bash
python semantic_evaluation.py 
```
## 7. Ablation Study

The **Ablation Study** provides the results of the ontologies with removing each component of RAG and we calculate Precision, Recall and F1 score. 