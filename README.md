# HARP-ML Supplementary Material

This repository contains the replication package for the paper:

> **HARP-ML: A Hybrid Assisted Refactoring Pipeline for ML Code Smells**
> Gargie Tambe and Y. Raghu Reddy
> Software Engineering Research Center (SERC)
> International Institute of Information Technology, Hyderabad, India

## Overview

HARP-ML is a hybrid refactoring framework that addresses ML-specific code
smells by classifying them into two categories:

- **Type I smells** — syntactically localized anti-patterns resolved via
  Concrete Syntax Tree (CST) transformations validated through
  Property-Based Testing (PBT).
- **Type II smells** — context-dependent smells requiring semantic reasoning,
  resolved via a Small Language Model (SLM) operating on a focused context
  window.

This repository contains the supplementary material supporting the paper's
claims, including the smell taxonomy classification, prompt templates,
knowledge base entries, PBT invariants, and the CST-based refactoring engine.

## Repository Structure

```
.
├── evaluation-repos
│   ├── niche_repositories.csv
│   └── README.md
├── knowledge-base
│   ├── gradients_not_cleared.json
│   ├── merge_api_parameter.json
│   └── README.md
├── pbt
│   ├── README.md
│   ├── test_chain_indexing.py
│   └── test_dataframe_conversion.py
├── prompts
│   ├── gradients_not_cleared.txt
│   └── merge_api_parameter.txt
├── README.md
├── refactoring
│   ├── cst-refactorer.py
│   └── README.md
└── taxanomy
    ├── README.md
    └── smell_classification.csv
```

## What Each Directory Contains

### `taxonomy/`

The full classification of all 22 ML-specific code smells from Zhang et al. (CAIN 2022) into Type I, Type II, and out-of-scope categories, along with the rationale for each decision, fix strategy, and whether the smell is currently implemented in the prototype. This directly supports Section 3.2 of the paper.

### `prompts/`

The exact prompt templates used by the SLM-based refactoring pathway for each targeted Type II smell. Placeholders such as `{code_snippet}` and `{smell_line}` are filled in at runtime by the refactoring script. This supports the reproducibility of the Type II pathway described in Section 3.3.

### `knowledge_base/`

Structured JSON files containing the full specification for each targeted Type II smell, including descriptions, canonical before/after examples, known false positive patterns, context window hints, and verification checks. The prompt templates are curated subsets of these entries. This supports Section 3.3 of the paper.

### `pbt/`

Property-based testing modules for the two implemented Type I smells. Each file takes a before and after Python file as input, extracts smell patterns using AST analysis, and runs Hypothesis-based equivalence tests to verify that the refactoring preserves behavioral equivalence. PBT also serves as a false positive filter. This supports Section 3.3 of the paper.

### `refactoring/`

The CST-based refactoring engine for Type I smells. Takes a CodeSmile-generated CSV as input and applies deterministic, formatting-preserving transformations using `libcst`. The SLM-based engine for Type II smells will be added upon acceptance.

### `evaluation/`

The list of 14 real-world repositories from the NICHE dataset used in the evaluation, including their GitHub URLs, application domains, and which targeted smells were detected in each repository. This supports Section 4.1 of the paper.

## Implemented Smells

| Smell                                             | Type    | Pathway   | Status      |
| ------------------------------------------------- | ------- | --------- | ----------- |
| Chain Indexing                                    | Type I  | CST + PBT | Implemented |
| Dataframe Conversion API Misused                  | Type I  | CST + PBT | Implemented |
| Gradients Not Cleared Before Backward Propagation | Type II | SLM       | Implemented |
| Merge API Parameter Not Explicitly Set            | Type II | SLM       | Implemented |

## Dependencies

```bash
# For CST-based refactoring
pip install libcst pandas numpy

# For PBT
pip install hypothesis pandas numpy pytest

# For SLM-based refactoring
pip install transformers torch accelerate
```

## Work in Progress

This repository is currently a work in progress and will be finalized upon acceptance. The following will be added:

- Before/after refactoring pairs from the experimental kits and NICHE repositories
- SLM-based refactoring scripts for Type II smells
- Full experimental data and CodeSmile output CSVs
