# HARP-ML Smell Taxonomy Classification

This file contains the full classification of the 22 ML-specific code smells from the catalogue proposed by Zhang et al. (CAIN 2022) into three categories used by the HARP-ML framework.

## Classification Criteria

A smell is classified as:

- **Type I** if its fix is syntactically complete and verifiable within a single statement or localized block, without requiring broader pipeline context. These smells are handled by the CST-based refactoring engine with Property-Based Testing (PBT) validation.

- **Type II** if its resolution depends on control flow or execution context that cannot be captured by local pattern matching. These smells are handled by the SLM-based refactoring engine with a focused context window.

- **Out-of-Scope** if its resolution requires data-level or cross-file reasoning that is beyond the scope of the current prototype.

## Classification Process

Two authors independently classified each of the 22 smells against the predefined criterion above. Disagreements were resolved through discussion. Of the 22 smells, 6 were classified as Type I, 8 as Type II, and 8 as out-of-scope for the current prototype.

## File Description

`smell_classification.csv` contains the following columns:

| Column                   | Description                                     |
| ------------------------ | ----------------------------------------------- |
| Smell Name               | Name of the smell as defined in Zhang et al.    |
| Pipeline Stage           | ML pipeline stage where the smell occurs        |
| Effect                   | Impact of the smell on code quality             |
| HARP-ML Category         | Type I / Type II / Out-of-Scope                 |
| Reasoning                | Rationale for the classification decision       |
| Fix Strategy             | High-level description of the fix approach      |
| Example Before           | Minimal code example showing the smell          |
| Example After            | Minimal code example showing the fix            |
| Implemented in Prototype | Whether HARP-ML currently implements this smell |

## Reference

Zhang, H., Cruz, L., and van Deursen, A. (2022). Code Smells for Machine Learning Applications. In Proceedings of the 1st International Conference on AI Engineering: Software Engineering for AI (CAIN '22). https://doi.org/10.1145/3522664.3528620
