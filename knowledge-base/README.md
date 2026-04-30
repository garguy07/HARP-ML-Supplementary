# HARP-ML Knowledge Base

This directory contains the knowledge base entries used by the SLM-based
refactoring pathway (Type II) in HARP-ML. Each entry is a structured JSON
file that grounds the SLM's suggestions in smell-specific semantics rather
than generic code edits.

## Structure

Each JSON file corresponds to one targeted Type II smell and contains the
following fields:

| Field                   | Description                                                   |
| ----------------------- | ------------------------------------------------------------- |
| smell_id                | Short identifier for the smell                                |
| smell_name              | Full name of the smell                                        |
| type                    | Always II for this directory                                  |
| pipeline_stage          | ML pipeline stage where the smell occurs                      |
| library                 | The ML library where the smell manifests                      |
| description             | What the smell is and why it is harmful                       |
| fix_strategy            | High-level description of how to fix it                       |
| defaults                | Safe default values used when context is insufficient         |
| canonical_examples      | Before/after code examples used for few-shot prompting        |
| false_positive_patterns | Known patterns that resemble the smell but are intentional    |
| context_window_hints    | What to look for in the surrounding code to infer correct fix |
| verification_check      | What the verification step checks after refactoring           |

## Files

| File                       | Smell                                             | Implemented |
| -------------------------- | ------------------------------------------------- | ----------- |
| gradients_not_cleared.json | Gradients Not Cleared Before Backward Propagation | Yes         |
| merge_api_parameter.json   | Merge API Parameter Not Explicitly Set            | Yes         |

## Relationship to Prompt Templates

The knowledge base entries are comprehensive reference artifacts that document
the full design space for each smell. The prompt templates in `/prompts` are
curated subsets of these entries, selected to fit within the model's context
window while grounding its output in smell-specific semantics. Not all fields
in the knowledge base are included in every prompt.

## Future Entries

Knowledge base entries for the remaining Type II smells classified in the
taxonomy — Memory Not Freed, Randomness Uncontrolled, TensorArray Not Used,
Training/Evaluation Mode Improper Toggling, Deterministic Algorithm Option
Not Used, and In-Place APIs Misused — will be added as HARP-ML's coverage
expands to the full taxonomy defined by Zhang et al. (CAIN 2022).

## Reference

Zhang, H., Cruz, L., and van Deursen, A. (2022). Code Smells for Machine
Learning Applications. In Proceedings of the 1st International Conference
on AI Engineering: Software Engineering for AI (CAIN '22).
https://doi.org/10.1145/3522664.3528620
