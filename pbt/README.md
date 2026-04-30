# HARP-ML Property-Based Testing (PBT) Invariants

This directory contains the property-based testing modules used by the Type~I refactoring pathway in HARP-ML to validate behavioral equivalence between original and refactored code.

## Overview

For each Type~I smell, a predefined PBT template establishes the invariants that must hold across the transformation. The PBT module parses the local AST of the flagged file to extract relevant variables, then instantiates a smell-specific test suite using the `hypothesis` framework. Tests are run against synthetically generated inputs to assert behavioral equivalence without requiring manual test oracles.

PBT also serves as a false positive filter: if a valid test cannot be instantiated — for instance because the flagged variable is not a DataFrame but a plain Python list or dictionary — the strategy construction fails and the instance is routed to human review rather than being silently refactored.

## Files

| File                         | Smell                            | Transformation                      |
| ---------------------------- | -------------------------------- | ----------------------------------- |
| test_chain_indexing.py       | Chain Indexing                   | `df['col'][i]` → `df.loc[i, 'col']` |
| test_dataframe_conversion.py | Dataframe Conversion API Misused | `df.values` → `df.to_numpy()`       |

## Invariants Checked

### Chain Indexing (`test_chain_indexing.py`)

| Invariant              | Test                                                       | Examples |
| ---------------------- | ---------------------------------------------------------- | -------- |
| Behavioral equivalence | `test_flat_df_equivalence` / `test_multiindex_equivalence` | 200      |
| Type preservation      | `test_type_preserved`                                      | 100      |
| Shape preservation     | `test_shape_preserved`                                     | 100      |

Edge cases covered: empty DataFrames, single-row DataFrames, NaN values, string slicing, numeric values, large DataFrames (500 rows).

### Dataframe Conversion API (`test_dataframe_conversion.py`)

| Invariant              | Test                      | Examples |
| ---------------------- | ------------------------- | -------- |
| Behavioral equivalence | `test_values_equivalence` | 200      |
| dtype preservation     | `test_dtype_preserved`    | 100      |
| Shape preservation     | `test_shape_preserved`    | 100      |

Edge cases covered: empty DataFrames, single-row DataFrames, single-column DataFrames, NaN values, integer DataFrames, large DataFrames (500 rows).

## Usage

```bash
# Install dependencies
pip install hypothesis pandas numpy pytest

# Run Chain Indexing tests
python test_chain_indexing.py path/to/before.py path/to/after.py

# Run Dataframe Conversion tests
python test_dataframe_conversion.py path/to/before.py path/to/after.py

# Run with detailed hypothesis statistics
python test_chain_indexing.py before.py after.py -v --hypothesis-show-statistics
```

## Interpreting Results

**All tests pass** — the refactoring is behaviorally equivalent and the fix is accepted.

**Tests fail** — the refactoring introduced a behavioral regression. The instance is flagged for human review.

**False positive detected** — the AST extraction found no valid DataFrame pattern, meaning the flagged expression is likely operating on a non-DataFrame object (e.g. a Python list or dictionary). The instance is routed to human review rather than being silently refactored. This is printed as:

```
False positive detected — routing to human review.
```

## Design Notes

Test classes are dynamically generated at import time based on the patterns extracted from the before file, and injected into the global namespace so pytest can discover them automatically. This means the same test file handles any number of smell instances in the input file without requiring manual configuration.

The `hypothesis` framework uses shrinking to find minimal failing examples when a test fails, making it easier to understand what input triggered the behavioral difference.
