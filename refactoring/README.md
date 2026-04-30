## CST-Based Refactoring

The CST engine handles Type~I smells — structural anti-patterns resolvable within a single statement or localized code block. It operates on a Concrete Syntax Tree (CST) rather than a standard AST, preserving all formatting details such as whitespace, indentation, and comments, ensuring that refactored code integrates naturally into the existing project.

### Smells Currently Implemented

| Smell                             | Transformation                      |
| --------------------------------- | ----------------------------------- |
| Chain Indexing                    | `df["col"][i]` → `df.loc[i, "col"]` |
| Dataframe Conversion API Misused  | `df.values` → `df.to_numpy()`       |
| Matrix Multiplication API Misused | `np.dot(A, B)` → `np.matmul(A, B)`  |

### Dependencies

```bash
pip install libcst pandas numpy
```

### Input Format

The script takes a CSV file with the following columns:

| Column       | Description                                                                                      |
| ------------ | ------------------------------------------------------------------------------------------------ |
| `filename`   | Absolute path to the source file                                                                 |
| `smell_name` | One of `Chain_Indexing`, `dataframe_conversion_api_misused`, `matrix_multiplication_api_misused` |
| `line`       | Line number where the smell occurs (1-indexed)                                                   |

This CSV is produced by running CodeSmile on the target repositories.

### Usage

```bash
python smell_refactorer.py path/to/smells.csv
```

Refactored files are saved to the output directory specified in the script (default: `refactored_files/`). A summary report is generated at `refactored_files/refactoring_summary.txt`.

### How It Works

1. The script reads the input CSV and groups smell instances by file.
2. For each file, it parses the source code into a CST using `libcst`.
3. A smell-specific transformer visits the relevant nodes and applies the deterministic fix in-place, preserving all formatting.
4. The refactored code is written to the output directory.

### False Positive Filtering

The CST engine includes a lightweight false positive filter. If the flagged expression does not match the expected pattern — for instance, if `.values` is accessed on a non-DataFrame object — the transformation is skipped and the instance is logged for human review. This filtering is complemented by the PBT module in the `/pbt` directory, which provides a more rigorous behavioral equivalence check.

---

## SLM-Based Refactoring (Coming upon acceptance)

The SLM-based engine for Type~II smells — including Gradients Not Cleared Before Backward Propagation and Merge API Parameter Not Explicitly Set — will be added to `slm/` upon acceptance. The prompt templates and knowledge base entries used by the SLM engine are already available in the `/prompts` and `/knowledge_base` directories respectively.
