#!/usr/bin/env python3
"""
Simplified refactoring tool for ML code smells using libCST.
This version focuses on actually detecting and transforming the patterns.
"""

import libcst as cst
import csv
from pathlib import Path
from collections import defaultdict


class ChainIndexingTransformer(cst.CSTTransformer):
    """
    Transforms chain indexing like df["col1"]["col2"]
    into df.loc[:, ("col1", "col2")]
    """

    def leave_Subscript(
        self, original_node: cst.Subscript, updated_node: cst.Subscript
    ) -> cst.BaseExpression:
        if not isinstance(updated_node.value, cst.Subscript):
            return updated_node

        inner_subscript = updated_node.value
        base = inner_subscript.value

        try:
            first_element = inner_subscript.slice[0]
            if not isinstance(first_element.slice, cst.Index):
                return updated_node
            first_key = first_element.slice.value

            second_element = updated_node.slice[0]
            if not isinstance(second_element.slice, cst.Index):
                return updated_node
            second_key = second_element.slice.value

        except (IndexError, AttributeError):
            return updated_node

        return cst.Subscript(
            value=cst.Attribute(value=base, attr=cst.Name("loc")),
            slice=[
                cst.SubscriptElement(slice=cst.Slice(lower=None, upper=None)),
                cst.SubscriptElement(
                    slice=cst.Index(
                        value=cst.Tuple(
                            elements=[
                                cst.Element(value=first_key),
                                cst.Element(value=second_key),
                            ],
                            lpar=[cst.LeftParen()],
                            rpar=[cst.RightParen()],
                        )
                    )
                ),
            ],
        )


class MatrixMultiplicationTransformer(cst.CSTTransformer):
    """
    Transforms np.dot() to np.matmul() for 2D matrix multiplication.
    Handles various numpy aliases: np, onp, numpy, jnp, etc.
    """

    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug
        self.transformations = 0

    def leave_Call(
        self, original_node: cst.Call, updated_node: cst.Call
    ) -> cst.BaseExpression:
        if isinstance(updated_node.func, cst.Attribute):
            if updated_node.func.attr.value == "dot":
                if isinstance(updated_node.func.value, cst.Name):
                    alias = updated_node.func.value.value
                    if alias in ["np", "onp", "numpy", "jnp", "jax.numpy"]:
                        if self.debug:
                            print(
                                f"  DEBUG: Found {alias}.dot(), replacing with {alias}.matmul()"
                            )
                        self.transformations += 1
                        return updated_node.with_changes(
                            func=cst.Attribute(
                                value=updated_node.func.value,
                                attr=cst.Name("matmul"),
                            )
                        )
        return updated_node


class DataframeConversionAPITransformer(cst.CSTTransformer):
    """
    Transforms df.values to df.to_numpy() for DataFrame to NumPy conversion.

    Handles two forms:
        df.values        ->  df.to_numpy()
        df.values()      ->  df.to_numpy()   (rare but possible misuse)

    Cases where .values is used on non-DataFrame objects (e.g. dict.values())
    are skipped by checking that the attribute access is not a standalone
    builtin pattern — routed to human review via the false positive counter.
    """

    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug
        self.transformations = 0
        self.false_positives = 0

    def _is_dict_values(self, node: cst.Attribute) -> bool:
        """
        Heuristic: if the object name contains 'dict' or ends with 'd'
        and is a plain Name, it is likely a dict — skip it.
        This is a lightweight false positive guard; PBT handles the rest.
        """
        if isinstance(node.value, cst.Name):
            name = node.value.value.lower()
            if "dict" in name or name.endswith("_d"):
                return True
        return False

    def leave_Attribute(
        self,
        original_node: cst.Attribute,
        updated_node: cst.Attribute,
    ) -> cst.BaseExpression:
        # Only interested in .values access
        if updated_node.attr.value != "values":
            return updated_node

        # Skip likely dict.values() patterns
        if self._is_dict_values(updated_node):
            self.false_positives += 1
            if self.debug:
                print(
                    f"  DEBUG: Skipping .values on likely dict object "
                    f"(false positive)"
                )
            return updated_node

        if self.debug:
            print(f"  DEBUG: Found .values accessor, replacing with .to_numpy()")

        self.transformations += 1

        # Replace .values with .to_numpy()
        # We return a Call node wrapping the new attribute
        return cst.Call(
            func=cst.Attribute(
                value=updated_node.value,
                attr=cst.Name("to_numpy"),
                dot=updated_node.dot,
            ),
            args=[],
        )


def refactor_chain_indexing(source_code: str) -> str:
    """Refactor chain indexing pattern."""
    try:
        tree = cst.parse_module(source_code)
        transformer = ChainIndexingTransformer()
        modified_tree = tree.visit(transformer)
        return modified_tree.code
    except Exception as e:
        print(f"Error refactoring chain indexing: {e}")
        import traceback

        traceback.print_exc()
        return source_code


def refactor_matrix_multiplication(source_code: str, debug=False) -> str:
    """Refactor np.dot() to np.matmul() for matrix multiplication."""
    try:
        tree = cst.parse_module(source_code)
        transformer = MatrixMultiplicationTransformer(debug=debug)
        modified_tree = tree.visit(transformer)
        if debug or transformer.transformations > 0:
            print(f"  DEBUG: Made {transformer.transformations} transformation(s)")
        return modified_tree.code
    except Exception as e:
        print(f"Error refactoring matrix multiplication: {e}")
        import traceback

        traceback.print_exc()
        return source_code


def refactor_dataframe_conversion_api(
    source_code: str, debug=False
) -> tuple[str, int, int]:
    """
    Refactor df.values to df.to_numpy() for DataFrame to NumPy conversion.

    Returns:
        Tuple of (refactored_code, transformations_made, false_positives_detected)
    """
    try:
        tree = cst.parse_module(source_code)
        transformer = DataframeConversionAPITransformer(debug=debug)
        modified_tree = tree.visit(transformer)
        if debug or transformer.transformations > 0:
            print(
                f"  DEBUG: Made {transformer.transformations} transformation(s), "
                f"skipped {transformer.false_positives} false positive(s)"
            )
        return (
            modified_tree.code,
            transformer.transformations,
            transformer.false_positives,
        )
    except Exception as e:
        print(f"Error refactoring dataframe conversion API: {e}")
        import traceback

        traceback.print_exc()
        return source_code, 0, 0


def process_csv(
    csv_path: str,
    output_dir: str = "/home/garguy/Gargie/IIITH/Research/smell_ml/refactored_files",
):
    """
    Process CSV file and refactor identified smells.

    Args:
        csv_path: Path to the CSV file
        output_dir: Directory to save refactored files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_smells = defaultdict(lambda: defaultdict(list))

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            file_path = row["filename"]
            smell_name = row["smell_name"]
            line_num = int(row["line"]) if row.get("line") else None
            file_smells[file_path][smell_name].append(line_num)

    refactored_files = []

    for file_path, smells in file_smells.items():
        source_file = Path(file_path)
        if not source_file.exists():
            print(f"Warning: File not found: {source_file}")
            continue

        try:
            with open(source_file, "r") as src:
                source_code = src.read()
        except Exception as e:
            print(f"Error reading {source_file}: {e}")
            continue

        print(f"\nProcessing: {file_path}")

        refactored_code = source_code

        for smell_name, line_numbers in smells.items():
            if smell_name == "Chain_Indexing":
                print(f"  Refactoring Chain Indexing at lines {line_numbers}")
                refactored_code = refactor_chain_indexing(refactored_code)

            elif smell_name == "matrix_multiplication_api_misused":
                print(
                    f"  Refactoring Matrix Multiplication API at lines {line_numbers}"
                )
                refactored_code = refactor_matrix_multiplication(
                    refactored_code, debug=True
                )

            elif smell_name == "dataframe_conversion_api_misused":
                print(f"  Refactoring Dataframe Conversion API at lines {line_numbers}")
                refactored_code, n_transforms, n_fp = refactor_dataframe_conversion_api(
                    refactored_code, debug=True
                )
                if n_fp > 0:
                    print(
                        f"  ⚠ {n_fp} instance(s) skipped "
                        f"(likely dict.values() — routed to human review)"
                    )

        if refactored_code != source_code:
            output_file = output_path / Path(file_path).name
            with open(output_file, "w") as out:
                out.write(refactored_code)

            refactored_files.append(
                {
                    "original": file_path,
                    "refactored": str(output_file),
                    "smells": list(smells.keys()),
                }
            )
            print(f"  ✓ Saved to {output_file}")
        else:
            print(f"  ⚠ No changes detected - pattern may not match")

    summary_file = output_path / "refactoring_summary.txt"
    with open(summary_file, "w") as f:
        f.write("ML Code Smell Refactoring Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total files refactored: {len(refactored_files)}\n\n")

        for item in refactored_files:
            f.write(f"Original: {item['original']}\n")
            f.write(f"Smells: {', '.join(item['smells'])}\n")
            f.write(f"Refactored: {item['refactored']}\n")
            f.write("-" * 60 + "\n")

    print(f"\n{'='*60}")
    print(f"Refactoring complete!")
    print(f"Files refactored: {len(refactored_files)}")
    print(f"Summary: {summary_file}")
    return refactored_files


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python smell_refactorer.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    process_csv(csv_file)
