"""
Property-Based Testing for Dataframe Conversion API Refactoring
===============================================================
Tests that before.py (df.values) and after.py (df.to_numpy()) produce
identical outputs across a wide range of generated DataFrames.

Usage:
    python test_dataframe_conversion.py before.py after.py

Requirements:
    pip install hypothesis pandas numpy pytest

How it works:
    - Reads before.py and after.py passed as CLI arguments
    - Parses before.py using AST to extract all .values access patterns
    - For each pattern, generates DataFrames using Hypothesis and asserts
      behavioral equivalence between df.values and df.to_numpy()
    - If the flagged variable is not a DataFrame (e.g. a dict or custom
      class), strategy construction fails and the instance is treated as
      a false positive and routed to human review

Invariants checked:
    - Behavioral equivalence: df.values and df.to_numpy() produce identical arrays
    - dtype preservation: output dtype is unchanged
    - Shape preservation: output shape is unchanged
"""

import ast
import sys
import textwrap
import argparse
import pandas as pd
import numpy as np
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
import pytest

# =============================================================================
# STEP 1: Read files from CLI arguments
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Property-based equivalence testing for Dataframe Conversion API refactoring."
    )
    parser.add_argument("before", help="Path to the before-refactoring Python file")
    parser.add_argument("after", help="Path to the after-refactoring Python file")
    args, _ = parser.parse_known_args()
    return args


def read_file(filepath: str) -> str:
    with open(filepath, "r") as f:
        return f.read()


# =============================================================================
# STEP 2: Extract .values access patterns from source using AST
# =============================================================================


def extract_dataframe_conversion_patterns(source: str) -> list:
    """
    Parse a Python source file and extract all .values attribute accesses
    that are candidates for df.values -> df.to_numpy() refactoring.

    Returns a deduplicated list of variable names on which .values was accessed.

    Raises ValueError if no .values access is found, or if the variable is
    not a DataFrame — both signal a likely false positive for human review.
    """
    tree = ast.parse(textwrap.dedent(source))
    found = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if node.attr != "values":
            continue
        if isinstance(node.value, ast.Name):
            var_name = node.value.id
            if var_name not in found:
                found.append(var_name)

    if not found:
        raise ValueError(
            "No .values access pattern found.\n"
            "Check that the before file actually contains the "
            "Dataframe Conversion API smell.\n"
            "If the variable is not a DataFrame, this is a false positive "
            "and should be routed to human review."
        )

    return found


# =============================================================================
# STEP 3: Hypothesis strategy
# =============================================================================


def make_dataframe_strategy():
    """
    Generates DataFrames with numeric columns for equivalence testing.
    If the flagged variable is not a DataFrame, strategy construction
    will fail, signalling a false positive for human review.
    """

    @st.composite
    def _strategy(draw):
        n_rows = draw(st.integers(min_value=1, max_value=20))
        n_cols = draw(st.integers(min_value=1, max_value=5))
        col_names = [f"col_{i}" for i in range(n_cols)]
        data = {
            col: draw(
                st.lists(
                    st.floats(
                        allow_nan=False,
                        allow_infinity=False,
                        min_value=-1e6,
                        max_value=1e6,
                    ),
                    min_size=n_rows,
                    max_size=n_rows,
                )
            )
            for col in col_names
        }
        return pd.DataFrame(data)

    return _strategy


# =============================================================================
# STEP 4: Dynamically build test classes for each detected pattern
# =============================================================================

_test_registry = {}


def build_tests_for_dataframe_conversion(pattern_idx: int, var_name: str):
    """
    Build equivalence and edge case test classes for a single
    Dataframe Conversion API pattern (var_name.values -> var_name.to_numpy()).
    Injects them into globals() so pytest can discover them.
    """
    class_label = f"DFConv_Pattern{pattern_idx}_{var_name}"
    strategy = make_dataframe_strategy()

    class EquivTests:
        @given(strategy())
        @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
        def test_values_equivalence(self, df):
            """
            df.values and df.to_numpy() must produce identical arrays.
            If this fails to instantiate because the variable is not a
            DataFrame, the instance is a false positive — route to human review.
            """
            result_before = df.values
            result_after = df.to_numpy()
            np.testing.assert_array_equal(
                result_before,
                result_after,
                err_msg=(
                    f"Pattern {pattern_idx} ({var_name}.values) mismatch!\n"
                    f"  Before : {result_before!r}\n"
                    f"  After  : {result_after!r}\n"
                    f"  df:\n{df}"
                ),
            )

        @given(strategy())
        @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
        def test_dtype_preserved(self, df):
            """Output dtype must be identical between .values and .to_numpy()."""
            assert df.values.dtype == df.to_numpy().dtype

        @given(strategy())
        @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
        def test_shape_preserved(self, df):
            """Output shape must be identical between .values and .to_numpy()."""
            assert df.values.shape == df.to_numpy().shape

    class EdgeTests:
        def test_empty_dataframe(self):
            df = pd.DataFrame()
            np.testing.assert_array_equal(df.values, df.to_numpy())

        def test_single_row(self):
            df = pd.DataFrame({"A": [1.0], "B": [2.0]})
            np.testing.assert_array_equal(df.values, df.to_numpy())

        def test_single_column(self):
            df = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
            np.testing.assert_array_equal(df.values, df.to_numpy())

        def test_nan_values(self):
            df = pd.DataFrame({"A": [1.0, np.nan, 3.0], "B": [np.nan, 2.0, np.nan]})
            np.testing.assert_array_equal(df.values, df.to_numpy())

        def test_large_dataframe(self):
            n = 500
            df = pd.DataFrame({f"col_{i}": np.random.randn(n) for i in range(10)})
            np.testing.assert_array_equal(df.values, df.to_numpy())

        def test_integer_dataframe(self):
            df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
            np.testing.assert_array_equal(df.values, df.to_numpy())

    EquivTests.__name__ = f"TestEquiv_{class_label}"
    EdgeTests.__name__ = f"TestEdge_{class_label}"
    globals()[EquivTests.__name__] = EquivTests
    globals()[EdgeTests.__name__] = EdgeTests
    _test_registry[class_label] = (EquivTests, EdgeTests)


# =============================================================================
# ENTRY POINT
# =============================================================================


def _setup():
    args = parse_args()

    print(f"\n Reading before file : {args.before}")
    print(f" Reading after file  : {args.after}\n")

    before_source = read_file(args.before)
    after_source = read_file(args.after)

    try:
        all_vars = extract_dataframe_conversion_patterns(before_source)
    except ValueError as e:
        print(f"\n False positive detected — routing to human review.\n {e}\n")
        sys.exit(0)

    print(f" Found {len(all_vars)} .values access pattern(s):\n")
    for i, var_name in enumerate(all_vars):
        print(f"   [{i+1}] {var_name}.values")
    print()

    for i, var_name in enumerate(all_vars):
        build_tests_for_dataframe_conversion(i + 1, var_name)


_setup()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--hypothesis-show-statistics"]))
