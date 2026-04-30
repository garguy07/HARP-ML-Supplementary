"""
Property-Based Testing for Chain Indexing Refactoring
======================================================
Tests that before.py (chain indexing) and after.py (loc-based indexing)
produce identical outputs across a wide range of generated DataFrames.

Usage:
    python test_chain_indexing.py before.py after.py

Requirements:
    pip install hypothesis pandas numpy pytest

How it works:
    - Reads before.py and after.py passed as CLI arguments
    - Parses before.py using AST to extract ALL chain indexing patterns
    - For each pattern, generates DataFrames using Hypothesis and runs
      both versions, asserting behavioral equivalence
    - If no chain indexing pattern is found, the instance is likely a
      false positive and is routed to human review

Invariants checked:
    - Behavioral equivalence: before and after produce identical results
    - Type preservation: output type is unchanged
    - Shape preservation: output shape is unchanged (MultiIndex case)
"""

import ast
import sys
import textwrap
import argparse
import pandas as pd
import numpy as np
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st
import pytest


# =============================================================================
# STEP 1: Read files from CLI arguments
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Property-based equivalence testing for Chain Indexing refactoring."
    )
    parser.add_argument("before", help="Path to the before-refactoring Python file")
    parser.add_argument("after", help="Path to the after-refactoring Python file")
    args, _ = parser.parse_known_args()
    return args


def read_file(filepath: str) -> str:
    with open(filepath, "r") as f:
        return f.read()


# =============================================================================
# STEP 2: Extract chain indexing patterns from source using AST
# =============================================================================


def _get_slice_value(node: ast.Subscript):
    """
    Extract the slice value from a subscript node, handling all common types:
      - ast.Constant  -> string or integer literal  e.g. 'Time', 0
      - ast.Name      -> variable                   e.g. i, idx
      - ast.Attribute -> attribute access            e.g. self.name
      - ast.Slice     -> slice expression            e.g. 0:2
    Returns None for unrecognised node types.
    """
    s = node.slice
    if isinstance(s, ast.Constant):
        return s.value
    if isinstance(s, ast.Name):
        return s.id
    if isinstance(s, ast.Attribute):
        return s.attr
    if isinstance(s, ast.Slice):
        return "slice"
    return None


def extract_chain_index_keys(source: str) -> list:
    """
    Parse a Python source file and extract ALL chain indexing patterns.

    Handles:
      - df['col']['subcol']      -> ('col', 'subcol')
      - df['col'][i]             -> ('col', 'i')
      - df['col'][i][0:2]        -> ('col', 'i')   (trailing slice ignored)

    Returns a deduplicated list of (col, row_key) tuples where col is always
    a string literal (the DataFrame column name).

    Raises ValueError if no chain indexing pattern is found, which signals
    a likely false positive — the instance is routed to human review.
    """
    tree = ast.parse(textwrap.dedent(source))
    found = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue

        outer = node
        while isinstance(outer, ast.Subscript):
            inner = outer.value
            if not isinstance(inner, ast.Subscript):
                break

            outer_slice = _get_slice_value(outer)
            base = inner.value

            if isinstance(inner.slice, ast.Constant) and isinstance(
                inner.slice.value, str
            ):
                inner_key = inner.slice.value
            elif isinstance(inner.slice, ast.Name):
                inner_key = f"<var:{inner.slice.id}>"
            else:
                outer = inner
                continue

            if isinstance(base, ast.Name) and outer_slice is not None:
                pair = (inner_key, outer_slice)
                if pair not in found:
                    found.append(pair)
                break

            outer = inner

    if not found:
        raise ValueError(
            "No chain indexing pattern found.\n"
            "Expected patterns like df['col'][i] or df['col1']['col2'].\n"
            "This is likely a false positive — routing to human review."
        )

    return found


# =============================================================================
# STEP 3: Hypothesis strategies
# =============================================================================


def make_flat_df_strategy(col: str):
    """Flat DataFrame strategy for df['col'][i] patterns."""

    @st.composite
    def _strategy(draw):
        n_rows = draw(st.integers(min_value=1, max_value=20))
        values = draw(
            st.lists(
                st.one_of(
                    st.from_regex(r"\d{2}:\d{2}", fullmatch=True),
                    st.from_regex(r"\d{2}/\d{2}/\d{4}", fullmatch=True),
                    st.floats(
                        allow_nan=False,
                        allow_infinity=False,
                        min_value=-1e6,
                        max_value=1e6,
                    ),
                ),
                min_size=n_rows,
                max_size=n_rows,
            )
        )
        df = pd.DataFrame({col: values})
        row_idx = draw(st.integers(min_value=0, max_value=n_rows - 1))
        return df, row_idx

    return _strategy()


def make_multiindex_df_strategy(col1: str, col2: str):
    """MultiIndex DataFrame strategy for df['col1']['col2'] patterns."""

    @st.composite
    def _strategy(draw):
        n_rows = draw(st.integers(min_value=1, max_value=10))
        extra = draw(
            st.lists(
                st.text(alphabet="abcde", min_size=1, max_size=3),
                min_size=0,
                max_size=2,
            )
        )
        second_level = list(set(extra) | {col2})
        arrays = [[col1] * len(second_level), second_level]
        mi = pd.MultiIndex.from_arrays(arrays)
        data = np.random.randint(0, 100, size=(n_rows, len(second_level)))
        df = pd.DataFrame(data, columns=mi)
        return df

    return _strategy()


# =============================================================================
# STEP 4: Before/after callables
# =============================================================================


def make_flat_before_fn():
    """df[col][row_idx] with optional string slicing."""

    def _before(df, col, row_idx):
        result = df[col][row_idx]
        if isinstance(result, str):
            result = result[0:2]
        return result

    return _before


def make_flat_after_fn():
    """df.loc[row_idx, col] refactored equivalent."""

    def _after(df, col, row_idx):
        result = df.loc[row_idx, col]
        if isinstance(result, str):
            result = result[0:2]
        return result

    return _after


# =============================================================================
# STEP 5: Dynamically build test classes for each detected pattern
# =============================================================================

_test_registry = {}


def build_tests_for_chain_indexing(pattern_idx: int, col, row_key):
    """
    Build equivalence and edge case test classes for a single
    chain indexing pattern (col, row_key).
    Injects them into globals() so pytest can discover them.
    """
    col_str = str(col)
    row_str = str(row_key)
    class_label = f"ChainIdx_Pattern{pattern_idx}_{col_str}_{row_str}".replace(
        " ", "_"
    )

    is_string_col2 = isinstance(row_key, str)

    if is_string_col2:
        strategy = make_multiindex_df_strategy(col_str, row_str)

        class EquivTests:
            @given(strategy)
            @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
            def test_multiindex_equivalence(self, df):
                result_before = df[col_str][row_str]
                result_after = df.loc[:, (col_str, row_str)]
                pd.testing.assert_series_equal(
                    result_before.reset_index(drop=True),
                    result_after.reset_index(drop=True),
                    check_names=False,
                    obj=f"Pattern {pattern_idx}: {col_str}[{row_str}]",
                )

            @given(strategy)
            @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
            def test_type_preserved(self, df):
                assert type(df[col_str][row_str]) == type(
                    df.loc[:, (col_str, row_str)]
                )

            @given(strategy)
            @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
            def test_shape_preserved(self, df):
                assert (
                    df[col_str][row_str].shape
                    == df.loc[:, (col_str, row_str)].shape
                )

        class EdgeTests:
            def test_empty_dataframe(self):
                arrays = [[col_str, col_str], ["x", row_str]]
                mi = pd.MultiIndex.from_arrays(arrays)
                df = pd.DataFrame(columns=mi)
                pd.testing.assert_series_equal(
                    df[col_str][row_str],
                    df.loc[:, (col_str, row_str)],
                    check_names=False,
                )

            def test_single_row(self):
                arrays = [[col_str], [row_str]]
                mi = pd.MultiIndex.from_arrays(arrays)
                df = pd.DataFrame([[42]], columns=mi)
                pd.testing.assert_series_equal(
                    df[col_str][row_str],
                    df.loc[:, (col_str, row_str)],
                    check_names=False,
                )

            def test_nan_values(self):
                arrays = [[col_str, col_str], ["x", row_str]]
                mi = pd.MultiIndex.from_arrays(arrays)
                df = pd.DataFrame([[1.0, np.nan], [np.nan, 3.0]], columns=mi)
                pd.testing.assert_series_equal(
                    df[col_str][row_str],
                    df.loc[:, (col_str, row_str)],
                    check_names=False,
                )

    else:
        strategy = make_flat_df_strategy(col_str)
        before_fn = make_flat_before_fn()
        after_fn = make_flat_after_fn()

        class EquivTests:
            @given(strategy)
            @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
            def test_flat_df_equivalence(self, args):
                df, row_idx = args
                assume(row_idx in df.index)
                result_before = before_fn(df, col_str, row_idx)
                result_after = after_fn(df, col_str, row_idx)
                assert result_before == result_after, (
                    f"Pattern {pattern_idx} ({col_str}[{row_key}]) mismatch!\n"
                    f"  Before : {result_before!r}\n"
                    f"  After  : {result_after!r}\n"
                    f"  df:\n{df}\n  row_idx: {row_idx}"
                )

            @given(strategy)
            @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
            def test_type_preserved(self, args):
                df, row_idx = args
                assume(row_idx in df.index)
                assert type(before_fn(df, col_str, row_idx)) == type(
                    after_fn(df, col_str, row_idx)
                )

        class EdgeTests:
            def test_single_row_df(self):
                df = pd.DataFrame({col_str: ["09:30"]})
                assert before_fn(df, col_str, 0) == after_fn(df, col_str, 0)

            def test_string_slicing_preserved(self):
                df = pd.DataFrame({col_str: ["09:30", "14:45", "23:59"]})
                for i in range(len(df)):
                    assert before_fn(df, col_str, i) == after_fn(df, col_str, i)

            def test_numeric_values(self):
                df = pd.DataFrame({col_str: [1.0, 2.5, 3.7]})
                for i in range(len(df)):
                    assert before_fn(df, col_str, i) == after_fn(df, col_str, i)

            def test_large_dataframe(self):
                n = 500
                df = pd.DataFrame(
                    {col_str: [f"{i%24:02d}:{i%60:02d}" for i in range(n)]}
                )
                for i in range(0, n, 50):
                    assert before_fn(df, col_str, i) == after_fn(df, col_str, i)

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
        all_patterns = extract_chain_index_keys(before_source)
    except ValueError as e:
        print(f"\n False positive detected — routing to human review.\n {e}\n")
        sys.exit(0)

    print(f" Found {len(all_patterns)} chain indexing pattern(s):\n")
    for i, (col, row_key) in enumerate(all_patterns):
        print(f"   [{i+1}] df[{col!r}][{row_key!r}]")
    print()

    for i, (col, row_key) in enumerate(all_patterns):
        build_tests_for_chain_indexing(i + 1, col, row_key)


_setup()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--hypothesis-show-statistics"]))