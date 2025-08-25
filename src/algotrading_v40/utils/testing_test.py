import numpy as np
import pandas as pd
import pytest

import algotrading_v40.utils.testing as ut


def _mutate_series(s: pd.Series) -> None:
  s.iloc[0] = 999


def _mutate_dataframe(df: pd.DataFrame) -> None:
  df.loc[0, "a"] = 999


def _mutate_index(idx: pd.Index) -> None:
  # Index objects are “immutable” w.r.t. their values but the `name`
  # attribute can be changed in-place.
  idx.rename("mutated", inplace=True)


def _mutate_ndarray(arr: np.ndarray) -> None:
  arr[0] = 999


def _mutate_list(lst: list) -> None:
  lst.append(999)


_MUTATING_CASES = [
  ("series", pd.Series([1, 2, 3]), _mutate_series, None),
  ("dataframe", pd.DataFrame({"a": [1, 2, 3]}), _mutate_dataframe, None),
  (
    "index",
    pd.Index([1, 2, 3]),
    _mutate_index,
    None,
  ),
  ("ndarray", np.array([1, 2, 3]), _mutate_ndarray, None),
  ("list", [1, 2, 3], _mutate_list, None),
]

_NON_MUTATING_FACTORIES = [
  ("series", lambda: pd.Series([1, 2, 3])),
  ("dataframe", lambda: pd.DataFrame({"a": [1, 2, 3]})),
  ("index", lambda: pd.Index([1, 2, 3])),
  ("ndarray", lambda: np.array([1, 2, 3])),
  ("list", lambda: [1, 2, 3]),
  ("tuple", lambda: (1, 2, 3)),
  ("int", lambda: 42),
]


@pytest.mark.parametrize("label,obj,mutator,custom_equal", _MUTATING_CASES)
def test_expect_no_mutation_detects_mutation(label, obj, mutator, custom_equal):
  ctx_kwargs = {} if custom_equal is None else {"equal": custom_equal}

  with pytest.raises(AssertionError):
    with ut.expect_no_mutation(obj, **ctx_kwargs):
      mutator(obj)


@pytest.mark.parametrize("label,factory", _NON_MUTATING_FACTORIES)
def test_expect_no_mutation_allows_no_mutation(label, factory):
  obj = factory()
  with ut.expect_no_mutation(obj):
    pass


def test_expect_no_mutation_detects_mutation_multiple_args_both_mutated():
  def _test_func_1(df: pd.DataFrame, s: pd.Series) -> None:
    df.loc[0, "a"] = 999
    s.iloc[0] = 999

  df = pd.DataFrame({"a": [1, 2, 3]})
  s = pd.Series([1, 2, 3])
  with pytest.raises(AssertionError):
    with ut.expect_no_mutation(df, s):
      _test_func_1(df, s)


def test_expect_no_mutation_detects_mutation_multiple_args_one_mutated():
  def _test_func_2(df: pd.DataFrame, s: pd.Series) -> None:
    df.loc[0, "a"] = 999

  df = pd.DataFrame({"a": [1, 2, 3]})
  s = pd.Series([1, 2, 3])
  with pytest.raises(AssertionError):
    with ut.expect_no_mutation(df, s):
      _test_func_2(df, s)

  # reset the objects
  df = pd.DataFrame({"a": [1, 2, 3]})
  s = pd.Series([1, 2, 3])
  with pytest.raises(AssertionError):
    with ut.expect_no_mutation(df):
      _test_func_2(df, s)


def test_expect_no_mutation_allows_non_target_arg_to_be_mutated():
  def _test_func_3(df: pd.DataFrame, s: pd.Series) -> None:
    df.loc[0, "a"] = 999

  df = pd.DataFrame({"a": [1, 2, 3]})
  s = pd.Series([1, 2, 3])
  with ut.expect_no_mutation(s):
    # _test_func_3 mutates the non-target object df
    _test_func_3(df, s)


def test_expect_no_mutation_is_robust_to_order_of_arguments():
  def _test_func_4(df: pd.DataFrame, s: pd.Series) -> None:
    if not df.equals(pd.DataFrame({"a": [1, 2, 3]})):
      raise ValueError("df is not equal to the original DataFrame")
    if not s.equals(pd.Series([1, 2, 3])):
      raise ValueError("s is not equal to the original Series")
    # if we return without raising an error, the objects were passed correctly

  df = pd.DataFrame({"a": [1, 2, 3]})
  s = pd.Series([1, 2, 3])
  # expect_no_mutation called with s, df
  with ut.expect_no_mutation(s, df):
    # function called with df, s (reversed order)
    _test_func_4(df, s)


def test_get_test_df():
  import datetime as dt

  start_date = dt.date(2023, 1, 1)
  end_date = dt.date(2023, 1, 5)

  df = ut.get_test_df(start_date, end_date)

  assert isinstance(df, pd.DataFrame)
  assert set(df.columns) == {"open", "high", "low", "close", "volume"}
  assert (df["volume"] == 1).all()
  assert len(df) > 0
