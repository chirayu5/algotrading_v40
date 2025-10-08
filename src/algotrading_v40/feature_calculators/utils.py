from typing import Callable

import numpy as np
import pandas as pd

import algotrading_v40.bar_groupers.time_based_uniform as bg_tbu
import algotrading_v40.utils.df as u_df


def calculate_features_with_last_value_guaranteed(
  df: pd.DataFrame,
  f_calc: Callable[[pd.DataFrame], pd.DataFrame],
  group_size_minutes: int,
):
  """
  Calculate features with the last value guaranteed.

  This function trades off speed for coverage. Apart from the last value, the other values can be nans.
  It will be useful for cases when only the last value is needed.
  Example: Streaming, find_min_data_points_needed_for_stable_good_last_value

  For complete coverage, use calculate_features_with_complete_coverage.
  """
  # calculate last value's offset
  tbubg_result_ = bg_tbu.get_time_based_uniform_bar_group_for_indian_market(
    df=df,
    group_size_minutes=group_size_minutes,
    offset_minutes=0,
  )
  last_offset = tbubg_result_.offsets.values[-1]

  # pass offset as last value's offset
  # this ensure last will be the first bar of the resampled df
  # => value will be calculated for it
  tbubg_result = bg_tbu.get_time_based_uniform_bar_group_for_indian_market(
    df=df,
    group_size_minutes=group_size_minutes,
    offset_minutes=last_offset,
  )
  with pd.option_context("mode.chained_assignment", None):
    df["bar_group"] = tbubg_result.bar_groups
    df["offset"] = tbubg_result.offsets
    result = u_df.calculate_grouped_values(df=df, compute_func=f_calc)
    df.drop(columns=["bar_group", "offset"], inplace=True)

  return result


def calculate_features_with_complete_coverage(
  df: pd.DataFrame,
  f_calc: Callable[[pd.DataFrame], pd.DataFrame]
  | list[Callable[[pd.DataFrame], pd.DataFrame]],
  group_size_minutes: int,
):
  """
  Resample df to group_size_minutes and calculate feature values.
  Doing this naively will result in nans littered throughout the df.
  This function avoids this by computing all possible resamplings of the df,
  applying all supplied f_calcs on each and stitching the results so that
  each original row receives a value.

  NaNs will appear at the start of the df only.
  It converts good values before any nan to nans.
  """
  results = []
  all_offsets = []

  for offset_minutes in range(group_size_minutes):
    tbubg_result = bg_tbu.get_time_based_uniform_bar_group_for_indian_market(
      df=df,
      group_size_minutes=group_size_minutes,
      offset_minutes=offset_minutes,
    )
    with pd.option_context("mode.chained_assignment", None):
      df["bar_group"] = tbubg_result.bar_groups
      df["offset"] = tbubg_result.offsets
      results.append(u_df.calculate_grouped_values(df=df, compute_func=f_calc))
      all_offsets.append(tbubg_result.offsets.values)
      df.drop(columns=["bar_group", "offset"], inplace=True)

  # Stack all offset arrays into a 2D array (rows=offset_minutes sent as input , cols=offsets returned as output)
  offsets_array = np.stack(all_offsets, axis=0)

  # Find which iteration has offset=0 for each row
  zero_offset_mask = offsets_array == 0
  iteration_indices = np.argmax(zero_offset_mask, axis=0)

  # Stack all result dataframes into a 3D array (offset_minutes sent as input, rows, columns)
  result_values = np.stack([result.values for result in results], axis=0)

  # Use indexing to select the correct values
  # For each row, select from the iteration that has offset=0 for that row
  row_indices = np.arange(len(df))
  final_values = result_values[iteration_indices, row_indices, :]

  # Create the final dataframe
  output_df = pd.DataFrame(final_values, index=df.index, columns=results[0].columns)

  # ensure all columns are of the form nan,nan,nan,3,4,5
  # (i.e. nans appear at the start only and then all values are good)
  # nan,2,nan,3,4,5 will be converted to nan,nan,nan,3,4,5
  # (good values before any nan are converted to nans)
  # THIS MAKES THE CODE STREAM UNSAFE SINCE A PRESENT GOOD VALUE
  # CAN BE CONVERTED TO NAN IF A FUTURE VALUE IS NAN.
  # A forward fill will make the code stream safe but will replace nan
  # values with good values which can hide bugs. So we will not do this.

  quals = u_df.analyse_numeric_columns_quality(df=output_df)
  for col, qual in quals.items():
    reversed_mask = qual.good_values_mask[::-1]
    false_indices = np.where(~reversed_mask)[0]
    if len(false_indices) > 0:
      first_false_from_end_idx = len(reversed_mask) - 1 - false_indices[0]
      first_false_from_end = qual.good_values_mask.index[first_false_from_end_idx]
      output_df.loc[:first_false_from_end, col] = np.nan

  return output_df


def find_min_data_points_needed_for_stable_good_last_value(
  df: pd.DataFrame,
  f_calc: Callable[[pd.DataFrame], pd.DataFrame],
) -> int:
  """
  Find the minimum number of data points needed for a stable good value for the last row.

  For certain features (e.g., those involving an exponential moving average), calculating a value with (say) 10 rows
  can differ from calculating a value using (say) 100 rows (for the same timestamp). Values can be good (i.e. non NaN, inf, None etc.) in both cases
  but not match for the same timestamp.

  This function uses binary search to find the minimum number of data points needed such that the last value is a stable good value.

  Concrete example:
  say calling feature_A() on a dataframe df with 100 rows yields first 5 values bad (NaN, inf, None etc.) and last 95 values good (6th is the first good value).
  it is possible that some of the initial seemingly good looking values out of all 95 are bad.
  i.e. if we call feature_A() on df.iloc[1:] and reindex the result to df.index, ideally we should get first 6 values bad, all good values from 7th onwards
  AND the 7th value to match the 7th value of the original result dataframe.
  In case of some features, it is possible that this is not the case. So we should discard the 7th value as well in the original result dataframe.
  But how many values to discard?
  This function can be used to answer this question.
  It binary searches for the smallest number of datapoints that are needed for a stable last value.

  So if this function returns 12 for feature_A(), we should discard the first 11 values from the original result dataframe (set them to NaN) since
  the 12th value is the first stable good value.

  NOTE: For the same feature_A, this function can return different results for different slices of a df since it depends on the actual values of the dataframe.
  However, the values should be close to each other and not differ wildly (i.e. getting 100 for one slice and 1200 for another slice (of the same df and for the same feature_A)
  is not expected).
  If you want to be extra cautious, use a safety factor to increase the number of values to discard (i.e. safety factor of 1.05 means discard 105 values if this function returns 100).

  """
  last_row = f_calc(df).iloc[-1]
  if not np.isfinite(last_row).all():
    raise ValueError("Last row should have all finite values")
  if np.isclose(f_calc(df.iloc[(len(df) - 1) :]).iloc[-1], last_row).all():
    return 1
  low, high = 0, len(df) - 1
  # value matches at low and does not match at high
  while low < high - 1:
    mid = (low + high) // 2
    if np.isclose(f_calc(df.iloc[mid:]).iloc[-1], last_row).all():
      low = mid
    else:
      high = mid
  return len(df) - low
