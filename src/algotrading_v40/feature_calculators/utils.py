from typing import Callable

import numpy as np
import pandas as pd

import algotrading_v40.bar_groupers.time_based_uniform as bg_tbu
import algotrading_v40.utils.df as u_df


def calculate_features_with_complete_coverage(
  df: pd.DataFrame,
  f_calc: Callable[[pd.DataFrame], pd.DataFrame],
  group_size_minutes: int,
):
  """
  Resample df to group_size_minutes and calculate feature values.
  Doing this naively will result in nans littered throughout the df.
  This function avoids this by computing all possible resamplings of the df,
  applying f_calc on each and stitching the results so that each original row receives a value.

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
