import pandas as pd

import algotrading_v40.utils.df as udf


def validate_and_run_cusum(
  s: pd.Series,  # series to select events on
  h: float,  # cusum threshold
) -> pd.Series:
  if udf.analyse_numeric_series_quality(s).n_bad_values > 0:
    raise ValueError("s must not have bad values")

  if h <= 0:
    raise ValueError("cusum threshold must be greater than 0")
  s_diff = s.diff()
  ans, s_pos, s_neg = [0], 0, 0
  for i in s_diff.index[1:]:
    s_pos, s_neg = max(0, s_pos + s_diff.loc[i]), min(0, s_neg + s_diff.loc[i])
    if s_neg < -h:
      s_neg = 0
      ans.append(-1)
    elif s_pos > h:
      s_pos = 0
      ans.append(1)
    else:
      ans.append(0)
  return pd.Series(ans, index=s.index).astype("int32")
