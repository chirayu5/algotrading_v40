import algotrading_v40_cpp.data_selectors as av40c_ds
import matplotlib.pyplot as plt
import pandas as pd

import algotrading_v40.utils.df as udf


def plot_cusum_result(s: pd.Series, cusum: pd.Series, sz=15, figsize=(12, 6), dpi=100):
  if not isinstance(s.index, pd.DatetimeIndex):
    raise ValueError("s.index must be a DatetimeIndex")
  if not s.index.equals(cusum.index):
    raise ValueError("s.index and cusum.index must be the same")

  s = s.copy()
  cusum = cusum.copy()

  s = s.reset_index(drop=True)
  cusum = cusum.reset_index(drop=True)

  plt.figure(figsize=figsize, dpi=dpi)
  plt.plot(s.index, s.values, "b-", linewidth=1, label="Series")

  green_points = s[cusum == 1]
  red_points = s[cusum == -1]

  if len(green_points) > 0:
    plt.scatter(
      green_points.index,
      green_points.values,
      color="green",
      s=sz,
      zorder=5,
      label="CUSUM +1",
    )

  if len(red_points) > 0:
    plt.scatter(
      red_points.index,
      red_points.values,
      color="red",
      s=sz,
      zorder=5,
      label="CUSUM -1",
    )

  plt.xlabel("Index")
  plt.ylabel("s")
  plt.title("CUSUM Event Detection")
  plt.legend()
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.show()


def cusum(
  *,
  s: pd.Series,  # series to select events on
  thresholds: pd.Series,  # cusum threshold
) -> pd.DataFrame:
  if udf.analyse_numeric_series_quality(s).n_bad_values > 0:
    raise ValueError("s must not have bad values")

  if udf.analyse_numeric_series_quality(thresholds).n_bad_values > 0:
    raise ValueError("thresholds must not have bad values")

  if thresholds.min() <= 0:
    raise ValueError("cusum threshold must be greater than 0")
  # all thresholds are > 0
  s_diff = s.diff()
  # s_diff[0] would be NaN but that is not used in av40c_ds.cusum_cpp.
  # so it is safe to ignore it.
  return pd.DataFrame(
    data=av40c_ds.cusum_cpp(s_diff.to_numpy(), thresholds.to_numpy()),
    index=s.index,
    columns=["selected"],
  )
