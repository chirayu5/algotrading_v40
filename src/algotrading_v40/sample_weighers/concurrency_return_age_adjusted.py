import numpy as np
import pandas as pd

import algotrading_v40.utils.df as u_df


def _get_concurrency(
  *,
  label_last_indices: np.ndarray,
) -> np.ndarray:
  N = len(label_last_indices)
  concurrency = np.zeros(N, dtype=float)
  # concurrency[i] is the number of labels that used the prices[i-1] to prices[i] return
  # (concurrency[0] is 0 since no label uses the prices[-1] to prices[0] return)
  for i in range(N):
    if not np.isfinite(label_last_indices[i]):
      continue

    t_in = i + 1
    # label_last_indices contains positional indices (0-based) into the arrays,
    # NOT the actual DataFrame index values
    t_out = round(label_last_indices[i])
    # to generate label[i], we used prices[i] to prices[lli[i]] returns
    # => we used prices[i],prices[i+1] | prices[i+1],prices[i+2] | ... | prices[lli[i]-1],prices[lli[i]] returns
    # => lli[i]-i returns were used
    # (lli means label_last_indices)
    if (t_out < t_in) or (t_out < 0) or (t_out >= N):
      raise ValueError(f"Invalid values for index {i}; got t_in={t_in}, t_out={t_out}")
    for t in range(t_in, t_out + 1):
      concurrency[t] += 1.0

  return concurrency


def _get_log_returns(
  *,
  prices: np.ndarray,
) -> np.ndarray:
  N = len(prices)
  log_returns = np.zeros(N)
  log_returns[0] = np.nan
  for t in range(1, N):
    log_returns[t] = np.log(prices[t]) - np.log(prices[t - 1])

  return log_returns


def _get_weights_raw(
  *,
  label_last_indices: np.ndarray,
  log_returns: np.ndarray,
  concurrency: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  N = len(label_last_indices)
  attribution_weights_raw = np.zeros(N)
  attribution_weights_raw.fill(np.nan)
  avg_uniqueness = np.zeros(N)
  avg_uniqueness.fill(np.nan)

  for i in range(N):
    if not np.isfinite(label_last_indices[i]):
      continue

    t_in = i + 1
    t_out = round(label_last_indices[i])
    # to generate label[i], we used prices[i] to prices[lli[i]] returns
    # => we used prices[i],prices[i+1] | prices[i+1],prices[i+2] | ... | prices[lli[i]-1],prices[lli[i]] returns
    # => lli[i]-i returns were used
    # (lli means label_last_indices)
    lifespan = t_out - t_in + 1
    # => lifespan is also t_out - t_in + 1 = lli[i]-i

    sum_uniqueness = 0.0
    sum_attributed_return = 0.0

    for t in range(t_in, t_out + 1):
      if concurrency[t] == 0:
        raise ValueError(f"Concurrency at index {t} is zero, which should never happen")
      inv_c = 1.0 / concurrency[t]
      sum_uniqueness += inv_c
      sum_attributed_return += log_returns[t] / concurrency[t]

    avg_uniqueness[i] = sum_uniqueness / lifespan
    attribution_weights_raw[i] = abs(sum_attributed_return)

  return attribution_weights_raw, avg_uniqueness


def _get_time_decay_factors(
  *,
  label_last_indices: np.ndarray,
  avg_uniqueness: np.ndarray,
  time_decay_c: float,
) -> np.ndarray:
  if (not np.isfinite(time_decay_c)) or time_decay_c < 0 or time_decay_c > 1:
    # time_decay_c < 0 is disabled. This mode is to discard the last time_decay_c fraction of events.
    # If you want to do such a thing, just drop those data points explicitly.
    # Refer to page 70 of Advances in Financial Machine Learning for the logic
    # in case we want to enable this mode in the future.
    raise ValueError("time_decay_c must be a finite number in [0,1]")

  N = len(label_last_indices)
  # Decay is based on cumulative uniqueness, not chronological time.
  cumulative_uniqueness = np.zeros(N)
  cumulative_uniqueness.fill(np.nan)
  running_sum = 0.0
  for i in range(N):
    if not np.isfinite(label_last_indices[i]):
      continue

    running_sum += avg_uniqueness[i]
    cumulative_uniqueness[i] = running_sum

  total_uniqueness = running_sum

  # df(x) = intercept + slope * x
  # Boundary conditions: df(total_uniqueness)=1, df(0)=c
  # => intercept = c
  # => slope = (1 - c) / total_uniqueness
  if total_uniqueness == 0:
    raise ValueError("total_uniqueness must be greater than 0")
  slope = (1.0 - time_decay_c) / total_uniqueness
  intercept = time_decay_c

  decay_factors = np.zeros(N)
  decay_factors.fill(np.nan)
  for i in range(N):
    if not np.isfinite(label_last_indices[i]):
      continue
    decay_factor = intercept + slope * cumulative_uniqueness[i]

    decay_factors[i] = decay_factor

  return decay_factors


def _concurrency_return_age_adjusted_weights(
  *,
  label_last_indices: np.ndarray,
  prices: np.ndarray,
  time_decay_c: float = 1.0,
) -> dict[str, np.ndarray]:
  N = len(label_last_indices)

  concurrency = _get_concurrency(
    label_last_indices=label_last_indices,
  )

  log_returns = _get_log_returns(prices=prices)

  attribution_weights_raw, avg_uniqueness = _get_weights_raw(
    label_last_indices=label_last_indices,
    log_returns=log_returns,
    concurrency=concurrency,
  )

  time_decay_factors = _get_time_decay_factors(
    label_last_indices=label_last_indices,
    avg_uniqueness=avg_uniqueness,
    time_decay_c=time_decay_c,
  )

  # Compute final sample weights
  sample_weight = np.zeros(N)
  sample_weight.fill(np.nan)
  for i in range(N):
    sample_weight[i] = attribution_weights_raw[i] * time_decay_factors[i]

  # The weights are not normalised here to sum to the number of samples that go in the ML model.
  # This should be done right before training the ML model downstream.
  return {
    "concurrency": concurrency,
    "attribution_weight_raw": attribution_weights_raw,
    "avg_uniqueness": avg_uniqueness,
    "time_decay_factor": time_decay_factors,
    "sample_weight": sample_weight,
  }


#################################################################################################################################


def _validate_inputs(
  *,
  df: pd.DataFrame,
  time_decay_c: float,
):
  if not (df.index.is_monotonic_increasing and df.index.is_unique):
    raise ValueError("index must be monotonic increasing and unique")

  for series, name in [
    (df["selected"], "selected"),
    (df["close"], "prices"),
  ]:
    if u_df.analyse_numeric_series_quality(series).n_bad_values != 0:
      raise ValueError(f"{name} must not have bad values")

  if not set(df["selected"].unique()).issubset({0, 1}):
    raise ValueError("selected must only contain values 0 or 1")

  if np.sum(df["selected"]) == 0:
    raise ValueError("No significant samples found")

  if np.isfinite(df["label_last_index"]).sum() == 0:
    raise ValueError("No non-NaN label_last_index found")

  lli_ns = df["label_last_index"].loc[df["selected"] == 0]
  if u_df.analyse_numeric_series_quality(lli_ns).n_bad_values != len(lli_ns):
    raise ValueError(
      "selected=0 cases must have all values bad for label_last_indices since they weren't labelled"
    )
  # all selected=0 cases have bad lli values
  # => for a good lli value, selected must be 1

  # label_last_indices can have bad values (i.e. NaN, inf etc.) due to the following reasons:
  # 1. selected=0 so no label was calculated for it
  # 2. selected=1, we used triple barrier labelling, vertical barrier >= n, and no horizontal barrier
  # was hit before reaching index n-1
  # (we can't even add a check to ensure that once we hit a (selected=1,label_last_index=bad) case, any subsequent selected=1 case
  # will also have label_last_index=bad. this is because it is possible that a subsequent selected=1 case hits a horizontal barrier
  # and so has label_last_index=good (even though the vertical barrier was out of bounds))
  # so, not checking selected=1 cases

  if not np.all(df["close"] > 0):
    raise ValueError("prices must be greater than 0")

  if (not np.isfinite(time_decay_c)) or time_decay_c < 0 or time_decay_c > 1:
    raise ValueError("time_decay_c must be a finite number in [0,1]")


def concurrency_return_age_adjusted_weights(
  df: pd.DataFrame,
  time_decay_c: float,
) -> pd.DataFrame:
  """
  Calculate sample weights based on concurrency, returns, and time decay.

  Args:
      df: DataFrame with columns:
          - close: prices (must be > 0)
          - selected: binary indicator (0 or 1)
          - label_last_index: integer positional indices (0-based) of last price used in label
            NOT the actual DataFrame index values
      time_decay_c: Time decay parameter in [0, 1].
                    1.0 = no decay, 0.0 = maximum decay

  Returns:
      DataFrame with columns:
          - concurrency: number of labels using this return
          - attribution_weight_raw: raw attribution weights
          - avg_uniqueness: average uniqueness of this label
          - time_decay_factor: time decay multiplier
          - sample_weight: final sample weight (raw * decay)

  Raises:
      ValueError: if inputs are invalid
  """
  _validate_inputs(
    df=df,
    time_decay_c=time_decay_c,
  )
  result = pd.DataFrame(
    data=_concurrency_return_age_adjusted_weights(
      label_last_indices=df["label_last_index"].values,
      prices=df["close"].values,
      time_decay_c=time_decay_c,
    ),
    index=df.index,
  )

  for col in result.columns:
    if col == "concurrency":
      continue
    if not u_df.analyse_numeric_series_quality(result[col]).good_values_mask.equals(
      u_df.analyse_numeric_series_quality(df["label_last_index"]).good_values_mask
    ):
      raise ValueError(
        f"{col} and label_last_index must have the same good values mask"
      )

  return result
