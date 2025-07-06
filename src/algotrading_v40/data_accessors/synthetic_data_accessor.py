import dataclasses
import datetime as dt

import numpy as np
import pandas as pd
import pytz

import algotrading_v40.structures.data as sd


def create_trading_datetime_index(
  start_date: dt.date, end_date: dt.date
) -> pd.DatetimeIndex:
  """Create a datetime index for weekdays between start_date and end_date in the Indian market hours.
  Timezone is UTC.
  Frequency is 1 minute.
  """
  ist = pytz.timezone("Asia/Kolkata")
  utc = pytz.UTC

  trading_start_time = dt.time(9, 15, 59, 999000)
  trading_end_time = dt.time(15, 29, 59, 999000)

  all_dates = []

  current_date: dt.date = start_date
  while current_date <= end_date:
    if current_date.weekday() < 5:
      trading_start = ist.localize(
        dt.datetime.combine(current_date, trading_start_time)
      )
      trading_end = ist.localize(dt.datetime.combine(current_date, trading_end_time))

      trading_start_utc = trading_start.astimezone(utc)
      trading_end_utc = trading_end.astimezone(utc)

      day_times = pd.date_range(
        start=trading_start_utc, end=trading_end_utc, freq="1min", tz=utc
      )
      all_dates.extend(day_times)

    current_date += dt.timedelta(days=1)

  return pd.DatetimeIndex(all_dates, tz=utc)


@dataclasses.dataclass(frozen=True)
class GMCIDResult:
  most_common_index_delta: int | None  # in minutes
  index_delta_distribution: pd.Series


def get_most_common_index_delta(index: pd.DatetimeIndex) -> GMCIDResult:
  """Get the most common index delta in minutes."""
  assert isinstance(index, pd.DatetimeIndex), (
    f"index must be a pd.DatetimeIndex, got {type(index)}"
  )
  if len(index) <= 1:
    return GMCIDResult(
      most_common_index_delta=None,
      index_delta_distribution=pd.Series([]),
    )
  index_deltas = ((index[1:] - index[:-1]).total_seconds() // 60).astype(int)
  vcs = index_deltas.value_counts(dropna=True).sort_values(ascending=False)
  return GMCIDResult(
    most_common_index_delta=int(vcs.index[0]),
    index_delta_distribution=vcs,
  )


@dataclasses.dataclass(frozen=True)
class GBMParams:
  """
  dS(t) = mu * S(t) * dt + sigma * S(t) * dW(t)
  OR
  dlog(S(t)) = (mu - 0.5 * sigma^2) * dt + sigma * dW(t)
  """

  S0: float
  mu: float
  sigma: float
  dt: float  # in years


def fit_gbm(prices: pd.Series) -> GBMParams:
  assert isinstance(prices.index, pd.DatetimeIndex), (
    f"index must be a pd.DatetimeIndex, got {type(prices.index)}"
  )
  index_delta = get_most_common_index_delta(prices.index).most_common_index_delta
  assert index_delta is not None
  dt = index_delta / (365 * 24 * 60)
  dlp = np.log(prices) - np.log(prices.shift(1))
  sigma = dlp.std() / np.sqrt(
    dt
  )  # std() divides by N-1 not N since ddof is 1 by default
  mu = (dlp.mean() / dt) + 0.5 * (sigma**2)
  return GBMParams(
    S0=float(prices.iloc[0]),
    mu=float(mu),
    sigma=float(sigma),
    dt=float(dt),
  )


def ohlc_from_path(path: np.ndarray, bucket_size: int) -> pd.DataFrame:
  assert len(path) % bucket_size == 0, (
    f"path length must be divisible by bucket_size, got {len(path)} and {bucket_size}"
  )
  price_path_reshaped = path.reshape(len(path) // bucket_size, bucket_size)
  return pd.DataFrame(
    {
      "open": price_path_reshaped[:, 0],
      "high": price_path_reshaped.max(axis=1),
      "low": price_path_reshaped.min(axis=1),
      "close": price_path_reshaped[:, -1],
    }
  )


def simulate_gbm_path(gbm_params: GBMParams, N: int) -> np.ndarray:
  """Simulate a GBM path of length N."""
  dW = np.random.normal(0, np.sqrt(gbm_params.dt), N)
  drift = (gbm_params.mu - 0.5 * gbm_params.sigma**2) * gbm_params.dt
  diffusion = gbm_params.sigma * dW
  log_returns = drift + diffusion
  price_path = np.exp(np.cumsum(log_returns))
  price_path = gbm_params.S0 * price_path / price_path[0]  # should start at S0
  return price_path


def simulate_gbm_ohlc(gbm_params: GBMParams, N: int) -> pd.DataFrame:
  price_path = simulate_gbm_path(gbm_params, N=10 * N)
  # we multiply by 10 to get more data points
  # then we bucket 10 of them to get one OHLC
  return ohlc_from_path(price_path, bucket_size=10)


@dataclasses.dataclass(frozen=True)
class SyntheticIndianEquityDataConfig:
  drop_fraction: float | None
  gbm_params: GBMParams

  def __post_init__(self):
    assert self.drop_fraction is None or (0 < self.drop_fraction < 0.9), (
      f"drop_fraction must be None or between 0 and 0.9, got {self.drop_fraction}"
    )
    assert np.isclose(self.gbm_params.dt, 1 / (365 * 24 * 60)), (
      f"dt must be 1 minute, got {self.gbm_params.dt * 365 * 24 * 60} minutes"
    )


class SyntheticIndianEquityDataAccessor:
  def __init__(self, symbols: list[str] | dict[str, SyntheticIndianEquityDataConfig]):
    if isinstance(symbols, list):
      default_gbm_params = GBMParams(
        S0=np.random.uniform(10, 1000),
        mu=np.random.uniform(-6, 6),
        sigma=np.random.uniform(0.1, 0.6),
        dt=1
        / (
          365 * 24 * 60
        ),  # create_trading_datetime_index always returns 1 minute intervals
      )
      self.symbols = {
        symbol: SyntheticIndianEquityDataConfig(
          drop_fraction=None,  # no drop by default
          gbm_params=default_gbm_params,
        )
        for symbol in symbols
      }
    else:
      self.symbols = symbols

  def get(self, start_date: dt.date, end_date: dt.date) -> sd.Data:
    symbol_to_df = {}
    for symbol, config in self.symbols.items():
      symbol_to_df[symbol] = self._get_ohlc(start_date, end_date, config)
    return sd.Data.create_from_symbol_to_df(symbol_to_df)

  def _get_ohlc(
    self,
    start_date: dt.date,
    end_date: dt.date,
    config: SyntheticIndianEquityDataConfig,
  ) -> pd.DataFrame:
    index = create_trading_datetime_index(start_date, end_date)
    df_ohlc = simulate_gbm_ohlc(
      gbm_params=config.gbm_params,
      N=len(index),
    )
    df_ohlc.index = index
    if config.drop_fraction is not None:
      df_ohlc = df_ohlc.sample(frac=1 - config.drop_fraction)
      df_ohlc.sort_index(inplace=True)  # sample() shuffles the index
    return df_ohlc
