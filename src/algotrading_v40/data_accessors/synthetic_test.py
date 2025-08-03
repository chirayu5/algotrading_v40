import datetime as dt
import random

import numpy as np
import pandas as pd
import pytest
import pytz

import algotrading_v40.data_accessors.synthetic as das
import algotrading_v40.structures.data as sd
import algotrading_v40.structures.date_range as sdr
import algotrading_v40.structures.instrument_desc as sid
import algotrading_v40.utils.df as udf


def set_seed(seed: int):
  random.seed(seed)
  np.random.seed(seed)


class TestCreateTradingDatetimeIndex:
  def test_single_weekday(self):
    start_date = dt.date(2024, 1, 2)
    end_date = dt.date(2024, 1, 2)

    result = das.create_trading_datetime_index(start_date, end_date)
    first_time: pd.Timestamp = result[0]  # type: ignore
    last_time: pd.Timestamp = result[-1]  # type: ignore

    assert result.is_unique
    assert result.is_monotonic_increasing
    assert len(result) == 375
    assert first_time == pd.Timestamp("2024-01-02 03:45:59.999", tz=pytz.UTC)
    assert last_time == pd.Timestamp("2024-01-02 09:59:59.999", tz=pytz.UTC)
    mcid = udf.get_most_common_index_delta(result)
    assert mcid.most_common_index_delta == 1
    pd.testing.assert_series_equal(
      mcid.index_delta_distribution, pd.Series([374], index=[1]), check_names=False
    )  # 375 data points will have 375-1 gaps

  def test_weekend_excluded(self):
    start_date = dt.date(2024, 1, 6)
    end_date = dt.date(2024, 1, 7)
    assert start_date.weekday() == 5
    assert end_date.weekday() == 6
    result = das.create_trading_datetime_index(start_date, end_date)
    assert result.is_unique
    assert result.is_monotonic_increasing
    assert len(result) == 0

  def test_mixed_week(self):
    start_date = dt.date(2024, 1, 1)  # monday
    end_date = dt.date(2024, 1, 7)  # sunday

    result = das.create_trading_datetime_index(start_date, end_date)
    assert result.is_unique
    assert result.is_monotonic_increasing
    assert isinstance(result, pd.DatetimeIndex)
    assert result.tz == pytz.UTC
    assert len(result) == 375 * 5
    assert result[0].date() == start_date  # type: ignore
    assert result[-1].date() == dt.date(2024, 1, 5)  # type: ignore
    assert udf.get_most_common_index_delta(result).most_common_index_delta == 1

  def test_end_before_start(self):
    start_date = dt.date(2024, 1, 5)
    end_date = dt.date(2024, 1, 3)

    result = das.create_trading_datetime_index(start_date, end_date)
    assert result.is_unique
    assert result.is_monotonic_increasing
    assert len(result) == 0

  def test_timezone_conversion(self):
    start_date = dt.date(2024, 1, 2)
    end_date = dt.date(2024, 1, 2)

    result = das.create_trading_datetime_index(start_date, end_date)
    assert result.is_unique
    assert result.is_monotonic_increasing
    first_time: pd.Timestamp = result[0]  # type: ignore
    assert first_time.tz == pytz.UTC

    ist_time = first_time.astimezone(pytz.timezone("Asia/Kolkata"))
    assert ist_time.hour == 9
    assert ist_time.minute == 15
    assert ist_time.second == 59
    assert ist_time.microsecond == 999000

    last_time: pd.Timestamp = result[-1]  # type: ignore
    assert last_time.tz == pytz.UTC

    ist_time = last_time.astimezone(pytz.timezone("Asia/Kolkata"))
    assert ist_time.hour == 15
    assert ist_time.minute == 29
    assert ist_time.second == 59
    assert ist_time.microsecond == 999000


class TestGBMEngine:
  def test_consistency(self):
    """
    Generate synthetic data with known parameters and check that fitting the data
    yields parameters that are close to the true parameters.
    """
    set_seed(42)
    index_synth = das.create_trading_datetime_index(
      dt.date(2024, 1, 1), dt.date(2034, 1, 1)
    )

    # mu < 0
    gbm_params = das.GBMParams(S0=100, mu=-7, sigma=0.2, dt=1 / (365 * 24 * 60))
    prices_synth = das.simulate_gbm_path(gbm_params, len(index_synth))
    ps = pd.Series(prices_synth, index=index_synth)
    assert ps.index.is_unique
    assert ps.index.is_monotonic_increasing
    gbm_params_synth = das.fit_gbm(ps)
    assert np.isclose(gbm_params_synth.mu, -7.216493651775861)
    assert np.isclose(gbm_params_synth.sigma, 0.20006683409737558)
    assert np.isclose(gbm_params_synth.S0, 100)
    assert np.isclose(gbm_params_synth.dt, 1 / (365 * 24 * 60))

    # mu > 0
    gbm_params = das.GBMParams(S0=100, mu=5, sigma=0.2, dt=1 / (365 * 24 * 60))
    prices_synth = das.simulate_gbm_path(gbm_params, len(index_synth))
    ps = pd.Series(prices_synth, index=index_synth)
    assert ps.index.is_unique
    assert ps.index.is_monotonic_increasing
    gbm_params_synth = das.fit_gbm(ps)
    assert np.isclose(gbm_params_synth.mu, 4.9504780792693515)
    assert np.isclose(gbm_params_synth.sigma, 0.20022561579168582)
    assert np.isclose(gbm_params_synth.S0, 100)
    assert np.isclose(gbm_params_synth.dt, 1 / (365 * 24 * 60))

  def test_sampling_invariance(self):
    """
    Check that even if we sample the GBM at a different frequency, the parameters
    are still close to the true parameters. This should happen because we compute the
    annualized version of the parameters.
    """
    set_seed(42)
    index_synth = das.create_trading_datetime_index(
      dt.date(2024, 1, 1), dt.date(2034, 1, 1)
    )

    # mu < 0
    gbm_params = das.GBMParams(S0=100, mu=-7, sigma=0.2, dt=1 / (365 * 24 * 60))
    prices_synth = das.simulate_gbm_path(gbm_params, len(index_synth))
    ps = pd.Series(prices_synth, index=index_synth)
    ps_17min = ps[::17]
    assert ps_17min.index.is_unique  # type: ignore
    assert ps_17min.index.is_monotonic_increasing  # type: ignore
    assert np.isclose(len(ps) / len(ps_17min), 17, atol=1e-2)
    gbm_params_synth = das.fit_gbm(ps_17min)  # type: ignore
    assert np.isclose(gbm_params_synth.mu, -7.216021693375584)
    assert np.isclose(gbm_params_synth.sigma, 0.2001085469048804)
    assert np.isclose(gbm_params_synth.S0, 100)
    assert np.isclose(gbm_params_synth.dt, 17 / (365 * 24 * 60))

    # mu > 0
    gbm_params = das.GBMParams(S0=100, mu=5, sigma=0.2, dt=1 / (365 * 24 * 60))
    prices_synth = das.simulate_gbm_path(gbm_params, len(index_synth))
    ps = pd.Series(prices_synth, index=index_synth)
    ps_23min = ps[::23]
    assert ps_23min.index.is_unique  # type: ignore
    assert ps_23min.index.is_monotonic_increasing  # type: ignore
    assert np.isclose(len(ps) / len(ps_23min), 23, atol=1e-2)
    gbm_params_synth = das.fit_gbm(ps_23min)  # type: ignore
    assert np.isclose(gbm_params_synth.mu, 4.949855106666751)
    assert np.isclose(gbm_params_synth.sigma, 0.2000650290117161)
    assert np.isclose(gbm_params_synth.S0, 100)
    assert np.isclose(gbm_params_synth.dt, 23 / (365 * 24 * 60))

  def test_ohlc_from_path(self):
    """Test that ohlc_from_path correctly converts a price path to OHLC data."""
    # Test with a simple ascending path
    path = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    ohlc = das.ohlc_from_path(path, bucket_size=5)
    expected_ohlc = pd.DataFrame(
      {"open": [10, 15], "high": [14, 19], "low": [10, 15], "close": [14, 19]}
    )
    pd.testing.assert_frame_equal(ohlc, expected_ohlc)

    # Test with a more complex path
    path = np.array([100, 105, 95, 110, 90, 120, 85, 115, 125, 100])
    ohlc = das.ohlc_from_path(path, bucket_size=5)
    expected_ohlc = pd.DataFrame(
      {"open": [100, 120], "high": [110, 125], "low": [90, 85], "close": [90, 100]}
    )
    pd.testing.assert_frame_equal(ohlc, expected_ohlc)

    # Test with bucket_size = 1
    path = np.array([50, 60, 70])
    ohlc = das.ohlc_from_path(path, bucket_size=1)
    expected_ohlc = pd.DataFrame(
      {
        "open": [50, 60, 70],
        "high": [50, 60, 70],
        "low": [50, 60, 70],
        "close": [50, 60, 70],
      }
    )
    pd.testing.assert_frame_equal(ohlc, expected_ohlc)

    # Test assertion error for non-divisible path length
    path = np.array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
      das.ohlc_from_path(path, bucket_size=3)

    # Test empty path
    path = np.array([])
    ohlc = das.ohlc_from_path(path, bucket_size=3)
    expected_ohlc = pd.DataFrame(
      {
        "open": [],
        "high": [],
        "low": [],
        "close": [],
      }
    )
    pd.testing.assert_frame_equal(ohlc, expected_ohlc)


class TestGetSyntheticData:
  def test_default_config(self):
    """Default‐config path for get_synthetic_data."""
    symbols = ["PGHH", "COLPAL", "NESTLEIND"]
    instrument_descs = [sid.EquityDesc(sid.Market.INDIAN_MARKET, s) for s in symbols]

    start_date = dt.date(2023, 1, 2)  # Monday
    end_date = dt.date(2023, 1, 9)  # Monday of next week
    date_range = sdr.DateRange(start_date, end_date)

    data = das.get_synthetic_data(instrument_descs, date_range)
    assert isinstance(data, sd.Data)
    assert {d.symbol for d in data.available_instrument_descs()} == set(symbols)
    for inst in instrument_descs:
      df = data.get_full_df_for_instrument_desc(inst)
      assert df.index.is_unique
      assert df.index.is_monotonic_increasing
      assert df.index.name == "bar_close_timestamp"
      assert isinstance(df, pd.DataFrame)
      assert list(df.columns) == ["open", "high", "low", "close"]
      assert isinstance(df.index, pd.DatetimeIndex)
      assert df.index.tz == pytz.UTC
      assert (df["high"] >= df["open"]).all()
      assert (df["high"] >= df["close"]).all()
      assert (df["low"] <= df["open"]).all()
      assert (df["low"] <= df["close"]).all()
      assert (df["high"] >= df["low"]).all()
      assert not df.isnull().any().any()  # type: ignore
      assert not df.isna().any().any()  # type: ignore
      assert np.isfinite(df).all().all()
      assert (df > 0).all().all()
      assert (
        len(df) == 375 * 6
      )  # 6 days, 375 minutes per day (saturday and sunday are excluded); drop_fraction is None, so no data is dropped
      assert df.index[0] == pd.Timestamp("2023-01-02 03:45:59.999", tz=pytz.UTC)
      assert df.index[-1] == pd.Timestamp("2023-01-09 09:59:59.999", tz=pytz.UTC)
      mcid = udf.get_most_common_index_delta(df.index)
      assert mcid.most_common_index_delta == 1
      assert (
        len(mcid.index_delta_distribution) == 3
      )  # there will be three values, one for the intraday gap of one minute, one for the overnight gap and one for the weekend gap
      assert (
        mcid.index_delta_distribution.index[0] == 1
      )  # most common should be the intraday gap of one minute
      assert (
        mcid.index_delta_distribution.index[1] == 1066  # overnight gap
      )  # the overnight gap is 1066 minutes (17 hours 46 minutes)
      assert mcid.index_delta_distribution.index[2] == 1066 + (
        2 * 24 * 60
      )  # the weekend gap is 1066 minutes (17 hours 46 minutes) + 2 days (48 hours)

    single_inst = sid.EquityDesc(sid.Market.INDIAN_MARKET, "SINGLE")
    single_data = das.get_synthetic_data([single_inst], date_range)
    assert len(single_data.available_instrument_descs()) == 1
    assert single_data.available_instrument_descs()[0] == single_inst

    empty_data = das.get_synthetic_data([], date_range)
    assert len(empty_data.available_instrument_descs()) == 0

  def test_custom_config(self):
    """Custom‐config path for get_synthetic_data."""

    # test that bad configs raise errors
    with pytest.raises(ValueError):
      das.SyntheticDataConfig(
        drop_fraction=1.0,
        gbm_params=das.GBMParams(S0=100, mu=0, sigma=0.2, dt=1 / (365 * 24 * 60)),
      )
    with pytest.raises(ValueError):
      das.SyntheticDataConfig(
        drop_fraction=-0.1,
        gbm_params=das.GBMParams(S0=100, mu=0, sigma=0.2, dt=1 / (365 * 24 * 60)),
      )
    with pytest.raises(ValueError):
      das.SyntheticDataConfig(
        drop_fraction=0.5,
        gbm_params=das.GBMParams(S0=100, mu=0, sigma=0.2, dt=60 / (365 * 24 * 60)),
      )

    inst_PGHH = sid.EquityDesc(sid.Market.INDIAN_MARKET, "PGHH")
    inst_COLPAL = sid.EquityDesc(sid.Market.INDIAN_MARKET, "COLPAL")
    inst_and_cfg = [
      (
        inst_PGHH,
        das.SyntheticDataConfig(
          drop_fraction=0.8,
          gbm_params=das.GBMParams(S0=123, mu=5, sigma=0.2, dt=1 / (365 * 24 * 60)),
        ),
      ),
      (
        inst_COLPAL,
        das.SyntheticDataConfig(
          drop_fraction=None,
          gbm_params=das.GBMParams(S0=244, mu=-5, sigma=0.7, dt=1 / (365 * 24 * 60)),
        ),
      ),
    ]
    start_date = dt.date(2023, 1, 2)  # Monday
    end_date = dt.date(2023, 1, 3)  # Tuesday
    date_range = sdr.DateRange(start_date, end_date)
    data = das.get_synthetic_data(inst_and_cfg, date_range)

    df_COLPAL = data.get_full_df_for_instrument_desc(inst_COLPAL)
    assert df_COLPAL.index.name == "bar_close_timestamp"
    assert df_COLPAL.index.is_unique
    assert df_COLPAL.index.is_monotonic_increasing
    assert len(df_COLPAL) == 375 * 2, (
      "COLPAL should have 2 days of complete data since drop_fraction is None"
    )
    assert df_COLPAL["open"].iloc[0] == 244

    df_PGHH = data.get_full_df_for_instrument_desc(inst_PGHH)
    assert df_PGHH.index.is_unique
    assert df_PGHH.index.is_monotonic_increasing
    assert len(df_PGHH) == 150, (
      "PGHH should have 20% of the data (i.e. 375 * 2 * (1 - 0.8) = 150) since drop_fraction is 0.8"
    )
    # df_PGHH["open"].iloc[0] is not guaranteed to be 123 since we drop data
    # so that is not tested here
