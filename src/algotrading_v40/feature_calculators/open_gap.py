import pandas as pd

import algotrading_v40.utils.features as uf


def open_gap_indian_market(
  df: pd.DataFrame,
  lag: int,
) -> pd.DataFrame:
  if lag < 0:
    raise ValueError("lag must be >= 0")
  df = df.copy()
  df_session_info_indian_market = uf.get_indian_market_session_info(index=df.index)
  df[["session_date", "session_id"]] = df_session_info_indian_market[
    ["session_date", "session_id"]
  ]
  df_session = (
    df.groupby("session_id")
    .agg({"close": "last", "open": "first"})
    .rename(columns={"close": "session_close_price", "open": "session_open_price"})
    .reset_index()
  )
  df_session.sort_values(by="session_id")

  df_session["open_gap"] = (
    df_session["session_open_price"] - df_session["session_close_price"].shift(1)
  ) / df_session["session_close_price"].shift(1)

  result = df[["session_id"]].copy()  # to avoid SettingWithCopyWarning
  result["session_id"] = result["session_id"] - lag
  result = result.merge(df_session, on="session_id", how="left")
  result.rename(columns={"open_gap": f"open_gap_{lag}"}, inplace=True)
  result.index = df.index
  return result[[f"open_gap_{lag}"]]
