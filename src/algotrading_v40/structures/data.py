import pandas as pd


class Data:
  def __init__(self, symbol_to_df_list: dict[str, list[pd.DataFrame]]):
    assert isinstance(symbol_to_df_list, dict)
    assert all(isinstance(df_list, list) for df_list in symbol_to_df_list.values())
    assert all(
      all(isinstance(df, pd.DataFrame) for df in df_list)
      for df_list in symbol_to_df_list.values()
    )
    self._symbol_to_df_list = symbol_to_df_list

  @classmethod
  def create_from_symbol_to_df(cls, symbol_to_df: dict[str, pd.DataFrame]) -> "Data":
    return cls({symbol: [df] for symbol, df in symbol_to_df.items()})

  def available_symbols(self) -> list[str]:
    return list(self._symbol_to_df_list.keys())

  def for_symbol(self, symbol: str) -> pd.DataFrame:
    df_list = self._symbol_to_df_list[symbol]
    df = pd.concat(df_list)
    df.sort_index(inplace=True)
    assert df.index.is_unique
    assert df.index.is_monotonic_increasing
    return df
