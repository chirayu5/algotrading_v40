import pandas as pd


class Data:
  def __init__(self, symbol_to_df: dict[str, pd.DataFrame]):
    self.symbol_to_df = symbol_to_df

  def available_symbols(self) -> list[str]:
    return list(self.symbol_to_df.keys())

  def for_symbol(self, symbol: str) -> pd.DataFrame:
    return self.symbol_to_df[symbol]
