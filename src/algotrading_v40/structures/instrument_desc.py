import enum
from dataclasses import dataclass


class Market(enum.Enum):
  INDIAN_MARKET = "indian_market"
  US_MARKET = "us_market"
  CRYPTO = "crypto"


@dataclass(frozen=True)
class InstrumentDesc:
  market: Market
  symbol: str


@dataclass(frozen=True, order=True)
class EquityDesc(InstrumentDesc):
  def __post_init__(self):
    if self.market == Market.CRYPTO:
      raise ValueError(
        f"EquityDesc can not be used for crypto market, got {self.market}"
      )


@dataclass(frozen=True, order=True)
class IndexDesc(InstrumentDesc):
  def __post_init__(self):
    if self.market == Market.CRYPTO:
      raise ValueError(
        f"IndexDesc can not be used for crypto market, got {self.market}"
      )


@dataclass(frozen=True, order=True)
class FutureDesc(InstrumentDesc):
  expiry: str | None  # None since Binance future contracts can be perpetual
