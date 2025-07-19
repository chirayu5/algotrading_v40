import enum
from dataclasses import dataclass


class Market(enum.Enum):
  INDIAN_MARKET = "indian_market"
  CRYPTO = "crypto"


@dataclass(frozen=True)
class InstrumentDesc:
  market: Market
  symbol: str


@dataclass(frozen=True, order=True)
class SpotDesc(InstrumentDesc):
  pass


@dataclass(frozen=True, order=True)
class IndexDesc(InstrumentDesc):
  pass


@dataclass(frozen=True, order=True)
class FutureDesc(InstrumentDesc):
  expiry: str | None  # None since Binance future contracts can be perpetual
