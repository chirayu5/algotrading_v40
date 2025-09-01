import numpy as np
import pandas as pd
from backtesting import Backtest
from backtesting.lib import Strategy


def equity(p: float, cash: float, trades: list[dict[str, float]]) -> float:
  return cash + sum(t["size"] * (p - t["entry"]) for t in trades)


def margin_used(p: float, trades: list[dict[str, float]]) -> float:  # leverage = 1
  return sum(abs(t["size"]) * p for t in trades)


def check_trades(trades: list[dict[str, float]]) -> None:
  """
  Check all trades are non-zero and have the same sign.
  """
  for tr in trades:
    if tr["size"] == 0:
      raise ValueError(f"Trade with zero size found: {tr}")
  if trades:
    first_sign = np.sign(trades[0]["size"])
    for tr in trades:
      if np.sign(tr["size"]) != first_sign:
        raise ValueError(f"Trades have mixed signs: {trades}")


def compute_backtesting_return(
  df: pd.DataFrame,
  commission_rate,
  initial_cash,
  error_on_order_rejection: bool,
) -> tuple[float, float]:
  close = df["Close"].to_numpy(float)
  target = df["final_ba_position"].to_numpy(int)
  if target[-1] != 0:
    raise ValueError("Final target position must be 0")
  cash: float = initial_cash
  trades: list[dict[str, float]] = []  # FIFO queue – {'size', 'entry'}

  for i in range(0, len(df)):  # bar executing fills
    check_trades(trades)
    px = close[i]
    want = target[i]  # desired position at the end of the bar
    pos = sum(t["size"] for t in trades)
    delta = want - pos  # change required this bar

    # stop if the account is out of money
    if equity(p=px, cash=cash, trades=trades) <= 0:
      # close everything at current price, zero-out cash and stop
      cash = 0.0
      trades.clear()
      break

    if delta == 0:
      continue

    # --------------------------------------------------------------
    # 1) Reduce / close opposite-facing trades (FIFO)
    # --------------------------------------------------------------
    j = 0
    while j < len(trades) and delta:
      if trades[j]["size"] * delta >= 0:
        break
      tr = trades[j]
      qty = np.sign(tr["size"]) * min(abs(tr["size"]), abs(delta))
      cash += qty * (px - tr["entry"])  # realised P/L
      cash -= abs(qty) * px * commission_rate  # exit commission
      tr["size"] -= qty
      delta += qty  # smaller (or 0)
      if tr["size"] == 0:
        j += 1

    trades = trades[j:]  # only keep trades with size not 0
    check_trades(trades)
    if delta == 0:
      continue
    # --------------------------------------------------------------
    # 2) Open *if and only if* broker can afford the whole remainder
    # --------------------------------------------------------------
    need = abs(delta)

    # affordability criterion:
    cost_total = need * (px + px * commission_rate)

    avail = max(
      0.0, equity(p=px, cash=cash, trades=trades) - margin_used(p=px, trades=trades)
    )  # free equity
    if cost_total <= avail:  # broker accepts order
      signed = int(np.sign(delta)) * need
      trades.append({"size": signed, "entry": px})
      cash -= need * px * commission_rate  # entry commission
    else:
      if error_on_order_rejection:
        raise ValueError(
          f"Order rejected: index={i}, timestamp={df.index[i]}, cost_total={cost_total}, avail={avail}"
        )
      # else: broker silently cancels
    check_trades(trades)
  if trades:
    raise ValueError("There are still open trades at the end")
  # ------------------- final valuation of open trades -------------------
  equity_final = equity(p=close[-1], cash=cash, trades=trades)
  return_pct = 100 * (equity_final - initial_cash) / initial_cash
  return equity_final, return_pct


##########


class PositionStrategy(Strategy):
  def init(self):
    super().init()

  def next(self):
    desired = int(self.data.final_ba_position[-1])
    current = self.position.size
    delta = desired - current
    if delta > 0:
      self.buy(size=delta)
    elif delta < 0:
      self.sell(size=-delta)


def compute_backtesting_return_reference(
  df: pd.DataFrame,
  commission_rate: float,
  initial_cash: float,
) -> pd.Series:
  df = df.copy()
  if df["final_ba_position"].iloc[-1] != 0:
    raise ValueError("Final target position must be 0")

  # backtesting library ignores first and last rows
  # so setting them as dummy rows
  first_row = df.iloc[0].copy()
  first_row["final_ba_position"] = 0
  first_row.name = df.index[0] - pd.Timedelta(minutes=1)

  last_row = df.iloc[-1].copy()
  last_row["final_ba_position"] = 0
  last_row.name = df.index[-1] + pd.Timedelta(minutes=1)

  df = pd.concat([pd.DataFrame([first_row]), df, pd.DataFrame([last_row])])

  bt = Backtest(
    df,
    PositionStrategy,
    commission=commission_rate,
    trade_on_close=True,
    hedging=False,
    exclusive_orders=False,
    finalize_trades=True,
    cash=initial_cash,
  )
  return bt.run()
