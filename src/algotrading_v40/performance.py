import numpy as np
import pandas as pd


def compute_backtesting_return(
  df: pd.DataFrame,
  commission_rate: float = 0.0005,
  initial_cash: float = 10_000.0,
) -> tuple[float, float]:
  """
  NumPy/Pandas reproduction of backtesting.py’s order-matching logic for a
  pre-defined `final_ba_position` column.

  Assumptions
    • trade_on_close=True   • hedging=False   • margin=1   • no spread
  """
  close = df["Close"].to_numpy(float)
  target = df["final_ba_position"].to_numpy(int)

  cash: float = initial_cash
  trades: list[dict[str, float]] = []  # FIFO queue – {'size', 'entry'}

  def equity(p: float) -> float:
    return cash + sum(t["size"] * (p - t["entry"]) for t in trades)

  def margin_used(p: float) -> float:  # leverage = 1
    return sum(abs(t["size"]) * p for t in trades)

  # ------------------------------------------------------------------
  # bar-0 never creates orders (matches library); first fills happen
  # on bar-2 (order submitted on bar-1, executed on bar-2 prev-close)
  # ------------------------------------------------------------------
  for i in range(2, len(df)):  # bar executing fills
    px = close[i - 1]  # execution price = prev close
    want = target[i - 1]  # desired position decided prev bar
    pos = sum(t["size"] for t in trades)
    delta = want - pos  # change required this bar
    if delta == 0:
      continue

    # --------------------------------------------------------------
    # 1) Reduce / close opposite-facing trades (FIFO)
    # --------------------------------------------------------------
    j = 0
    while j < len(trades) and delta and trades[j]["size"] * delta < 0:
      tr = trades[j]
      qty = np.sign(tr["size"]) * min(abs(tr["size"]), abs(delta))
      cash += qty * (px - tr["entry"])  # realised P/L
      cash -= abs(qty) * px * commission_rate  # exit commission
      tr["size"] -= qty
      delta += qty  # smaller (or 0)
      if tr["size"] == 0:
        trades.pop(j)
      else:
        j += 1

    if delta == 0:
      continue

    # --------------------------------------------------------------
    # 2) Open *if and only if* broker can afford the whole remainder
    # --------------------------------------------------------------
    need = abs(delta)

    # Library’s exact affordability criterion:
    # cost_total = need * (px + need * px * commission_rate)
    cost_total = need * (px + need * px * commission_rate)

    avail = max(0.0, equity(px) - margin_used(px))  # free equity
    if cost_total <= avail:  # broker accepts order
      signed = int(np.sign(delta)) * need
      trades.append({"size": signed, "entry": px})
      cash -= need * px * commission_rate  # entry commission
    # else: broker silently cancels

  # ------------------- final valuation of open trades -------------------
  equity_final = equity(close[-1])
  return_pct = (equity_final - initial_cash) / initial_cash * 100
  return equity_final, return_pct
