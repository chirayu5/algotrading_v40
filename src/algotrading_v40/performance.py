import numpy as np
import pandas as pd


def equity(p: float, cash: float, trades: list[dict[str, float]]) -> float:
  return cash + sum(t["size"] * (p - t["entry"]) for t in trades)


def margin_used(p: float, trades: list[dict[str, float]]) -> float:  # leverage = 1
  return sum(abs(t["size"]) * p for t in trades)


def check_trades(trades: list[dict[str, float]]) -> None:
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
) -> tuple[float, float]:
  close = df["Close"].to_numpy(float)
  target = df["final_ba_position"].to_numpy(int)

  cash: float = initial_cash
  trades: list[dict[str, float]] = []  # FIFO queue – {'size', 'entry'}

  for i in range(1, len(df) - 1):  # bar executing fills
    check_trades(trades)
    px = close[i]
    want = target[i]  # desired position at the end of the bar
    pos = sum(t["size"] for t in trades)
    delta = want - pos  # change required this bar
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
    cost_total = need * (px + need * px * commission_rate)

    avail = max(
      0.0, equity(p=px, cash=cash, trades=trades) - margin_used(p=px, trades=trades)
    )  # free equity
    if cost_total <= avail:  # broker accepts order
      signed = int(np.sign(delta)) * need
      trades.append({"size": signed, "entry": px})
      cash -= need * px * commission_rate  # entry commission
    # else: broker silently cancels
    check_trades(trades)

  # ------------------- final valuation of open trades -------------------
  equity_final = equity(p=close[-1], cash=cash, trades=trades)
  return_pct = 100 * (equity_final - initial_cash) / initial_cash
  return equity_final, return_pct
