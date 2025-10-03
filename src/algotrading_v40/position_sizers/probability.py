import algotrading_v40_cpp.position_sizers as av40c_ps
import pandas as pd

# def get_bet_return(
#   p0: float,
#   p1: float,
#   side: int,
# ) -> float:
#   return side * ((p1 / p0) - 1)


# @dataclasses.dataclass(frozen=True)
# class Bet:
#   size: float
#   side: int  # -1 for short, 1 for long
#   bet_open_timestamp_int: int
#   bet_entry_price: float
#   tpb: float
#   slb: float
#   vb_timestamp_int_exec: int

#   def __post_init__(self):
#     if self.size < 0:
#       raise ValueError("size must be non-negative")
#     if self.size > 1:
#       raise ValueError("size must be less than or equal to 1")
#     if self.side not in [-1, 1]:
#       raise ValueError("side must be -1 or 1")
#     if self.bet_entry_price <= 0:
#       raise ValueError("bet_entry_price must be positive")

#   def should_close(
#     self,
#     current_timestamp_int: int,  # close timestamp_int of the current bar
#     # while streaming, bar might not be fully formed but we know what the close timestamp_int will be
#     highest_price: float,
#     lowest_price: float,
#   ) -> bool:
#     if current_timestamp_int >= self.vb_timestamp_int_exec:
#       return True
#     bet_return_h = get_bet_return(
#       p0=self.bet_entry_price, p1=highest_price, side=self.side
#     )
#     bet_return_l = get_bet_return(
#       p0=self.bet_entry_price, p1=lowest_price, side=self.side
#     )
#     if (
#       max(bet_return_h, bet_return_l) >= self.tpb
#       or min(bet_return_h, bet_return_l) <= self.slb
#     ):
#       return True
#     return False


# def average_bets(bets: list[Bet]) -> float:
#   if not bets:
#     return 0.0
#   return sum((bet.size * bet.side) for bet in bets) / len(bets)


# def probability_position_sizer_(
#   prob: pd.Series,  # probability of the prediction being correct
#   side: pd.Series,  # 1 or -1
#   open: pd.Series,  # open price series from OHLCV bars
#   high: pd.Series,  # high price series from OHLCV bars
#   low: pd.Series,  # low price series from OHLCV bars
#   selected: pd.Series,  # whether a new bet can be opened on this index. existing bets can be closed at all indices though.
#   tpb: pd.Series,  # take profit barriers; a take profit of 3% will be 0.03
#   slb: pd.Series,  # stop loss barriers; signed; so a stop loss of 3% will be -0.03
#   close_timestamp_int: pd.Series,  # integer representation of the index timestamp
#   # example: 2021-01-01 03:45:59.999000+00:00 -> 1609472759999000000
#   # (can be found by doing `df.index.astype(int)`)
#   vb_timestamp_int_exec: pd.Series,  # integer timestamp of vertical barriers
#   # any bet opened here needs to be closed on or before this timestamp.
#   # the _exec (execution) suffix is needed as vertical barriers during execution can
#   # be different from vertical barriers during labelling.
#   # example - while labelling, we might always set the vertical barrier to be 1 hour away from current time.
#   # but during execution, we might need to always close all positions before the end of
#   # a session.
#   # ------------------------------------------------------------------------------------------------
#   # qa_step_size: step size for discretising the dimensionless position size (i.e. between -1 and 1)
#   # should be in (0,1]
#   qa_step_size: float,
#   # step size for discretising position sizes in base asset (stock units, BTCUSDT units, etc.)
#   ba_step_size: float | None,
#   # maximum absolute position size in quote asset (INR, USD, USDT, etc.)
#   qa_max: float,
# ) -> pd.DataFrame:
#   if not prob.index.is_monotonic_increasing:
#     raise ValueError("prob index must be monotonic increasing")
#   if not prob.index.is_unique:
#     raise ValueError("prob index must be unique")
#   if not all(
#     prob.index.equals(other.index)
#     for other in [
#       side,
#       open,
#       high,
#       low,
#       selected,
#       tpb,
#       slb,
#       close_timestamp_int,
#       vb_timestamp_int_exec,
#     ]
#   ):
#     raise ValueError("All series must have the same index")

#   if qa_step_size <= 0:
#     raise ValueError("qa_step_size must be positive")
#   if qa_step_size > 1:
#     # qa_step_size acts on average bet size which is [0, 1]
#     # so having it > 1 does not make sense
#     raise ValueError("qa_step_size must be less than or equal to 1")
#   if ba_step_size is not None and ba_step_size <= 0:
#     raise ValueError("ba_step_size must be positive if provided")

#   if qa_max <= 0:
#     raise ValueError("qa_max must be positive")

#   if not pd.api.types.is_integer_dtype(side):
#     raise ValueError("side must be of integer dtype")
#   if not side.isin([1, -1]).all():
#     # only the binary case is supported for now
#     raise ValueError("side values must be only 1 or -1")
#   if not pd.api.types.is_integer_dtype(selected):
#     raise ValueError("selected must be of integer dtype")
#   if not selected.isin([0, 1]).all():
#     raise ValueError("selected values must be only 0 or 1")
#   if not ((prob >= 0.5) & (prob <= 1)).all():
#     # if prob < 0.5, the user should send -side with probability (1 - prob)
#     raise ValueError("prob values must be between 0.5 and 1 (inclusive)")
#   if not pd.api.types.is_integer_dtype(close_timestamp_int):
#     raise ValueError("close_timestamp_int must be of integer dtype")
#   if not pd.api.types.is_integer_dtype(vb_timestamp_int_exec):
#     raise ValueError("vb_timestamp_int_exec must be of integer dtype")

#   n = len(prob)

#   raw_qa_positions = np.empty(n, dtype=float)
#   discretised_qa_positions = np.empty(n, dtype=float)
#   final_ba_position = np.empty(n, dtype=float)
#   active_bets: list[Bet] = []

#   for i in range(n):
#     prob_i = prob.iloc[i]
#     side_i = side.iloc[i]
#     open_price_i = open.iloc[i]
#     # logic should work when data is coming as a stream
#     # while streaming, this step will run when the bar has just started forming
#     # say we are receiving the [t,t+1) bar right now.
#     # this step will run as soon as we receive the first tick AFTER time t (that will be the open price)
#     # the previous [t-1,t) bar is fully formed
#     # so to evaluate the barrier hits, we have access to the high and low prices of the [t-1,t)
#     # we DO NOT have access to the high and low prices of the [t,t+1) bar...we only have the open price
#     # that's why we use the previous bar's high and low prices
#     high_price_prev = high.iloc[i - 1] if i - 1 >= 0 else open_price_i
#     low_price_prev = low.iloc[i - 1] if i - 1 >= 0 else open_price_i
#     if (open_price_i <= 0) or (high_price_prev <= 0) or (low_price_prev <= 0):
#       raise ValueError(
#         "open_price, high_price_prev, and low_price_prev must be positive"
#       )
#     if not (
#       np.isfinite(open_price_i)
#       and np.isfinite(high_price_prev)
#       and np.isfinite(low_price_prev)
#     ):
#       raise ValueError("open_price, high_price_prev, and low_price_prev must be finite")
#     sel_i = selected.iloc[i]
#     tpb_i = tpb.iloc[i]
#     slb_i = slb.iloc[i]
#     close_ts_int_i = close_timestamp_int.iloc[i]
#     vb_ts_int_exec_i = vb_timestamp_int_exec.iloc[i]

#     if tpb_i <= 0 or slb_i >= 0:
#       raise ValueError("tpb and slb must be positive and negative respectively")

#     # open new bet if timestamp is selected by the data selector upstream
#     if sel_i == 1:
#       # prob_i is enforced to be >= 0.5
#       # so size would be >= 0
#       size = get_size(prob=prob_i)
#       active_bets.append(
#         Bet(
#           size=size,
#           side=side_i,
#           bet_open_timestamp_int=close_ts_int_i,
#           bet_entry_price=open_price_i,
#           tpb=tpb_i,
#           slb=slb_i,
#           vb_timestamp_int_exec=vb_ts_int_exec_i,
#         )
#       )

#     # close bets that hit any of the barriers
#     # this is done after opening a new bet to handle the case where the
#     # new bet needs to be closed as well
#     active_bets = [
#       bet
#       for bet in active_bets
#       if not bet.should_close(
#         current_timestamp_int=close_ts_int_i,
#         highest_price=max(high_price_prev, open_price_i),
#         lowest_price=min(low_price_prev, open_price_i),
#       )
#     ]

#     raw_qa_positions[i] = average_bets(bets=active_bets)
#     # discretise_position does bankers rounding and does not always round down
#     # this is fine since we clip the position to be between -qa_max and qa_max
#     # on the next line
#     rdqa_pos_i = discretise_position(
#       position=raw_qa_positions[i], step_size=qa_step_size
#     )
#     discretised_qa_positions[i] = np.clip(
#       a=qa_max * rdqa_pos_i,
#       a_min=-qa_max,
#       a_max=qa_max,
#     )

#     # Prevent unnecessary base asset position changes when quote asset position is unchanged.
#     # This avoids spurious trades caused by price movements when the underlying position
#     # sizing decision (discretised_qa_position) hasn't actually changed.
#     if i > 0 and discretised_qa_positions[i - 1] == discretised_qa_positions[i]:
#       final_ba_position[i] = final_ba_position[i - 1]
#     else:
#       final_ba_position[i] = discretised_qa_positions[i] / open_price_i
#       if ba_step_size is not None:
#         # always round down in magnitude to the nearest step size
#         final_ba_position[i] = (
#           np.sign(final_ba_position[i])
#           * np.floor(np.abs(final_ba_position[i]) / ba_step_size)
#           * ba_step_size
#         )

#   return pd.DataFrame(
#     {
#       # this is the raw dimensionless position size between -1 and 1
#       "raw_qa_position": raw_qa_positions,
#       # this is the quote asset (INR, USD, USDT etc.) position size between -qa_max and qa_max
#       "discretised_qa_position": discretised_qa_positions,
#       # this is in # of stock units, # of BTCUSDT units, etc.
#       "final_ba_position": final_ba_position,
#     },
#     index=prob.index,
#   )


def probability_position_sizer(
  *,
  prob: pd.Series,  # probability of the prediction being correct
  side: pd.Series,  # 1 or -1
  open: pd.Series,  # open price series from OHLCV bars
  high: pd.Series,  # high price series from OHLCV bars
  low: pd.Series,  # low price series from OHLCV bars
  selected: pd.Series,  # whether a new bet can be opened on this index. existing bets can be closed at all indices though.
  tpb: pd.Series,  # take profit barriers; a take profit of 3% will be 0.03
  slb: pd.Series,  # stop loss barriers; signed; so a stop loss of 3% will be -0.03
  vb_timestamp_exec_int: pd.Series,  # integer timestamp of vertical barriers
  # any bet opened here needs to be closed on or before this timestamp.
  # the _exec (execution) suffix is needed as vertical barriers during execution can
  # be different from vertical barriers during labelling.
  # example - while labelling, we might always set the vertical barrier to be 1 hour away from current time.
  # but during execution, we might need to always close all positions before the end of
  # a session.
  # ------------------------------------------------------------------------------------------------
  # qa_step_size: step size for discretising the dimensionless position size (i.e. between -1 and 1)
  # should be in (0,1]
  qa_step_size: float,
  # step size for discretising position sizes in base asset (stock units, BTCUSDT units, etc.)
  ba_step_size: float | None,
  # maximum absolute position size in quote asset (INR, USD, USDT, etc.)
  qa_max: float,
  position_allowed: pd.Series = None,  # position allowed
) -> pd.DataFrame:
  index = prob.index
  for sn, s in (
    ("side", side),
    ("open", open),
    ("high", high),
    ("low", low),
    ("selected", selected),
    ("tpb", tpb),
    ("slb", slb),
    ("vb_timestamp_exec_int", vb_timestamp_exec_int),
  ):
    if not s.index.equals(index):
      raise ValueError(f"{sn} index must be the same as prob index")

  if position_allowed is not None:
    if not position_allowed.index.equals(index):
      raise ValueError("position_allowed index must be the same as prob index")
    if not position_allowed.isin([0, 1]).all():
      raise ValueError("position_allowed values must be only 0 or 1")
  else:
    position_allowed = pd.Series([1] * len(prob), index=index)

  close_timestamp_int = index.astype(int)
  # integer representation of the index timestamp
  # example: 2021-01-01 03:45:59.999000+00:00 -> 1609472759999000000
  return pd.DataFrame(
    data=av40c_ps.probability_position_sizer_cpp(
      prob=prob.to_numpy(),
      side=side.to_numpy(),
      open=open.to_numpy(),
      high=high.to_numpy(),
      low=low.to_numpy(),
      selected=selected.to_numpy(),
      tpb=tpb.to_numpy(),
      slb=slb.to_numpy(),
      close_timestamp_int=close_timestamp_int.to_numpy(),
      vb_timestamp_exec_int=vb_timestamp_exec_int.to_numpy(),
      position_allowed=position_allowed.to_numpy(),
      qa_step_size=qa_step_size,
      ba_step_size=ba_step_size,
      qa_max=qa_max,
    ),
    index=index,
  )
