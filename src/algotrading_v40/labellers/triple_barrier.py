import algotrading_v40_cpp.labellers as av40c_l
import pandas as pd

import algotrading_v40.utils.df as u_df


def _validate_inputs(
  *,
  prices: pd.Series,
  selected: pd.Series,  # whether to run the search on this index
  tpb: pd.Series,  # take profit barriers
  slb: pd.Series,  # stop loss barriers
  vb_tte: pd.Series,  # vertical barrier in trading time elapsed
  tte: pd.Series,  # trading time elapsed
  side: pd.Series,  # 1 for long bet, -1 for short bet
  do_index_check: bool = True,
):
  if do_index_check:
    u_df.check_indices_match(prices, selected, tpb, slb, vb_tte, tte, side)

  u_df.check_index_u_and_mi(prices.index)
  u_df.check_no_bad_values(prices, selected, tte, side)
  u_df.check_all_in(selected, values=(0, 1))
  u_df.check_all_gt0(prices, tpb.loc[selected == 1])
  u_df.check_all_gte0(vb_tte.loc[selected == 1])
  u_df.check_all_lt0(slb.loc[selected == 1])
  u_df.check_all_in(side, values=(-1, 1))


def triple_barrier(
  *,
  prices: pd.Series,
  selected: pd.Series,  # whether to run the search on this index
  tpb: pd.Series,  # take profit barriers
  slb: pd.Series,  # stop loss barriers
  vb_tte: pd.Series,  # vertical barrier in trading time elapsed
  tte: pd.Series,  # trading time elapsed
  side: pd.Series,  # 1 for long bet, -1 for short bet
) -> pd.DataFrame:
  _validate_inputs(
    prices=prices,
    selected=selected,
    tpb=tpb,
    slb=slb,
    vb_tte=vb_tte,
    tte=tte,
    side=side,
  )

  result = pd.DataFrame(
    av40c_l.triple_barrier_cpp(
      prices=prices.to_numpy(),
      selected=selected.to_numpy(),
      tpb=tpb.to_numpy(),
      slb=slb.to_numpy(),
      vb_tte=vb_tte.to_numpy(),
      tte=tte.to_numpy(),
      side=side.to_numpy(),
    ),
    index=prices.index,
  ).astype(
    {
      "tpha": "Int32",
      "slha": "Int32",
      "vbha": "Int32",
      "first_touch_at": "Int32",
      "first_touch_type": "Int32",
      "first_touch_raw_return": "float32",
    }
  )
  return result
