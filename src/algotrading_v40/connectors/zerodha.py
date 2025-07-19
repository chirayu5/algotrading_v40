import os
import pickle

import kiteconnect


def get_live_kite_object(force_refresh: bool) -> kiteconnect.KiteConnect:
  stored_kite_path = os.path.join(os.path.dirname(__file__), "zerodha_kite.pkl")
  if force_refresh:
    API_KEY = "w1n384co2mp3mfej"
    API_SECRET = "65yvqbx9yezhxwu5ncakgz06a9gb397z"
    kite = kiteconnect.KiteConnect(api_key=API_KEY)
    print(kite.login_url())
    request_token = input("Enter the request token: ")
    data = kite.generate_session(request_token, api_secret=API_SECRET)
    kite.set_access_token(data["access_token"])  # type: ignore
    # replace old kite object with new one
    pickle.dump(kite, open(stored_kite_path, "wb"))
    return kite

  if not os.path.exists(stored_kite_path):
    print("No stored kite object found, refreshing")
    return get_live_kite_object(force_refresh=True)

  stored_kite = pickle.load(open(stored_kite_path, "rb"))
  try:
    # this will fail if the kite object is not valid
    _ = stored_kite.ltp("NSE:HDFCBANK")["NSE:HDFCBANK"]["instrument_token"]  # type: ignore
    return stored_kite
  except Exception:
    print("Kite object is not valid, refreshing")
    return get_live_kite_object(force_refresh=True)
