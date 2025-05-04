import os
import pandas as pd
import numpy as np
from scipy.signal import resample_poly
from math import gcd

#— parse filenames —
def parse_filename(fn):
    subj, date, kind = os.path.splitext(fn)[0].split("_")
    return subj, date, kind  # e.g. "01","20231104","BCG"

#— load raw BCG CSV —
def load_bcg_csv(path):
    df = pd.read_csv(path)
    sig = df.iloc[:,0].astype(float).to_numpy()
    fs  = float(df.iloc[0,2])
    return sig, fs

#— load RR CSV (reference) —
def load_rr_csv(path):
    """
    Load Reference RR CSV with columns:
      Timestamp (yyyy/MM/dd H:mm:ss), Heart Rate (bpm), RR Interval in seconds
    Returns:
      times: 1‑d array of seconds (float) from first beat
      hr:    1‑d array of heart‑rate (bpm)
    """
    df = pd.read_csv(path)
    # parse the Timestamp strings:
    dt = pd.to_datetime(df.iloc[:,0], format="%Y/%m/%d %H:%M:%S")
    # convert to seconds since first timestamp
    t_secs = (dt.astype(np.int64) / 1e9)  - (dt.astype(np.int64).iloc[0] / 1e9)
    hr     = df.iloc[:,1].astype(float).to_numpy()
    return t_secs.to_numpy(), hr

#— rational resample via polyphase (no aliasing) —
def resample_signal(sig, orig_fs, target_fs):
    up, down = int(target_fs), int(orig_fs)
    g = gcd(up,down)
    up//=g; down//=g
    return resample_poly(sig.astype(float), up, down)
