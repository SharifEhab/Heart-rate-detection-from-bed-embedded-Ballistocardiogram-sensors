import os
import pandas as pd
import numpy as np
from scipy.signal import resample
from datetime import datetime


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
      Timestamp (yyyy/MM/dd H:mm:ss), Heart Rate (bpm), RR Interval in seconds.
    Returns:
      times_ms    : 1‑d array of beat times in Unix ms
      hr          : 1‑d array of heart‑rate (bpm)
      rr_interval : 1‑d array of RR‑interval (s)
    """
    df = pd.read_csv(path)
    # parse the Timestamp strings → datetime
    dt = pd.to_datetime(df.iloc[:,0], format="%Y/%m/%d %H:%M:%S")
    # convert to Unix milliseconds
    times_ms = (dt.values.astype(np.int64) // 10**6).astype(float)
    print(times_ms)
    hr        = df['Heart Rate'].astype(float).to_numpy()
    rr_int    = df['RR Interval in seconds'].astype(float).to_numpy()
    return times_ms, hr, rr_int



#— resample BCG via Fourier method (scipy.signal.resample) —
def resample_signal(sig, orig_fs, target_fs):
    """
    Down/up‑sample `sig` from orig_fs → target_fs using Fourier resampling.
    Returns the resampled signal.
    """
    N_old = len(sig)
    N_new = int(round(N_old * (target_fs / orig_fs)))
    return resample(sig, N_new)