import pandas as pd
import numpy as np
from scipy.signal import resample_poly
from math import gcd

def load_bcg_csv(path):
    """
    Load a BCG CSV (subjectID_date_BCG.csv) as described in Li et al. (2024).
    Returns:
      bcg: 1-D numpy array of BCG amplitudes (as float)
      fs:  original sampling frequency (Hz)
    """
    df = pd.read_csv(path)
    bcg = df.iloc[:,0].astype(float).to_numpy() # get the bcg data
    fs   = float(df.iloc[0,2]) #get sampling freq from the dataframe
    return bcg, fs

def resample_bcg(bcg, orig_fs, target_fs):
    """
    Resample via polyphase FIR filtering (no aliasing) :contentReference[oaicite:1]{index=1}.
    Args:
      bcg:       1-D numpy array of original signal
      orig_fs:   original sampling rate (e.g. 140)
      target_fs: desired sampling rate (e.g. 50)
    Returns:
      bcg_rs: 1-D numpy array at target_fs
    """
    # up/down factors for resample_poly must be integers
    # target_fs/orig_fs = 5/14 → up=5, down=14
    up   = int(target_fs)
    down = int(orig_fs)
    g = gcd(up, down)
    up //= g; down //= g

    # ensure float input to avoid int‐type bug :contentReference[oaicite:2]{index=2}
    bcg = bcg.astype(float)
    bcg_rs = resample_poly(bcg, up, down)
    return bcg_rs

# usage example:
# bcg, fs = load_bcg_csv("01_20240115_BCG.csv")
# bcg50   = resample_bcg(bcg, fs, 50)
