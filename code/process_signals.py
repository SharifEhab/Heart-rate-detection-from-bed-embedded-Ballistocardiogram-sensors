import os
import numpy as np
import pandas as pd
from scipy.signal import resample
import matplotlib.pyplot as plt
from detect_body_movements import detect_patterns
# Constants
TARGET_FS = 50.0  # Target sampling frequency in Hz
WIN_SEC = 6 * 60  # Window size in seconds (6 minutes)
WIN_SAMPS = int(WIN_SEC * TARGET_FS)  # Window size in samples
MOVEMENT_WIN_SEC = 10  # Window size for movement detection (10 seconds)
MOVEMENT_WIN_SAMPS = int(MOVEMENT_WIN_SEC * TARGET_FS)  # Window size for movement detection


def parse_filename(fn):
    """
    Parse names like "01_20231104_BCG.csv" → subject="01", date="20231104", dtype="BCG"
    """
    base, _ = os.path.splitext(fn)
    subj, date, dtype = base.split("_")
    return subj, date, dtype

def get_bcg_rr_pairs(DATA_ROOT):
    """
    Get all the BCG and RR pairs in the dataset
    """
    pairs = []   # list of (subj, date, bcg_path, rr_path_or_None)
    for subj in sorted(os.listdir(DATA_ROOT)):
        bdir = os.path.join(DATA_ROOT,subj,"BCG")
        rdir = os.path.join(DATA_ROOT,subj,"Reference","RR")

        if not os.path.isdir(bdir): continue

        bcg_files = [f for f in os.listdir(bdir) if f.endswith("_BCG.csv")]
        rr_files  = os.path.isdir(rdir) and [f for f in os.listdir(rdir) if f.endswith("_RR.csv")] or []
        for bfn in bcg_files:
            subj_,date,kind = parse_filename(bfn)
            # find matching rr
            match_rr = None
            for rfn in rr_files:
                if parse_filename(rfn)[1]==date:
                    match_rr = os.path.join(rdir,rfn)
            pairs.append((subj, date,os.path.join(bdir,bfn),match_rr))
    # quick summary
    print(f"Found {len(pairs)} BCG nights, of which {sum(1 for p in pairs if p[3])} have RR.")
    paired = [(s,d,bcg,rr) for s,d,bcg,rr in pairs if rr is not None]
    return paired

def load_bcg_csv(path):
    """
    Load BCG CSV file with columns: BCG signal, Timestamp (ms), sampling rate (Hz)
    Returns:
        sig: 1-d array of BCG signal
        fs_orig: original sampling frequency
        start_ts_ms: start timestamp in milliseconds
    """
    df = pd.read_csv(path)
    sig = df.iloc[:, 0].astype(float).to_numpy()
    start_ts_ms = float(df.iloc[0, 1])  # "Timestamp" column, Unix ms
    fs_orig = float(df.iloc[0, 2])      # "fs" column, Hz
    return sig, fs_orig, start_ts_ms

def load_rr_csv(path):
    """
    Load Reference RR CSV with columns:
      Timestamp (yyyy/MM/dd H:mm:ss), Heart Rate (bpm), RR Interval in seconds.
    Returns:
      times_ms: 1-d array of beat times in Unix ms
      hr: 1-d array of heart-rate (bpm)
      rr_interval: 1-d array of RR-interval (s)
    """
    df = pd.read_csv(path)
    # parse the Timestamp strings → datetime
    dt = pd.to_datetime(df.iloc[:, 0], format="%Y/%m/%d %H:%M:%S")
    # convert to Unix milliseconds
    times_ms = (dt.values.astype(np.int64) // 10**6).astype(float)
    hr = df['Heart Rate'].astype(float).to_numpy()
    rr_int = df['RR Interval in seconds'].astype(float).to_numpy()
    return times_ms, hr, rr_int

def resample_signal(sig, fs_orig, fs_target):
    """
    Resample signal from fs_orig to fs_target using Fourier method
    Returns:
        sig_rs: resampled signal
        N_new: new length of signal
    """
    N_old = len(sig)
    N_new = int(round(N_old * (fs_target/fs_orig)))
    sig_rs = resample(sig, N_new)
    return sig_rs, N_new

def synchronize_signals(bcg_path, rr_path):
    """
    Load BCG and RR signals, resample BCG to 50 Hz, and synchronize them based on timestamps
    Returns:
        bcg_sync: synchronized BCG signal at 50 Hz
        rr_hr_sync: synchronized heart rate
        rr_int_sync: synchronized RR intervals
        t_sync_ms: synchronized time axis in milliseconds
    """
    # Load and resample BCG signal
    bcg_sig, fs_orig, start_bcg_ms = load_bcg_csv(bcg_path)
    bcg50, N50 = resample_signal(bcg_sig, fs_orig, TARGET_FS)
    
    # Create time axis for resampled BCG
    t_bcg50_ms = start_bcg_ms + np.arange(N50) * (1000.0 / TARGET_FS)
    
    # Load RR reference
    rr_times_ms, rr_hr, rr_int = load_rr_csv(rr_path)
    
    # Find overlapping time interval
    t0 = max(t_bcg50_ms[0], rr_times_ms.min()) 
    t1 = min(t_bcg50_ms[-1], rr_times_ms.max())
    
    print(f"Overlap: {t0:.0f} … {t1:.0f} ms ({(t1-t0)/1000:.1f} s)")
    
    # Extract overlapping segments
    mask_b = (t_bcg50_ms >= t0) & (t_bcg50_ms <= t1)
    mask_r = (rr_times_ms >= t0) & (rr_times_ms <= t1)
    
    bcg_sync = bcg50[mask_b]
    rr_hr_sync = rr_hr[mask_r]
    rr_int_sync = rr_int[mask_r]
    
    t_sync_ms = t_bcg50_ms[mask_b]
    rr_times_sync = rr_times_ms[mask_r]
    
    print(f"Synchronized {len(bcg_sync)} BCG samples, {len(rr_hr_sync)} RR beats")
    
    return bcg_sync, rr_hr_sync, rr_int_sync, t_sync_ms, rr_times_sync