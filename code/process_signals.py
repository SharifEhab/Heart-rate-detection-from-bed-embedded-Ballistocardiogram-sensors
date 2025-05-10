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



### All the below functions --- I am not sure about the correctness of the code
### we should check the correctness and test on the notebook first
def interpolate_rr_to_bcg_grid(rr_times_ms, rr_values, t_bcg_ms):
    """
    Interpolate RR data (either HR or RR intervals) to match BCG time grid
    Returns:
        rr_interp: interpolated RR data on BCG time grid
    """
    rr_interp = np.interp(t_bcg_ms, rr_times_ms, rr_values)
    return rr_interp

def detect_and_remove_movements(bcg_signal, t_ms, plot=True):
    """
    Apply movement detection to identify and remove segments with body movements
    Returns:
        filtered_bcg: BCG signal with movement segments removed
        filtered_time: time points with movement segments removed
        movement_mask: boolean mask indicating where movements were detected
    """
    # Convert time from ms to samples (relative index)
    t_samples = np.arange(len(bcg_signal))
    
    # Apply detect_patterns function
    # Parameters: start_point, end_point, window_size, data, time, plot_flag
    pt1 = 0
    pt2 = MOVEMENT_WIN_SAMPS
    
    filtered_bcg, filtered_time = detect_patterns(
        pt1, pt2, MOVEMENT_WIN_SAMPS, bcg_signal, t_ms, plot
    )
    
    # Create a mask to track which time points were kept
    movement_mask = np.ones(len(bcg_signal), dtype=bool)
    for i in range(len(t_ms)):
        if t_ms[i] not in filtered_time:
            movement_mask[i] = False
    
    return filtered_bcg, filtered_time, movement_mask

def window_signals(bcg_signal, rr_signal, time_ms, window_sec=WIN_SEC, overlap_percent=0):
    """
    Segment signals into windows
    Returns:
        bcg_windows: list of BCG signal windows
        rr_windows: list of RR signal windows
        time_windows: list of time windows
    """
    window_samps = int(window_sec * TARGET_FS)
    step_samps = int(window_samps * (1 - overlap_percent/100))
    
    n_samples = len(bcg_signal)
    n_windows = (n_samples - window_samps) // step_samps + 1
    
    bcg_windows = []
    rr_windows = []
    time_windows = []
    
    for i in range(n_windows):
        start_idx = i * step_samps
        end_idx = start_idx + window_samps
        
        if end_idx <= n_samples:
            bcg_windows.append(bcg_signal[start_idx:end_idx])
            rr_windows.append(rr_signal[start_idx:end_idx])
            time_windows.append(time_ms[start_idx:end_idx])
    
    return bcg_windows, rr_windows, time_windows

def process_pair(subj, date, bcg_path, rr_path, output_dir=None):
    """
    Process a paired BCG and RR file:
    1. Synchronize signals
    2. Detect and remove movement sections
    3. Interpolate RR to match BCG time grid
    4. Window the filtered signals
    
    Returns:
        bcg_windows: list of BCG windows
        rr_windows: list of RR windows
        time_windows: list of time windows
    """
    print(f"Processing subject {subj}, date {date}")
    
    # Step 1: Synchronize signals
    bcg_sync, rr_hr_sync, rr_int_sync, t_sync_ms, rr_times_sync = synchronize_signals(bcg_path, rr_path)
    
    # Step 2: Detect and remove movement segments
    filtered_bcg, filtered_time, movement_mask = detect_and_remove_movements(bcg_sync, t_sync_ms, plot=True)
    
    # Step 3: Interpolate RR to match BCG time grid
    # First filter RR times and values to match movement-filtered data
    filtered_rr_times = rr_times_sync[np.isin(rr_times_sync, filtered_time)]
    filtered_rr_hr = rr_hr_sync[np.isin(rr_times_sync, filtered_time)]
    
    # Then interpolate to filtered BCG time grid
    rr_hr_interp = interpolate_rr_to_bcg_grid(filtered_rr_times, filtered_rr_hr, filtered_time)
    
    # Step 4: Window the signals
    bcg_windows, rr_windows, time_windows = window_signals(filtered_bcg, rr_hr_interp, filtered_time)
    
    # Save results if output_dir is provided
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save a plot of the filtered data with movement segments marked
        plt.figure(figsize=(15, 6))
        plt.subplot(211)
        plt.plot(t_sync_ms, bcg_sync, 'k-', alpha=0.5, label='Original BCG')
        plt.plot(filtered_time, filtered_bcg, 'r-', label='Filtered BCG')
        plt.legend()
        plt.title(f'Subject {subj}, Date {date} - BCG Signal')
        
        plt.subplot(212)
        plt.plot(rr_times_sync, rr_hr_sync, 'ko-', alpha=0.5, label='Original RR')
        plt.plot(filtered_time, rr_hr_interp, 'bo-', label='Interpolated RR')
        plt.legend()
        plt.title('Heart Rate')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{subj}_{date}_processed.png'))
        plt.close()
        
        # Save the windows
        np.save(os.path.join(output_dir, f'{subj}_{date}_bcg_windows.npy'), bcg_windows)
        np.save(os.path.join(output_dir, f'{subj}_{date}_rr_windows.npy'), rr_windows)
        np.save(os.path.join(output_dir, f'{subj}_{date}_time_windows.npy'), time_windows)
    
    return bcg_windows, rr_windows, time_windows

def process_all_pairs(paired_files, output_dir='results'):
    """
    Process all paired BCG and RR files
    Returns:
        all_bcg_windows: dict of BCG windows for each subject/date
        all_rr_windows: dict of RR windows for each subject/date
    """
    all_bcg_windows = {}
    all_rr_windows = {}
    all_time_windows = {}
    
    for subj, date, bcg_path, rr_path in paired_files:
        bcg_windows, rr_windows, time_windows = process_pair(subj, date, bcg_path, rr_path, output_dir)
        
        all_bcg_windows[(subj, date)] = bcg_windows
        all_rr_windows[(subj, date)] = rr_windows
        all_time_windows[(subj, date)] = time_windows
    
    return all_bcg_windows, all_rr_windows, all_time_windows

if __name__ == "__main__":
    # Example usage
    DATA_ROOT = "../dataset/data"
    OUTPUT_DIR = "../results"
    
    # Discover all BCG & RR paths
    pairs = []  # list of (subj, date, bcg_path, rr_path)
    
    for subj in sorted(os.listdir(DATA_ROOT)):
        bdir = os.path.join(DATA_ROOT, subj, "BCG")
        rdir = os.path.join(DATA_ROOT, subj, "Reference", "RR")
        
        if not os.path.isdir(bdir):
            continue
            
        bcg_files = [f for f in os.listdir(bdir) if f.endswith("_BCG.csv")]
        rr_files = os.path.isdir(rdir) and [f for f in os.listdir(rdir) if f.endswith("_RR.csv")] or []
        
        for bfn in bcg_files:
            subj_, date, kind = parse_filename(bfn)
            # Find matching RR file
            match_rr = None
            for rfn in rr_files:
                if parse_filename(rfn)[1] == date:
                    match_rr = os.path.join(rdir, rfn)
            
            if match_rr:  # Only add pairs that have both BCG and RR
                pairs.append((subj, date, os.path.join(bdir, bfn), match_rr))
    
    print(f"Found {len(pairs)} BCG-RR pairs.")
    
    # Process all pairs
    bcg_windows, rr_windows, time_windows = process_all_pairs(pairs, OUTPUT_DIR)
    print("Processing complete!") 