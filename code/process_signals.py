import os
import numpy as np
import pandas as pd
from scipy.signal import resample
import matplotlib.pyplot as plt
from detect_body_movements import detect_patterns
from band_pass_filtering import band_pass_filtering
from modwt_matlab_fft import modwt
from modwt_mra_matlab_fft import modwtmra
from scipy.signal import find_peaks
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

def extract_clean_windows(bcg_sync, t_sync_ms, binary_mask, rr_times_sync, rr_hr_sync, rr_int_sync, subj, date, win_samples):
            """
            Extract clean windows from synchronized BCG and RR data.
            
            Args:
                bcg_sync: Synchronized BCG signal
                t_sync_ms: Synchronized time axis in milliseconds
                binary_mask: Movement detection mask (1=no movement, 0=movement|no occupancy)
                rr_times_sync: Synchronized RR beat times
                rr_hr_sync: Synchronized heart rate values
                rr_int_sync: Synchronized RR intervals
                subj: Subject ID
                date: Recording date
                win_samples: Window size in samples
                
            Returns:
                List of clean windows with no movements and corresponding RR data
            """
            num_windows = len(bcg_sync) // win_samples
            clean_windows = []
            print(f"num_windows: {num_windows}")
            for k in range(num_windows):
                start_idx = k * win_samples
                end_idx = start_idx + win_samples
                
                # Check if window has no movement (all 1s in binary mask)
                if np.all(binary_mask[start_idx:end_idx] == 1):
                    bcg_window = bcg_sync[start_idx:end_idx]
                    t_bcg_window = t_sync_ms[start_idx:end_idx]
                    t_start = t_bcg_window[0]
                    t_end = t_bcg_window[-1]
                    
                    # Select corresponding RR beats
                    mask_rr = (rr_times_sync >= t_start) & (rr_times_sync < t_end)
                    rr_times_window = rr_times_sync[mask_rr]
                    rr_hr_window = rr_hr_sync[mask_rr]
                    rr_int_window = rr_int_sync[mask_rr]
                    
                   


                    # Store window if RR data exists
                    if len(rr_hr_window) > 0:
                         # windows time range of both the bcg and the rr
                        print(f"RR times window{k}:",rr_times_window[0],rr_times_window[-1])
                        print(f"t_bcg_window{k}:",t_bcg_window[0],t_bcg_window[-1])
                        clean_windows.append({
                            'subj': subj,
                            'date': date,
                            't_start': t_start,
                            't_end': t_end,
                            'bcg': bcg_window,
                            't_bcg': t_bcg_window,
                            'rr_times': rr_times_window,
                            'rr_hr': rr_hr_window,
                            'rr_int': rr_int_window
                        })
            return clean_windows









############# Bad Code ##############

# def process_bcg_window(bcg_window, time_window, subj, date, window_idx):
#     """
#     Process a single clean BCG window to extract heart rate:
#     1. Apply band-pass filter to extract BCG signal
#     2. Apply wavelet transform to extract S4 component (J-peak emphasis)
#     3. Detect J-peaks and compute heart rate
    
#     Parameters:
#         bcg_window: Clean BCG window signal
#         time_window: Time values for the window
#         subj: Subject ID
#         date: Recording date
#         window_idx: Window index for this subject/date
        
#     Returns:
#         hr_avg: Average heart rate computed from BCG
#         j_peak_intervals: Array of intervals between J-peaks (in seconds)
#         filtered_bcg: Filtered BCG signal
#         wavelet_s4: S4 component from wavelet decomposition
#         j_peak_indices: Indices of detected J-peaks
#     """
#     # Step 1: Apply band-pass filter to extract BCG signal (heart rate component)
#     filtered_bcg = band_pass_filtering(bcg_window, TARGET_F_SAMPLE, "bcg")
    
#     # Step 2: Apply wavelet transform
#     w = modwt(filtered_bcg, 'bior3.9', 4)  # Level 4 wavelet decomposition
#     dc = modwtmra(w, 'bior3.9')  # Multi-resolution analysis
#     wavelet_s4 = dc[4]  # Extract S4 component (J-peak emphasis)
    
#     # Step 3: Detect J-peaks
#     # Parameters for peak detection
#     min_peak_distance = int(0.5 * TARGET_F_SAMPLE)  # Minimum 0.5 seconds between peaks
#     min_peak_height = 0.3 * np.max(wavelet_s4)  # Peaks must be at least 30% of max amplitude
    
#     # Find peaks in the S4 component
#     j_peak_indices, _ = find_peaks(wavelet_s4, height=min_peak_height, distance=min_peak_distance)
    
#     # Step 4: Compute heart rate from J-peaks
#     if len(j_peak_indices) > 1:
#         # Calculate intervals between consecutive J-peaks (in samples)
#         j_peak_intervals_samples = np.diff(j_peak_indices)
        
#         # Convert intervals to seconds
#         j_peak_intervals = j_peak_intervals_samples / TARGET_F_SAMPLE
        
#         # Convert intervals to beats per minute (BPM)
#         hr_values = 60 / j_peak_intervals
        
#         # Filter unreasonable heart rates (e.g., below 40 or above 150 BPM)
#         hr_values = hr_values[(hr_values >= 40) & (hr_values <= 150)]
        
#         # Calculate average heart rate
#         if len(hr_values) > 0:
#             hr_avg = np.mean(hr_values)
#         else:
#             hr_avg = None
#     else:
#         j_peak_intervals = []
#         hr_avg = None
    
#     # Step 5: Plot the results for visualization
#     os.makedirs(RESULTS_DIR, exist_ok=True)
#     plt.figure(figsize=(15, 12))
    
#     # Plot the original BCG signal
#     plt.subplot(4, 1, 1)
#     plt.plot(np.arange(len(bcg_window))/TARGET_F_SAMPLE, bcg_window)
#     plt.title(f'Subject {subj}, Date {date}, Window {window_idx+1}: Original BCG Signal')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Amplitude')
    
#     # Plot the filtered BCG signal
#     plt.subplot(4, 1, 2)
#     plt.plot(np.arange(len(filtered_bcg))/TARGET_F_SAMPLE, filtered_bcg)
#     plt.title('Filtered BCG Signal (Band-pass: 2.5-5 Hz)')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Amplitude')
    
#     # Plot the S4 wavelet component with detected J-peaks
#     plt.subplot(4, 1, 3)
#     plt.plot(np.arange(len(wavelet_s4))/TARGET_F_SAMPLE, wavelet_s4)
#     if len(j_peak_indices) > 0:
#         plt.plot(j_peak_indices/TARGET_F_SAMPLE, wavelet_s4[j_peak_indices], 'ro', label='J-peaks')
#     plt.title('S4 Wavelet Component with J-peak Detection')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Amplitude')
#     plt.legend()
    
#     # Plot the calculated heart rate from J-peaks
#     plt.subplot(4, 1, 4)
#     if len(j_peak_intervals) > 0:
#         # Plot instantaneous heart rate at each peak
#         peak_times = j_peak_indices[1:] / TARGET_F_SAMPLE  # Time of each peak after the first one
#         plt.plot(peak_times, hr_values, 'bo-', label='Instantaneous HR')
        
#         # Plot the average HR as a horizontal line
#         plt.axhline(y=hr_avg, color='r', linestyle='--', 
#                    label=f'Average HR: {hr_avg:.1f} BPM')
#     plt.title('Heart Rate from J-peaks')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Heart Rate (BPM)')
#     plt.ylim(40, 150)
#     plt.legend()
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(RESULTS_DIR, f'{subj}_{date}_window{window_idx+1}_jpeaks.png'))
#     plt.close()
    
#     return hr_avg, j_peak_intervals, filtered_bcg, wavelet_s4, j_peak_indices






