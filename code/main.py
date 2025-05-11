import math
import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from ecgdetectors import Detectors
from band_pass_filtering import band_pass_filtering
from compute_vitals import vitals
from detect_apnea_events import apnea_events
from detect_body_movements import detect_patterns_movements
from modwt_matlab_fft import modwt
from modwt_mra_matlab_fft import modwtmra
from remove_nonLinear_trend import remove_nonLinear_trend
from data_subplot import data_subplot
import process_signals as process
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Constants
DATA_ROOT = r'C:\Users\DELL\Downloads\dataset(1)\dataset\data'  # Adjust this path to your dataset directory
TARGET_F_SAMPLE = 50  # 50 Hz
WIN_SEC = 6 * 60  # 6 minutes
WIN_SAMPLES = int(WIN_SEC * TARGET_F_SAMPLE)  # 18000 samples at 50 Hz
SEGMENT_SEC = 5  # 5 seconds
SEGMENT_SAMPLES = int(SEGMENT_SEC * TARGET_F_SAMPLE)  # 250 samples

def main():
    # Main program starts here
    print('\nstart processing ...')
    paired = process.get_bcg_rr_pairs(DATA_ROOT)
    all_hr_pairs = []  # To store (BCG_HR, Ref_HR) pairs across all subjects and dates

    for subj, date, bcg_path, rr_path in paired:
        print("###################################################")
        print(f"Processing {subj} {date}: {bcg_path}, {rr_path}")
        bcg_sync, rr_hr_sync, rr_int_sync, t_sync_ms, rr_times_sync = process.synchronize_signals(bcg_path, rr_path)
        
        # Skip if no data after synchronization
        if len(bcg_sync) == 0 or len(rr_hr_sync) == 0:
            print(f"{subj} {date}: No valid data after synchronization")
            continue
        
        # Detect and remove movements
        binary_mask = detect_patterns_movements(0, SEGMENT_SAMPLES, SEGMENT_SAMPLES, bcg_sync, t_sync_ms, plot=0)
        
        # Extract clean windows from the synchronized data
        clean_windows = process.extract_clean_windows(bcg_sync, t_sync_ms, binary_mask, 
                                                      rr_times_sync, rr_hr_sync, rr_int_sync,
                                                      subj, date, WIN_SAMPLES)
        
        print(f"{subj} {date}: Found {len(clean_windows)} clean windows")
        
        # Process each clean window
        for idx, window in enumerate(clean_windows):
            bcg_window = window['bcg']
            time_window = window['t_bcg']
            rr_hr_window = window['rr_hr']
            
            # Compute BCG-based HR
            hr_avg_bcg, _, _, _, _ = process.process_bcg_window(bcg_window, time_window, subj, date, idx)
            
            if hr_avg_bcg is not None and len(rr_hr_window) > 0:
                # Compute average reference HR from RR intervals
                hr_avg_ref = np.mean(rr_hr_window)
                # Store the pair (BCG_HR, Ref_HR)
                all_hr_pairs.append((hr_avg_bcg, hr_avg_ref))
        
        print("###################################################")

    # Compute error metrics and generate plots after processing all subjects and dates
    if all_hr_pairs:
        y_true = np.array([pair[1] for pair in all_hr_pairs])  # Reference HR
        y_pred = np.array([pair[0] for pair in all_hr_pairs])  # BCG-based HR
        
        # Error metrics
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        print(f"Mean Absolute Error (MAE): {mae:.2f} BPM")
        print(f"Root Mean Square Error (RMSE): {rmse:.2f} BPM")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        
        # Bland-Altman plot
        mean_hr = (y_true + y_pred) / 2
        diff_hr = y_pred - y_true
        plt.figure(figsize=(8, 6))
        plt.scatter(mean_hr, diff_hr, alpha=0.5)
        plt.axhline(np.mean(diff_hr), color='r', linestyle='--', label='Mean Difference')
        plt.axhline(np.mean(diff_hr) + 1.96 * np.std(diff_hr), color='g', linestyle='--', label='+1.96 SD')
        plt.axhline(np.mean(diff_hr) - 1.96 * np.std(diff_hr), color='g', linestyle='--', label='-1.96 SD')
        plt.xlabel('Mean Heart Rate (BPM)')
        plt.ylabel('Difference (BCG - ECG) (BPM)')
        plt.title('Bland-Altman Plot')
        plt.legend()
        plt.savefig('bland_altman_plot.png')
        
        # Pearson correlation scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([np.min(y_true), np.max(y_true)], [np.min(y_true), np.max(y_true)], color='r', linestyle='--', label='Line of Identity')
        plt.xlabel('ECG Heart Rate (BPM)')
        plt.ylabel('BCG Heart Rate (BPM)')
        plt.title('Pearson Correlation Scatter Plot')
        corr, _ = pearsonr(y_true, y_pred)
        plt.text(np.min(y_true) + 0.05 * (np.max(y_true) - np.min(y_true)), 
                 np.max(y_pred) - 0.05 * (np.max(y_pred) - np.min(y_pred)), 
                 f'Correlation: {corr:.2f}', fontsize=12)
        plt.savefig('correlation_plot.png')
    else:
        print("No valid HR pairs found.")

    print('\nEnd processing ...')

if __name__ == '__main__':
    main()