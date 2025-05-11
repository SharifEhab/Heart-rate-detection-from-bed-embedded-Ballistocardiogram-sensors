# Import required libraries
import math
import os
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# A collection of ECG heartbeat detection algorithms implemented in Python
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
from beat_to_beat import compute_rate
from scipy.stats import linregress
# ======================================================================================================================

DATA_ROOT = 'D:/Data Analytics/projectrepo/dataset/data' # path to the dataset folder which has all patients(01, 02 , .... )
TARGET_F_SAMPLE = 50 # 50 Hz
WIN_SEC = 2 * 60  # 2 minutes
WIN_SAMPLES = int(WIN_SEC * 50)  # 6000 samples at 50 Hz
SEGMENT_SEC = 5  # 5 seconds
SEGMENT_SAMPLES = int(SEGMENT_SEC * 50)  # 250 samples
def main():
    # Main program starts here
    print('\nstart processing ...')
    paired = process.get_bcg_rr_pairs(DATA_ROOT)
    all_results = []
    for subj, date, bcg_path, rr_path in paired:
        print("###################################################")
        print(f"Processing {subj} {date}: {bcg_path}, {rr_path}")
        bcg_sync, rr_hr_sync, rr_int_sync, t_sync_ms, rr_times_sync = process.synchronize_signals(bcg_path,rr_path)
        # Skip if no data after synchronization
        if len(bcg_sync) == 0 or len(rr_hr_sync) == 0:
            print(f"{subj} {date}: No valid data after synchronization")
            continue
        # Detect and remove movements
        # Apply movement detection to entire BCG signal
        binary_mask = detect_patterns_movements(0, SEGMENT_SAMPLES, SEGMENT_SAMPLES, bcg_sync, t_sync_ms, plot=0)
            
        # Extract clean windows from the synchronized data
        clean_windows = process.extract_clean_windows(bcg_sync, t_sync_ms, binary_mask, 
                                             rr_times_sync, rr_hr_sync, rr_int_sync,
                                             subj, date, WIN_SAMPLES)
        
        print(f"{subj} {date}: Found {len(clean_windows)} clean windows")
        print("###################################################")
        # Calculate heart rates for this subject's windows
        window_results = process.calculate_window_heart_rates(clean_windows)
        all_results.extend(window_results)
        
        print("###################################################")
    # Compute error metrics across all patients
    hr_bcg_all = [r['hr_bcg'] for r in all_results if r['hr_bcg'] > 0 and r['hr_ref'] > 0]
    hr_ref_all = [r['hr_ref'] for r in all_results if r['hr_bcg'] > 0 and r['hr_ref'] > 0]
    if len(hr_bcg_all) > 0:
        mae = np.mean(np.abs(np.array(hr_bcg_all) - np.array(hr_ref_all)))
        rmse = np.sqrt(np.mean((np.array(hr_bcg_all) - np.array(hr_ref_all))**2))
        mask = np.array(hr_ref_all) > 0
        if np.any(mask):
            mape = np.mean(np.abs((np.array(hr_bcg_all)[mask] - np.array(hr_ref_all)[mask]) / np.array(hr_ref_all)[mask])) * 100
        else:
            mape = np.nan
        print(f"Error Metrics (All Patients):")
        print(f"MAE: {mae:.2f} bpm")
        print(f"RMSE: {rmse:.2f} bpm")
        print(f"MAPE: {mape:.2f}%" if not np.isnan(mape) else "MAPE: undefined")
        
        # Generate plots for all data
        # Bland-Altman
        diff = np.array(hr_bcg_all) - np.array(hr_ref_all)
        avg = (np.array(hr_bcg_all) + np.array(hr_ref_all)) / 2
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        
        plt.figure()
        plt.scatter(avg, diff)
        plt.axhline(mean_diff, color='r', linestyle='--')
        plt.axhline(mean_diff + 1.96 * std_diff, color='g', linestyle='--')
        plt.axhline(mean_diff - 1.96 * std_diff, color='g', linestyle='--')
        plt.xlabel('Average HR (bpm)')
        plt.ylabel('Difference (BCG - Ref) (bpm)')
        plt.title('Bland-Altman Plot (All Patients)')
        plt.savefig('results/bland_altman_all.png')
        plt.close()
        
        # Scatter plot
        slope, intercept, r_value, p_value, std_err = linregress(hr_bcg_all, hr_ref_all)
        plt.figure()
        plt.scatter(hr_bcg_all, hr_ref_all)
        plt.plot(hr_bcg_all, intercept + slope * np.array(hr_bcg_all), 'r')
        plt.xlabel('BCG HR (bpm)')
        plt.ylabel('Reference HR (bpm)')
        plt.title(f'Correlation: r={r_value:.2f} (All Patients)')
        plt.savefig('results/scatter_all.png')
        plt.close()
        
        # Boxplot
        errors = np.array(hr_bcg_all) - np.array(hr_ref_all)
        plt.figure()
        plt.boxplot(errors)
        plt.ylabel('Error (BCG - Ref) (bpm)')
        plt.title('Boxplot of HR Errors (All Patients)')
        plt.savefig('results/boxplot_all.png')
        plt.close()
    else:
        print("No valid HR data to compute metrics or generate plots")


    # After windowing, we have a list of clean bcg windows with no movements for each subject and date and the corresponding RR beats
    # Now we have multiple steps to follow:
    # 1. For each clean bcg window, we need to apply the band pass filter to extract the bcg signal
    # 2. Then we need to apply the wavelet transform function to extract the 4th level wavelet coefficients
    # 3. Then extract j peaks and compute average heart rate of window size 
    # 4. Then compute the average heart rate of corresponding RR beats
    # 5. Then stats and plot the results






    # start_point, end_point, window_shift, fs = 0, 500, 500, 50
    # # ==========================================================================================================
    # data_stream, utc_time = detect_patterns(start_point, end_point, window_shift, data_stream, utc_time, plot=1)
    # # ==========================================================================================================
    # # BCG signal extraction
    # movement = band_pass_filtering(data_stream, fs, "bcg")
    # # ==========================================================================================================
    # # Respiratory signal extraction
    # breathing = band_pass_filtering(data_stream, fs, "breath")
    # breathing = remove_nonLinear_trend(breathing, 3)
    # breathing = savgol_filter(breathing, 11, 3)
    # # ==========================================================================================================
    # w = modwt(movement, 'bior3.9', 4)
    # dc = modwtmra(w, 'bior3.9')
    # wavelet_cycle = dc[4]
    # # ==========================================================================================================
    # # Vital Signs estimation - (10 seconds window is an optimal size for vital signs measurement)
    # t1, t2, window_length, window_shift = 0, 500, 500, 500
    # hop_size = math.floor((window_length - 1) / 2)
    # limit = int(math.floor(breathing.size / window_shift))
    # # ==========================================================================================================
    # # Heart Rate
    # beats = vitals(t1, t2, window_shift, limit, wavelet_cycle, utc_time, mpd=1, plot=0)
    # print('\nHeart Rate Information')
    # print('Minimum pulse : ', np.around(np.min(beats)))
    # print('Maximum pulse : ', np.around(np.max(beats)))
    # print('Average pulse : ', np.around(np.mean(beats)))
    # # Breathing Rate
    # beats = vitals(t1, t2, window_shift, limit, breathing, utc_time, mpd=1, plot=0)
    # print('\nRespiratory Rate Information')
    # print('Minimum breathing : ', np.around(np.min(beats)))
    # print('Maximum breathing : ', np.around(np.max(beats)))
    # print('Average breathing : ', np.around(np.mean(beats)))
    # # ==============================================================================================================
    # thresh = 0.3
    # events = apnea_events(breathing, utc_time, thresh=thresh)
    # # ==============================================================================================================
    # # Plot Vitals Example
    # t1, t2 = 2500, 2500 * 2
    # data_subplot(data_stream, movement, breathing, wavelet_cycle, t1, t2)
    # # ==============================================================================================================
    print('\nEnd processing ...')
if __name__ == '__main__':
    main()
