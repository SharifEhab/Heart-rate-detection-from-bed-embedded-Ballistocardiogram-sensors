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
WIN_SEC = 1 * 60  # 1 minutes
WIN_SAMPLES = int(WIN_SEC * 50)  # 3000 samples at 50 Hz
SEGMENT_SEC = 5  # 5 seconds
SEGMENT_SAMPLES = int(SEGMENT_SEC * 50)  # 250 samples
def main():
    # Main program starts here
    print('\nstart processing ...')
    paired = process.get_bcg_rr_pairs(DATA_ROOT)
    results = []
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

        # Process each clean window for heart rate
        for window in clean_windows:
            bcg_window = window['bcg']
            t_bcg_window = window['t_bcg']
            rr_hr_window = window['rr_hr']

            # Bandpass filtering
            filtered_bcg = band_pass_filtering(bcg_window, fs=50.0, filter_type='bcg')

            # MODWT
            w = modwt(filtered_bcg, 'bior3.9', 4)
            dc = modwtmra(w, 'bior3.9')
            wavelet_cycle = dc[4]  # Detail at level 4

            # Peak detection and heart rate
            mpd = 15  # Minimum peak distance (0.3s at 50 Hz)
            rate, indices = compute_rate(wavelet_cycle, t_bcg_window, mpd=mpd)

            # Reference heart rate
            hr_ref = np.mean(rr_hr_window) if len(rr_hr_window) > 0 else np.nan

            # Store results
            if rate > 0 and not np.isnan(hr_ref):
                results.append({
                    'subj': subj,
                    'date': date,
                    't_start': window['t_start'],
                    'hr_bcg': rate,
                    'hr_ref': hr_ref
                }) 
        
        print("###################################################")
    # Compute error metrics
    hr_bcg_all = [r['hr_bcg'] for r in results if not np.isnan(r['hr_bcg']) and not np.isnan(r['hr_ref'])]
    hr_ref_all = [r['hr_ref'] for r in results if not np.isnan(r['hr_bcg']) and not np.isnan(r['hr_ref'])]
    
        #########################################################################
    if len(hr_bcg_all) > 0:
        mae = np.mean(np.abs(np.array(hr_bcg_all) - np.array(hr_ref_all)))
        rmse = np.sqrt(np.mean((np.array(hr_bcg_all) - np.array(hr_ref_all))**2))
        mape = np.mean(np.abs((np.array(hr_bcg_all) - np.array(hr_ref_all)) / np.array(hr_ref_all))) * 100
        print(f"Error Metrics:")
        print(f"MAE: {mae:.2f} bpm")
        print(f"RMSE: {rmse:.2f} bpm")
        print(f"MAPE: {mape:.2f}%")
        
        # Generate plots
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
        plt.title('Bland-Altman Plot')
        plt.savefig(f'results/bland_altman_{subj}_{date}.png')
        plt.close()
        
        # Scatter plot
        slope, intercept, r_value, p_value, std_err = linregress(hr_ref_all, hr_bcg_all)
        plt.figure()
        plt.scatter(hr_ref_all, hr_bcg_all)
        plt.plot(hr_ref_all, intercept + slope * np.array(hr_ref_all), 'r')
        plt.xlabel('Reference HR (bpm)')
        plt.ylabel('BCG HR (bpm)')
        plt.title(f'Correlation: r={r_value:.2f}')
        plt.savefig(f'results/scatter_{subj}_{date}.png')
        plt.close()
        
        # Boxplot
        errors = np.array(hr_bcg_all) - np.array(hr_ref_all)
        plt.figure()
        plt.boxplot(errors)
        plt.ylabel('Error (BCG - Ref) (bpm)')
        plt.title('Boxplot of HR Errors')
        plt.savefig(f'results/boxplot_{subj}_{date}.png')
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
