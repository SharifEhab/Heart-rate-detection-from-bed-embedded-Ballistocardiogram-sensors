# Import required libraries
import math
import os

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
# ======================================================================================================================

DATA_ROOT = 'D:/Data Analytics/projectrepo/dataset/data'
TARGET_F_SAMPLE = 50 # 50 Hz
WIN_SEC = 6 * 60  # 6 minutes
WIN_SAMPLES = int(WIN_SEC * 50)  # 18000 samples at 50 Hz
SEGMENT_SEC = 5  # 5 seconds
SEGMENT_SAMPLES = int(SEGMENT_SEC * 50)  # 250 samples
def main():
    # Main program starts here
    print('\nstart processing ...')
    paired = process.get_bcg_rr_pairs(DATA_ROOT)
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


        # Windowing ################################
        # will be placed in a function inshaa'Allah
        num_windows = len(bcg_sync) // WIN_SAMPLES
        clean_windows = []
        for k in range(num_windows):
            start_idx = k * WIN_SAMPLES
            end_idx = start_idx + WIN_SAMPLES
            
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
        
        print(f"{subj} {date}: Found {len(clean_windows)} clean windows")
        print("###################################################")
        #########################################################################


    # After windowing, we have a list of clean bcg windows with no movements for each subject and date and the corresponding RR beats
    # Now we have multiple steps to follow:
    # 1. For each clean bcg window, we need to apply the band pass filter to extract the bcg signal
    # 2. Then we need to apply the wavelet transform function to extract the 4th level wavelet coefficients
    # 3. Then extract j peaks and compute average of window size 
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
