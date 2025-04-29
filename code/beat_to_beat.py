import numpy as np

from detect_peaks import detect_peaks


def compute_rate(beats, time, mpd):

    indices = detect_peaks(beats, mpd=mpd)

    if len(indices) > 1:
        peak_to_peak = []
        for i in range(0, indices.size - 1):
            peak_to_peak = np.append(peak_to_peak, time[indices[i + 1]] - time[indices[i]])
        mean_heart_rate = np.average(peak_to_peak, axis=0)
        bpm_avg = 1000 * (60 / mean_heart_rate)
        return np.round(bpm_avg, decimals=2), indices
    else:
        return 0.0, 0.0


# def compute_rate(beats,mpd,fs):

#     indices = detect_peaks(beats, mpd=mpd)

#     if len(indices) > 1:
#         diff_sample = indices[-1] - indices[0] + 1
#         t_N = diff_sample / fs
#         heartRate = (len(indices) - 1) / t_N * 60
#         return heartRate, indices
#     else:
#         return 0.0, 0.0
    
# def heart_rate( peaks:np.array, sig_length:int, t_window_sec = 5 , fs = 50) -> np.array:
#     """
#     Calculate heart rate from peaks given window size.
    
#     :param peaks: array of peak indexes
#     :param sig_length: length of the signal
#     :param t_window_sec: time window in seconds
#     :param fs: sampling frequency

#     :return: heart rate array
#     """
#     heartRate = np.array([]).astype(np.float32)
#     t_window_n = t_window_sec * fs
#     # loop over all peaks and count how many peaks are in the window
#     for i in range(0, sig_length+1, round(t_window_n)):
#         # calculate the time window in seconds
#         t_window_sec  = t_window_n / fs
#         # calculate the heart rate from counting peak indexes that fall in the window
#         peaks_in_window = [peak for peak in peaks if peak > i and peak < i + t_window_n]
#         peak_count = 0
#         for peak in peaks:
#             if peak > i and peak < i + t_window_n:
#                 peak_count += 1
#         # calculate heart rate

#         heartRate = np.append(heartRate, peak_count / t_window_sec * 60)

#     return heartRate
    