# Heart Rate Estimation from Ballistocardiogram (BCG) Signals

## 📚 Course Project: Data Analytics

This project focuses on **heart rate (HR) estimation** from ballistocardiogram (BCG) signals by detecting **J-peaks** and comparing the results against reference ECG-based heart rate values.

The project is based on the following resources:
- [Dataset paper](https://www.nature.com/articles/s41597-024-03950-5)
- [Dataset download](https://doi.org/10.6084/m9.figshare.26013157)
- [Initial codebase for J-peak detection](https://codeocean.com/capsule/1398208/tree)

---
📈 Project Overview
-------------------

This project focuses on estimating heart rates from ballistocardiogram (BCG) signals collected using piezoelectric film sensors and comparing them to reference heart rate (RR) data. The BCG signals, resampled at 50 Hz, are processed to detect J-peaks, which are used to estimate heart rates. The pipeline includes preprocessing, movement detection, clean window extraction, and heart rate computation, with performance evaluated against reference RR data using error metrics and visualizations.

### Objectives

*   **Preprocess** BCG datasets by synchronizing with reference RR data and removing movement artifacts.
    
*   **Extract** clean 2-minute windows free of movement for heart rate estimation.
    
*   **Estimate heart rate (HR)** from BCG signals using J-peak detection.
    
*   **Compare** estimated HR against reference HR values from RR data.
    
*   **Evaluate** performance using statistical metrics and diagnostic plots.
    

🛠️ Methodology
---------------

The methodology involves a multi-step signal processing pipeline to extract and validate heart rates from BCG signals, ensuring robust comparison with reference data.

### 1\. Data Preprocessing

*   **Dataset**: BCG signals (140 Hz) and corresponding RR reference files from multiple subjects across various nights, stored as CSV files.
    
*   **Synchronization**: Resample BCG to 50 Hz, then align BCG and RR signals by identifying overlapping time intervals to ensure temporal consistency.
    
*   **Movement Detection**: Apply a movement detection algorithm to create a binary mask identifying no-movement periods in the BCG signal, ensuring only stable data is used for analysis.
    

### 2\. Clean Window Extraction

*   Divide synchronized BCG signals into 1-minute windows (3,000 samples at 50 Hz).
    
*   Extract windows where all samples indicate no movement, based on the binary mask, to ensure high-quality data for heart rate estimation.
    

### 3\. J-Peak Detection and Heart Rate Estimation

*   **Bandpass Filtering**: Apply a Chebyshev bandpass filter (2.5–5.0 Hz) to isolate heartbeat-related components in the BCG signal.
    
*   **Wavelet Transform**: Use the Maximal Overlap Discrete Wavelet Transform (MODWT) with the 'bior3.9' wavelet up to level 4, extracting the 4th detail component (1.5625–3.125 Hz) to focus on heart rate frequencies.
    
*   **Peak Detection**: Detect J-peaks in the wavelet signal using the detect\_peaks function with a minimum peak distance of 15 samples (0.3 seconds at 50 Hz, supporting up to 200 bpm).
    
*   **Heart Rate Calculation**: Compute the average inter-peak interval (in seconds) and convert to beats per minute (bpm) as 60 / mean\_interval.
    

### 4\. Reference Heart Rate Extraction

*   Extract reference heart rates by averaging the per-beat heart rate values provided in the RR data for each clean window.
    

### 5\. Evaluation Metrics

Performance is assessed by comparing estimated BCG heart rates to reference RR heart rates across all clean windows from all subjects:

*   **Mean Absolute Error (MAE)**: Average absolute difference between estimated and reference heart rates.
    
*   **Root Mean Square Error (RMSE)**: Square root of the mean squared difference, emphasizing larger errors.
    
*   **Mean Absolute Percentage Error (MAPE)**: Average percentage difference, normalized by reference heart rates.
    

### 6\. Visualization

Diagnostic plots are generated to evaluate agreement and correlation:

*   **Bland-Altman Plot**: Displays the difference between BCG and RR heart rates against their average, highlighting bias and limits of agreement.
    
*   **Correlation Scatter Plot**: Shows BCG heart rates versus RR heart rates with a linear regression line and Pearson correlation coefficient.
    
*   **Boxplot**: Summarizes the distribution of errors (BCG HR – RR HR) to identify outliers and variability.
    

📊 Current Results
------------------

The pipeline processes BCG and RR data from multiple subjects, extracting clean windows and computing heart rates. Current challenges include high error metrics (e.g., MAE ~13.37 bpm, RMSE ~16.74 bpm, MAPE ~21.43%), indicating issues with wavelet component selection or filtering. Ongoing improvements involve adjusting the wavelet level and bandpass filter range to better capture heart rate frequencies.

🚀 Future Work
--------------

*   Optimize J-peak detection by testing different wavelet levels (e.g., combining d3 and d4) and filter ranges (e.g., 0.5–10 Hz).
    
*   Validate results with visual inspection of signals and peaks.
    
*   Experiment with different window sizes (e.g., 3 or 10 minutes) to balance data quantity and quality.
    
*   Achieve error metrics closer to industry standards (MAE ≤ 5 bpm, RMSE ≤ 2.59 bpm, per AAMI recommendations).

## 📚 References

1. Sadek, Ibrahim, and Bessam Abdulrazak. “A comparison of three heart rate detection algorithms over ballistocardiogram signals.” *Biomedical Signal Processing and Control* (2021).

2. Sadek, Ibrahim, et al. “A new approach for detecting sleep apnea using a contactless bed sensor: Comparison study.” *Journal of Medical Internet Research* (2020).

---
