# Heart Rate Estimation from Ballistocardiogram (BCG) Signals

## 📚 Course Project: Software Development

This project focuses on **heart rate (HR) estimation** from ballistocardiogram (BCG) signals by detecting **J-peaks** and comparing the results against reference ECG-based heart rate values.

The project is based on the following resources:
- [Dataset paper](https://www.nature.com/articles/s41597-024-03950-5)
- [Dataset download](https://doi.org/10.6084/m9.figshare.26013157)
- [Initial codebase for J-peak detection](https://codeocean.com/capsule/1398208/tree)

---

## 📈 Project Overview

### Objectives
- **Download** and preprocess BCG datasets.
- **Adapt** the provided J-peak detection code to work with the downloaded signals (re-sampling if necessary).
- **Estimate heart rate (HR)** from detected J-peaks.
- **Compare** estimated HR against reference HR values from ECG signals.

### Methodology
1. **J-Peak Detection**  
   Adapt and utilize the provided J-peak detection code to identify peaks in the BCG signals.

2. **Reference HR Extraction**  
   Use a reliable algorithm such as the [Pan Tompkins algorithm](https://pypi.org/project/py-ecg-detectors/) to extract HR from ECG signals.

3. **Evaluation Metrics**
   - **Mean Absolute Error (MAE)**
   - **Root Mean Square Error (RMSE)**
   - **Mean Absolute Percentage Error (MAPE)**

4. **Visualization**
   - **Bland-Altman Plot** between estimated HR and reference HR.
   - **Pearson Correlation Plot**.
   - **Boxplot** to summarize HR distribution comparison.

---

## 📚 References

1. Sadek, Ibrahim, and Bessam Abdulrazak. “A comparison of three heart rate detection algorithms over ballistocardiogram signals.” *Biomedical Signal Processing and Control* (2021).

2. Sadek, Ibrahim, et al. “A new approach for detecting sleep apnea using a contactless bed sensor: Comparison study.” *Journal of Medical Internet Research* (2020).

---
