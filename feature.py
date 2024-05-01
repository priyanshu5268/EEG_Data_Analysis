import os
import numpy as np
import pandas as pd
from scipy.signal import welch
import matplotlib.pyplot as plt

# Function to extract time domain features
def extract_time_domain_features(signal):
    features = {}
    features['mean'] = np.mean(signal)
    features['std_dev'] = np.std(signal)
    features['max'] = np.max(signal)
    features['min'] = np.min(signal)
    features['median'] = np.median(signal)
    features['rms'] = np.sqrt(np.mean(signal**2))
    return features

# Function to extract frequency domain features
def extract_frequency_domain_features(signal, fs):
    f, Pxx = welch(signal, fs, nperseg=len(signal))
    max_power_freq_index = np.argmax(Pxx)
    features = {}
    features['max_power_freq'] = f[max_power_freq_index]
    features['total_power'] = np.sum(Pxx)
    features['peak_frequency'] = f[np.argmax(Pxx)]
    features['mean_frequency'] = np.sum(f * Pxx) / np.sum(Pxx)
    features['median_frequency'] = np.median(f)
    features['spectral_centroid'] = np.sum(f * Pxx) / np.sum(Pxx)
    features['spectral_entropy'] = -np.sum(Pxx * np.log2(Pxx + 1e-12))
    features['psd'] = Pxx
    return features

# Read EEG data from CSV file
def read_eeg_csv(filename):
    df = pd.read_csv(filename)
    # Assuming the first column contains timestamps and the second column contains EEG signal
    timestamps = df.iloc[:, 0].values
    eeg_signal = df.iloc[:, 1].values
    return timestamps, eeg_signal

# Function to save features to CSV file
def save_features_to_csv(filename, features_dict):
    if not os.path.exists(filename):
        pd.DataFrame(columns=features_dict.keys()).to_csv(filename, index=False)
    df = pd.read_csv(filename)
    df = pd.concat([df, pd.DataFrame([features_dict])], ignore_index=True)
    df.to_csv(filename, index=False)

# Example usage
filename = "D:\\data\\Data2\\s00.csv"
timestamps, eeg_signal = read_eeg_csv(filename)

# Sampling frequency (Assuming uniform sampling)
fs = 1 / (timestamps[1] - timestamps[0])

# Extracting features
time_domain_features = extract_time_domain_features(eeg_signal)
frequency_domain_features = extract_frequency_domain_features(eeg_signal, fs)

# Plotting PSD separately
plt.subplot(2,1,2)
plt.plot(frequency_domain_features['psd'], color='green')
plt.title('Power Spectral Density (PSD)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')

plt.tight_layout()
plt.show()