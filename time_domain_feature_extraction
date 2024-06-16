import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Function to calculate statistical features
def calculate_time_domain_features(signal):
    mean = np.mean(signal)
    variance = np.var(signal)
    std_dev = np.std(signal)
    skewness = pd.Series(signal.flatten()).skew()
    kurtosis = pd.Series(signal.flatten()).kurtosis()
    peak_to_peak_amp = np.ptp(signal)
    rms = np.sqrt(np.mean(signal**2))
    return [mean, variance, std_dev, skewness, kurtosis, peak_to_peak_amp, rms]

# Load data
data_files = ["D:\\data\\Data2\\s00.csv", "D:\\data\\Data2\\s01.csv", "D:\\data\\Data2\\s02.csv", "D:\\data\\Data2\\s03.csv",
              "D:\\data\\Data2\\s04.csv", "D:\\data\\Data2\\s05.csv", "D:\\data\\Data2\\s06.csv", "D:\\data\\Data2\\s07.csv",
              "D:\\data\\Data2\\s08.csv", "D:\\data\\Data2\\s09.csv", "D:\\data\\Data2\\s10.csv", "D:\\data\\Data2\\s11.csv",
              "D:\\data\\Data2\\s12.csv", "D:\\data\\Data2\\s13.csv", "D:\\data\\Data2\\s14.csv", "D:\\data\\Data2\\s15.csv",
              "D:\\data\\Data2\\s16.csv", "D:\\data\\Data2\\s17.csv", "D:\\data\\Data2\\s18.csv", "D:\\data\\Data2\\s19.csv",
              "D:\\data\\Data2\\s20.csv", "D:\\data\\Data2\\s21.csv", "D:\\data\\Data2\\s22.csv", "D:\\data\\Data2\\s23.csv",
              "D:\\data\\Data2\\s24.csv", "D:\\data\\Data2\\s25.csv", "D:\\data\\Data2\\s26.csv", "D:\\data\\Data2\\s27.csv",
              "D:\\data\\Data2\\s28.csv", "D:\\data\\Data2\\s29.csv", "D:\\data\\Data2\\s30.csv", "D:\\data\\Data2\\s31.csv",
              "D:\\data\\Data2\\s32.csv", "D:\\data\\Data2\\s33.csv", "D:\\data\\Data2\\s34.csv", "D:\\data\\Data2\\s35.csv"]

# Load and preprocess the dataset for each subject
data = []
for file in data_files:
    df = pd.read_csv(file, header=None).transpose().to_numpy()
    data.append(df)

# Convert data to numpy array and calculate features
dataset = np.array(data)
features = []

for sample in dataset:
    sample_features = calculate_time_domain_features(sample.flatten())
    features.append(sample_features)

# Convert features list to a DataFrame for better visualization
features_df = pd.DataFrame(features, columns=['Mean', 'Variance', 'Std Dev', 'Skewness', 'Kurtosis', 'Peak-to-Peak Amplitude', 'RMS'])

# Save the features DataFrame to a CSV file
features_df.to_csv("D:\\data\\Data2\\features.csv", index=False)

# Print the features DataFrame
print(features_df)

# Calculate the overall combined features
overall_features = calculate_time_domain_features(np.concatenate(dataset).flatten())
overall_features_df = pd.DataFrame([overall_features], columns=['Mean', 'Variance', 'Std Dev', 'Skewness', 'Kurtosis', 'Peak-to-Peak Amplitude', 'RMS'])

# Save the overall features to a CSV file
overall_features_df.to_csv("D:\\data\\Data2\\overall_features.csv", index=False)

# Print the overall features DataFrame
print(overall_features_df)

# Calculate the average features
average_features = features_df.mean(axis=0)
average_features_df = pd.DataFrame([average_features], columns=features_df.columns)

# Save the average features to a CSV file
average_features_df.to_csv("D:\\data\\Data2\\average_features.csv", index=False)

# Print the average features DataFrame
print(average_features_df)

# Function to save scatter plot with different colors
def save_scatter_plot(data, title, filename, color):
    plt.figure()
    plt.scatter(range(len(data)), data, color=color)
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel(title)
    plt.savefig(filename)
    plt.close()

# Colors for plots
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']

# Plotting and saving the features
save_scatter_plot(features_df['Mean'], 'Mean', "D:\\data\\Data2\\mean_plot.png", colors[0])
save_scatter_plot(features_df['Variance'], 'Variance', "D:\\data\\Data2\\variance_plot.png", colors[1])
save_scatter_plot(features_df['Std Dev'], 'Standard Deviation', "D:\\data\\Data2\\std_dev_plot.png", colors[2])
save_scatter_plot(features_df['Skewness'], 'Skewness', "D:\\data\\Data2\\skewness_plot.png", colors[3])
save_scatter_plot(features_df['Kurtosis'], 'Kurtosis', "D:\\data\\Data2\\kurtosis_plot.png", colors[4])
save_scatter_plot(features_df['Peak-to-Peak Amplitude'], 'Peak-to-Peak Amplitude', "D:\\data\\Data2\\peak_to_peak_plot.png", colors[5])
save_scatter_plot(features_df['RMS'], 'RMS', "D:\\data\\Data2\\rms_plot.png", colors[6])

# Plotting and saving the average features
save_scatter_plot(average_features_df.T[0], 'Average Features', "D:\\data\\Data2\\average_features_plot.png", 'blue')

# Plotting and saving the overall features
save_scatter_plot(overall_features_df.T[0], 'Overall Features', "D:\\data\\Data2\\overall_features_plot.png", 'blue')
