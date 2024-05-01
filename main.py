import os
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
import matplotlib.pyplot as plt

# Function to extract time-domain features
def extract_time_domain_features(data):
    features = {
        'mean': np.mean(data),
        'variance': np.var(data),
        'std_deviation': np.std(data),
        'skewness': pd.Series(data).skew(),
        'kurtosis': pd.Series(data).kurtosis(),
        'peak_to_peak_amplitude': np.ptp(data),
        'rms': np.sqrt(np.mean(np.square(data)))
    }
    return features

# Function to extract frequency-domain features
def extract_frequency_domain_features(data, fs):
    f, psd = signal.welch(data, fs=fs)
    peak_freq = f[np.argmax(psd)]
    spectral_entropy = -np.sum(psd * np.log2(psd))
    features = {
        'peak_frequency': peak_freq,
        'spectral_entropy': spectral_entropy
    }
    return features

# Function to extract features from EEG signals
def extract_features(eeg_data, fs):
    time_domain_features = extract_time_domain_features(eeg_data)
    frequency_domain_features = extract_frequency_domain_features(eeg_data, fs)
    # Combine all features into a single dictionary
    all_features = {**time_domain_features, **frequency_domain_features}
    return all_features

# Function to save features to CSV file
def save_features_to_csv(features, file_name):
    if os.path.exists(file_name):
        # File exists, load existing data and append new features
        existing_data = pd.read_csv(file_name)
        new_data = pd.DataFrame(features)
        updated_data = pd.concat([existing_data, new_data], axis=0, ignore_index=True)
        updated_data.to_csv(file_name, index=False)
    else:
        # File doesn't exist, create new file and save features
        data = pd.DataFrame(features)
        data.to_csv(file_name, index=False)

# Function to train CNN classifier
def train_cnn(X_train, y_train, input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    return model

# Function to evaluate classifiers
def evaluate_classifiers(classifiers, X_test, y_test):
    knn_classifier, svm_classifier, nb_classifier, cnn_model = classifiers

    # Predict labels for traditional classifiers
    knn_pred = knn_classifier.predict(X_test)
    svm_pred = svm_classifier.predict(X_test)
    nb_pred = nb_classifier.predict(X_test)

    # Predict labels for CNN classifier
    cnn_pred_prob = cnn_model.predict(X_test)
    cnn_pred = np.argmax(cnn_pred_prob, axis=1)

    # Compute evaluation metrics
    knn_accuracy = accuracy_score(y_test, knn_pred)
    knn_precision = precision_score(y_test, knn_pred)
    knn_recall = recall_score(y_test, knn_pred)
    knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_pred)
    knn_auc = auc(knn_fpr, knn_tpr)
    knn_cm = confusion_matrix(y_test, knn_pred)

    svm_accuracy = accuracy_score(y_test, svm_pred)
    svm_precision = precision_score(y_test, svm_pred)
    svm_recall = recall_score(y_test, svm_pred)
    svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_pred)
    svm_auc = auc(svm_fpr, svm_tpr)
    svm_cm = confusion_matrix(y_test, svm_pred)

    nb_accuracy = accuracy_score(y_test, nb_pred)
    nb_precision = precision_score(y_test, nb_pred)
    nb_recall = recall_score(y_test, nb_pred)
    nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_pred)
    nb_auc = auc(nb_fpr, nb_tpr)
    nb_cm = confusion_matrix(y_test, nb_pred)

    cnn_accuracy = accuracy_score(y_test, cnn_pred)
    cnn_precision = precision_score(y_test, cnn_pred)
    cnn_recall = recall_score(y_test, cnn_pred)
    cnn_fpr, cnn_tpr, _ = roc_curve(y_test, cnn_pred)
    cnn_auc = auc(cnn_fpr, cnn_tpr)
    cnn_cm = confusion_matrix(y_test, cnn_pred)

    # Print evaluation metrics
    print("k-NN Classifier:")
    print("Accuracy:", knn_accuracy)
    print("Precision:", knn_precision)
    print("Recall:", knn_recall)
    print("AUC:", knn_auc)
    print("Confusion Matrix:")
    print(knn_cm)
    print("\n")

    print("SVM Classifier:")
    print("Accuracy:", svm_accuracy)
    print("Precision:", svm_precision)
    print("Recall:", svm_recall)
    print("AUC:", svm_auc)
    print("Confusion Matrix:")
    print(svm_cm)
    print("\n")

    print("Naive Bayes Classifier:")
    print("Accuracy:", nb_accuracy)
    print("Precision:", nb_precision)
    print("Recall:", nb_recall)
    print("AUC:", nb_auc)
    print("Confusion Matrix:")
    print(nb_cm)
    print("\n")

    print("CNN Classifier:")
    print("Accuracy:", cnn_accuracy)
    print("Precision:", cnn_precision)
    print("Recall:", cnn_recall)
    print("AUC:", cnn_auc)
    print("Confusion Matrix:")
    print(cnn_cm)

    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    plt.plot(knn_fpr, knn_tpr, label='k-NN (AUC = %0.2f)' % knn_auc)
    plt.plot(svm_fpr, svm_tpr, label='SVM (AUC = %0.2f)' % svm_auc)
    plt.plot(nb_fpr, nb_tpr, label='Naive Bayes (AUC = %0.2f)' % nb_auc)
    plt.plot(cnn_fpr, cnn_tpr, label='CNN (AUC = %0.2f)' % cnn_auc)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    # Plot confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].imshow(knn_cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0, 0].set_title('k-NN Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted Label')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 1].imshow(svm_cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0, 1].set_title('SVM Confusion Matrix')
    axes[0, 1].set_xlabel('Predicted Label')
    axes[0, 1].set_ylabel('True Label')
    axes[1, 0].imshow(nb_cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 0].set_title('Naive Bayes Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted Label')
    axes[1, 0].set_ylabel('True Label')
    axes[1, 1].imshow(cnn_cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 1].set_title('CNN Confusion Matrix')
    axes[1, 1].set_xlabel('Predicted Label')
    axes[1, 1].set_ylabel('True Label')
    plt.tight_layout()
    plt.show()

# Load EEG data (replace this with your actual EEG dataset)
eeg_data = pd.read_csv("D:\\data\\Data2\\s00.csv")

# Assuming no labels provided
# X = eeg_data  # Features

# Split data into training and testing sets (80% train, 20% test)
# X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Extract features
# fs = 250  # Sample rate of EEG data in Hz
# features_train = [extract_features(data, fs) for data in X_train]
# features_test = [extract_features(data, fs) for data in X_test]

# Save features to CSV file
# save_features_to_csv(features_train, "eeg_features_train.csv")
# save_features_to_csv(features_test, "eeg_features_test.csv")

# Standardize features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# Train classifiers
# knn_classifier = KNeighborsClassifier()
# svm_classifier = SVC()
# nb_classifier = GaussianNB()

# knn_classifier.fit(X_train_scaled, y_train)
# svm_classifier.fit(X_train_scaled, y_train)
# nb_classifier.fit(X_train_scaled, y_train)

# Train CNN classifier
# input_shape = (X_train_scaled.shape[1], 1)
# cnn_model = train_cnn(X_train_scaled.reshape(X_train_scaled.shape[0], input_shape[0], input_shape[1]),
#                       y_train, input_shape=input_shape, num_classes=len(np.unique(y_train)))

# Evaluate classifiers
# evaluate_classifiers((knn_classifier, svm_classifier, nb_classifier, cnn_model), X_test_scaled, y_test)
