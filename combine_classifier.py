import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

# Load and preprocess the dataset for each subject
data_files = ["D:\\data\\Data2\\s00.csv", "D:\\data\\Data2\\s01.csv"]  # Add paths for all subjects
data = []
labels = []

for file in data_files:
    df = pd.read_csv(file, header=None).transpose().to_numpy()
    data.append(df)
    labels.extend([int(file.split("\\")[-1][1:3])] * len(df))

# Combine data and labels into numpy arrays
X = np.concatenate(data, axis=0)
y = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)

# Train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(X_train, y_train)

# Combine predictions of Decision Tree and Random Forest classifiers
y_pred_dt = dt_classifier.predict(X_test)
y_pred_rf = rf_classifier.predict(X_test)
combined_predictions = (y_pred_dt + y_pred_rf) >= 1

# Calculate accuracy
accuracy = accuracy_score(y_test, combined_predictions)
print("Combined Classifier Accuracy:", accuracy)

# Calculate probabilities for each class
y_proba_dt = dt_classifier.predict_proba(X_test)[:, 1]
y_proba_rf = rf_classifier.predict_proba(X_test)[:, 1]
combined_proba = (y_proba_dt + y_proba_rf) / 2

# Compute ROC curve and AUC for combined classifier
fpr, tpr, _ = roc_curve(y_test, combined_proba)
roc_auc = auc(fpr, tpr)

# Print AUC
print("AUC:", roc_auc)

# Calculate precision, recall, f1-score, and support
report = classification_report(y_test, combined_predictions)

# Print precision, recall, f1-score, and support
print("Classification Report:")
print(report)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
