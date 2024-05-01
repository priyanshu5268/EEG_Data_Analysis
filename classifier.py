import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Binarizer
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Load EEG data
filename = "D:\\data\\Data2\\s00.csv"
df = pd.read_csv(filename)
X = df.iloc[:, 1:].values  # Assuming first column is not features data
y = df.iloc[:, 0].values

# Preprocessing: Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Binarize the target variable
threshold = 0.5  # Choose an appropriate threshold
binarizer = Binarizer(threshold=threshold)
y_binarized = binarizer.fit_transform(y.reshape(-1, 1)).flatten()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binarized, test_size=0.2, random_state=42)

# CNN Model
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Reshape data for CNN
X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Train CNN model
model.fit(X_train_cnn, y_train, epochs=10, batch_size=32, verbose=1)

# Predictions
y_pred_cnn_probs = model.predict(X_test_cnn)
y_pred_cnn = (y_pred_cnn_probs > 0.5).astype(int).flatten()

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# SVM
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# Evaluation
def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return acc, roc_auc, recall, cm

print("CNN:")
acc_cnn, roc_auc_cnn, recall_cnn, cm_cnn = evaluate(y_test, y_pred_cnn)
print("Accuracy:", acc_cnn)
print("ROC AUC:", roc_auc_cnn)
print("Recall:", recall_cnn)
print("Confusion Matrix:")
print(cm_cnn)

print("KNN:")
acc_knn, roc_auc_knn, recall_knn, cm_knn = evaluate(y_test, y_pred_knn)
print("Accuracy:", acc_knn)
print("ROC AUC:", roc_auc_knn)
print("Recall:", recall_knn)
print("Confusion Matrix:")
print(cm_knn)

print("SVM:")
acc_svm, roc_auc_svm, recall_svm, cm_svm = evaluate(y_test, y_pred_svm)
print("Accuracy:", acc_svm)
print("ROC AUC:", roc_auc_svm)
print("Recall:", recall_svm)
print("Confusion Matrix:")
print(cm_svm)

print("Naive Bayes:")
acc_nb, roc_auc_nb, recall_nb, cm_nb = evaluate(y_test, y_pred_nb)
print("Accuracy:", acc_nb)
print("ROC AUC:", roc_auc_nb)
print("Recall:", recall_nb)
print("Confusion Matrix:")
print(cm_nb)
