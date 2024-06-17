import pandas as pd
# import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

base_path = "D:/manualCDmanagement/codes/Projects/VMs/skl algorithms/Logistic Regression/Diabetes.1/Storage"
file_name = "diabetes.csv"
file_path = os.path.join(base_path, file_name)

df = pd.read_csv(file_path) 

# data already clean

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# imbalance data, use smote
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Predict probabilities instead of classes
y_probs = model.predict_proba(X_test)[:, 1]

threshold = 0.8
y_pred_adjusted = (y_probs >= threshold).astype(int)

# Evaluate adjusted predictions
conf_matrix_adjusted = confusion_matrix(y_test, y_pred_adjusted)
class_report_adjusted = classification_report(y_test, y_pred_adjusted)
roc_auc_adjusted = roc_auc_score(y_test, y_probs)

# Print evaluation metrics
print("Confusion Matrix (Adjusted):")
print(conf_matrix_adjusted)
print("\nClassification Report (Adjusted):")
print(class_report_adjusted)
print("\nROC AUC Score (Adjusted):", roc_auc_adjusted)

# Calculate accuracy with the adjusted predictions
accuracy = accuracy_score(y_test, y_pred_adjusted)
print("Accuracy (Adjusted) with Threshold {}: {:.2f}".format(threshold, accuracy))
print("! Accuracy not a good base, using thresholds.")

