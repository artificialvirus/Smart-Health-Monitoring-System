# File: /scripts/model_evaluation.py
# This file will contain the code for evaluating the model
import numpy as np
from sklearn.metrics import classification_report
import joblib

# Load models
heart_model = joblib.load('models/heart_model.pkl')
diabetes_model = joblib.load('models/diabetes_model.pkl')

# Load validation data
X_heart_val = np.load('data/X_heart_val.npy')
y_heart_val = np.load('data/y_heart_val.npy')

X_diabetes_val = np.load('data/X_diabetes_val.npy')
y_diabetes_val = np.load('data/y_diabetes_val.npy')

# Evaluate Heart Disease Model
y_heart_pred = heart_model.predict(X_heart_val)
print("Heart Disease Model Evaluation\n", classification_report(y_heart_val, y_heart_pred))

# Evaluate Diabetes Model
y_diabetes_pred = diabetes_model.predict(X_diabetes_val)
print("Diabetes Model Evaluation\n", classification_report(y_diabetes_val, y_diabetes_pred))
