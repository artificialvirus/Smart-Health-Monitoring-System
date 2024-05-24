# File: /scripts/model_training.py
# This file will contain the code for training the model
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import joblib

# Ensure the models directory exists
os.makedirs('models', exist_ok=True)

# Load preprocessed data
X_heart_train = np.load('data/X_heart_train.npy')
X_heart_val = np.load('data/X_heart_val.npy')
y_heart_train = np.load('data/y_heart_train.npy')
y_heart_val = np.load('data/y_heart_val.npy')

X_diabetes_train = np.load('data/X_diabetes_train.npy')
X_diabetes_val = np.load('data/X_diabetes_val.npy')
y_diabetes_train = np.load('data/y_diabetes_train.npy')
y_diabetes_val = np.load('data/y_diabetes_val.npy')

# Define hyperparameter grid
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}

# Train Heart Disease Model
heart_model = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
heart_model.fit(X_heart_train, y_heart_train)
y_heart_pred = heart_model.predict(X_heart_val)
print("Heart Disease Model\n", classification_report(y_heart_val, y_heart_pred))
joblib.dump(heart_model, 'models/heart_model.pkl')

# Train Diabetes Model
diabetes_model = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
diabetes_model.fit(X_diabetes_train, y_diabetes_train)
y_diabetes_pred = diabetes_model.predict(X_diabetes_val)
print("Diabetes Model\n", classification_report(y_diabetes_val, y_diabetes_pred))
joblib.dump(diabetes_model, 'models/diabetes_model.pkl')
