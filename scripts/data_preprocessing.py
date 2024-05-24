# File: /scripts/data_preprocessing.py
# This file will contain the code for data preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load datasets
diabetes_data = pd.read_csv('data/diabetes.csv')
heart_data = pd.read_csv('data/heart.csv')

# Handle missing values (if any)
heart_data.replace('?', np.nan, inplace=True)
heart_data.dropna(inplace=True)
heart_data['ca'] = heart_data['ca'].astype(float)
heart_data['thal'] = heart_data['thal'].astype(float)

# Split features and targets
X_heart = heart_data.drop('num', axis=1)
y_heart = heart_data['num']
X_diabetes = diabetes_data.drop('Outcome', axis=1)
y_diabetes = diabetes_data['Outcome']

# Split data into training and validation sets
X_heart_train, X_heart_val, y_heart_train, y_heart_val = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)
X_diabetes_train, X_diabetes_val, y_diabetes_train, y_diabetes_val = train_test_split(X_diabetes, y_diabetes, test_size=0.2, random_state=42)

# Standardize the features
scaler_heart = StandardScaler()
X_heart_train = scaler_heart.fit_transform(X_heart_train)
X_heart_val = scaler_heart.transform(X_heart_val)

scaler_diabetes = StandardScaler()
X_diabetes_train = scaler_diabetes.fit_transform(X_diabetes_train)
X_diabetes_val = scaler_diabetes.transform(X_diabetes_val)

# Save preprocessed data
np.save('data/X_heart_train.npy', X_heart_train)
np.save('data/X_heart_val.npy', X_heart_val)
np.save('data/y_heart_train.npy', y_heart_train)
np.save('data/y_heart_val.npy', y_heart_val)

np.save('data/X_diabetes_train.npy', X_diabetes_train)
np.save('data/X_diabetes_val.npy', X_diabetes_val)
np.save('data/y_diabetes_train.npy', y_diabetes_train)
np.save('data/y_diabetes_val.npy', y_diabetes_val)
