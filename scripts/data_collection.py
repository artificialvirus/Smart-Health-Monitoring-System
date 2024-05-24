# File: /scripts/data_collection.py
# This file will contain the code for data collection
import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Load diabetes dataset from local file
diabetes_data = pd.read_csv('data/diabetes.csv')

# Fetch heart disease dataset from UCI repository
heart_data = fetch_ucirepo(id=45)
heart_data.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']

# Convert UCI data to DataFrame
X_heart = heart_data.data.features
y_heart = heart_data.data.targets
heart_df = pd.concat([X_heart, y_heart], axis=1)

# Save datasets to CSV files (if needed for further processing)
diabetes_data.to_csv('data/diabetes.csv', index=False)
heart_df.to_csv('data/heart.csv', index=False)
