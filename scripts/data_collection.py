# File: /scripts/data_collection.py
# This file will contain the code for data collection
import pandas as pd
from ucimlrepo import fetch_ucirepo 

# Load diabetes dataset from Kaggle
diabetes_data = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/pima-indians-diabetes.csv')

# Fetch heart disease dataset from UCI repository
heart_data = fetch_ucirepo(id=45) 
heart_data.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# Convert UCI data to DataFrame
X_heart = heart_data.data.features 
y_heart = heart_data.data.targets 
heart_df = pd.concat([X_heart, y_heart], axis=1)

# Save datasets to CSV files
diabetes_data.to_csv('data/diabetes.csv', index=False)
heart_df.to_csv('data/heart.csv', index=False)
