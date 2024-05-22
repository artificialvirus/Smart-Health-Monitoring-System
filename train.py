# File: /train.py
# This file will contain the code for model training
import subprocess

# Run data collection
subprocess.run(['python', 'scripts/data_collection.py'])

# Run data preprocessing
subprocess.run(['python', 'scripts/data_preprocessing.py'])

# Run model training
subprocess.run(['python', 'scripts/model_training.py'])
