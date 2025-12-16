import pandas as pd
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

mlflow.set_experiment("CI_CD_Automation")

print("Starting training script...")

# Load Data
try:
    data_path = 'TelcoCustomerChurn_preprocessing'
    if not os.path.exists(data_path):
        data_path = 'data'
    
    X_train = pd.read_csv(os.path.join(data_path, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(data_path, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv')).iloc[:, 0]
    y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv')).iloc[:, 0]
except FileNotFoundError:
    print("Dataset not found.")
    exit(1)

with mlflow.start_run() as run:
    
    # Train model
    n_estimators = 20
    max_depth = 10
    
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(X_train, y_train)

    # Save Model (Local then Upload)
    # This ensures the model folder structure is correct before uploading
    local_model_path = "temp_model_folder"
    if os.path.exists(local_model_path):
        shutil.rmtree(local_model_path)
    
    mlflow.sklearn.save_model(rf, local_model_path)
    mlflow.log_artifacts(local_model_path, artifact_path="model")
    
    print("Model logged to MLflow.")

    # Evaluate
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", acc)
    
    # Generate Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    
    cm_file = "confusion_matrix.png"
    plt.savefig(cm_file)
    plt.close()
    
    mlflow.log_artifact(cm_file)
    
    # Cleanup
    if os.path.exists(cm_file):
        os.remove(cm_file)
    if os.path.exists(local_model_path):
        shutil.rmtree(local_model_path)
        
    print("Training finished.")