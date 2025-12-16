import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

mlflow.set_experiment("CI_CD_Automation")

print("Memulai Training Model untuk CI/CD...")

# 1. Load Data
try:
    # Membaca data yang sudah bersih (Preprocessed)
    X_train = pd.read_csv('TelcoCustomerChurn_preprocessing/X_train.csv')
    X_test = pd.read_csv('TelcoCustomerChurn_preprocessing/X_test.csv')
    y_train = pd.read_csv('TelcoCustomerChurn_preprocessing/y_train.csv').iloc[:, 0] # Flatten
    y_test = pd.read_csv('TelcoCustomerChurn_preprocessing/y_test.csv').iloc[:, 0]   # Flatten
except FileNotFoundError:
    print("Error: Dataset tidak ditemukan di folder ini.")
    exit(1)

with mlflow.start_run():
    # Hyperparameters
    n_estimators = 20
    max_depth = 10
    
    # Train model
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")

    # Log parameters and metrics
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", acc)
    
    # Create confusion matrix artifact
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    cm_file = "confusion_matrix.png"
    plt.savefig(cm_file)
    plt.close()
    
    # Upload artifact
    mlflow.log_artifact(cm_file)
    
    # Cleanup local file
    if os.path.exists(cm_file):
        os.remove(cm_file)
        
    # Log model to MLflow
    mlflow.sklearn.log_model(rf, "model")
    
    print("Run completed. Model and artifacts logged.")