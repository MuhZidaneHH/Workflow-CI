import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

# 2. Training
# Kita pakai settingan ringan agar hemat waktu CI
rf = RandomForestClassifier(n_estimators=10, random_state=42)
rf.fit(X_train, y_train)

# 3. Evaluasi
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc}")

# 4. Save Model (Pickle)
with open('model.pkl', 'wb') as f:
    pickle.dump(rf, f)

print("Model berhasil disimpan sebagai model.pkl")