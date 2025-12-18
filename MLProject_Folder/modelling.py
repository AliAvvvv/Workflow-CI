import os
import pandas as pd
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Setup Autentikasi DagsHub
# Mengambil token dari environment (GitHub Actions)
token = os.getenv("DAGSHUB_TOKEN")

if token:
    # Jika berjalan di GitHub Actions
    os.environ["MLFLOW_TRACKING_USERNAME"] = "AliAvvvv"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    mlflow.set_tracking_uri("https://dagshub.com/AliAvvvv/Heart-Disease-MLOps.mlflow")
else:
    # Jika berjalan di laptop lokal Ali
    try:
        dagshub.init(repo_owner='AliAvvvv', repo_name='Heart-Disease-MLOps', mlflow=True)
    except:
        print("Gagal inisialisasi DagsHub secara otomatis. Pastikan token tersedia.")

# 2. Load Dataset
# Mencoba mencari path file baik di GitHub maupun Lokal
data_path = "MLProject_Folder/heart_disease_preprocessed.csv"
if not os.path.exists(data_path):
    data_path = "heart_disease_preprocessed.csv"

df = pd.read_csv(data_path)
X = df.drop('target', axis=1)
y = df['target']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Training & Logging
# Gunakan nested=False untuk memastikan run baru selalu dibuat
with mlflow.start_run(run_name="CI_Advanced_AliAssegaf"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Log Metric
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log Model
    mlflow.sklearn.log_model(model, "heart_disease_model")
    
    print(f"Training selesai! Akurasi: {accuracy:.4f}")
    print("Data berhasil terkirim ke DagsHub!")