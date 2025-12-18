import os
import pandas as pd
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Konfigurasi Autentikasi untuk GitHub Actions
token = os.getenv("MLFLOW_TRACKING_PASSWORD")
if token:
    os.environ["MLFLOW_TRACKING_TOKEN"] = token
    mlflow.set_tracking_uri("https://dagshub.com/AliAvvvv/Heart-Disease-MLOps.mlflow")
else:
    # Jika dijalankan di laptop lokal
    dagshub.init(repo_owner='AliAvvvv', repo_name='Heart-Disease-MLOps', mlflow=True)

# Memuat dataset (sesuaikan path untuk server GitHub)
data_path = "MLProject_Folder/heart_disease_preprocessed.csv"
if not os.path.exists(data_path):
    data_path = "heart_disease_preprocessed.csv"

df = pd.read_csv(data_path)
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="CI_Advanced_AliAssegaf"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Logging metrik
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Logging model
    mlflow.sklearn.log_model(model, "heart_disease_model")
    
    print(f"Training selesai! Akurasi: {accuracy}")