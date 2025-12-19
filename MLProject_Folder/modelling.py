import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# --- KONFIGURASI MANUAL MLFLOW (Bypass dagshub.init) ---
token = os.getenv("DAGSHUB_TOKEN")
# Ganti URL ini sesuai repositori kamu
repo_url = "https://dagshub.com/AliAvvvv/Heart-Disease-MLOps.mlflow"

if token:
    mlflow.set_tracking_uri(repo_url)
    os.environ["MLFLOW_TRACKING_USERNAME"] = "AliAvvvv"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    print("Berhasil terhubung ke DagsHub via Token!")

# 1. Load Data
df = pd.read_csv('heart_disease_preprocessed.csv')
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Autologging & Manual Logging (Syarat Advanced Repo CI)
mlflow.autolog()

with mlflow.start_run(run_name="CI_Advanced_AliAssegaf", nested=True):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # --- ARTEFAK TAMBAHAN (Syarat Advanced) ---
    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('CI Confusion Matrix - Ali Assegaf')
    plt.savefig("ci_confusion_matrix.png")
    mlflow.log_artifact("ci_confusion_matrix.png")
    
    pd.DataFrame(y_pred).to_csv("predictions.csv", index=False)
    mlflow.log_artifact("predictions.csv")
    
    print(f"Training CI Berhasil! Akurasi: {acc:.4f}")