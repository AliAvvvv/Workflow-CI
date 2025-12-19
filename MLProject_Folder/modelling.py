import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Inisialisasi DagsHub HARUS di paling atas
dagshub.init(repo_owner='AliAvvvv', repo_name='Heart-Disease-MLOps', mlflow=True)

# 2. Load Data
df = pd.read_csv('heart_disease_preprocessed.csv')
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Autologging Wajib (Kriteria CI Advanced)
mlflow.autolog()

# Gunakan nested=True agar tidak terjadi error 'Run not found' saat dipanggil lewat mlflow run
with mlflow.start_run(run_name="CI_Advanced_AliAssegaf", nested=True):
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # --- LOGGING MANUAL ARTEFAK (Syarat Advanced untuk Repo CI) ---
    # 1. Simpan Grafik Confusion Matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('CI Confusion Matrix - Ali Assegaf')
    plt.savefig("ci_confusion_matrix.png")
    mlflow.log_artifact("ci_confusion_matrix.png")
    
    # 2. Simpan File CSV Prediksi sebagai artefak tambahan
    pd.DataFrame(y_pred).to_csv("predictions.csv", index=False)
    mlflow.log_artifact("predictions.csv")
    
    print(f"Training CI Selesai! Akurasi: {acc:.4f}")