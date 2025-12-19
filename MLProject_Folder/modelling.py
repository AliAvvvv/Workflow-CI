import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load Data
df = pd.read_csv('heart_disease_preprocessed.csv')
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Aktifkan Autologging (Wajib Kriteria CI Advanced)
mlflow.autolog()

# 3. Proses Training
# Tidak perlu mlflow.start_run() karena sudah dihandle oleh 'mlflow run'
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# --- ARTEFAK TAMBAHAN (Syarat Advanced) ---
# Simpan Grafik Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('CI Confusion Matrix - Ali Assegaf')
plt.savefig("ci_confusion_matrix.png")
mlflow.log_artifact("ci_confusion_matrix.png")

# Simpan File CSV Prediksi
pd.DataFrame(y_pred).to_csv("predictions.csv", index=False)
mlflow.log_artifact("predictions.csv")

print(f"Training CI Berhasil! Akurasi: {acc:.4f}")