import pandas as pd
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Inisialisasi DagsHub
dagshub.init(repo_owner='AliAvvvv', repo_name='Heart-Disease-MLOps', mlflow=True)

df = pd.read_csv('heart_disease_preprocessed.csv')
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="CI_Execution_Ali"):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    mlflow.log_metric("accuracy", model.score(X_test, y_test))
    mlflow.sklearn.log_model(model, "model")
    print("CI Training Selesai!")