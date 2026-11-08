import sys, json
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("diabetes-risk-project")

MODEL_NAME = "diabetes-risk-model"
ACCURACY_THRESHOLD = 0.75

model_type = sys.argv[1] if len(sys.argv) > 1 else "logistic"
test_size = 0.2

data = pd.read_csv("diabetes_risk.csv")
print("Loaded", len(data), "rows")

X = data[["age", "bmi", "glucose"]]
y = data["high_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)
print("Training on", len(X_train), "rows, testing on", len(X_test), "rows")

with mlflow.start_run():
    if model_type == "tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_type == "forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Model: {model_type} | Accuracy: {acc:.3f}")

    mlflow.log_param("model_type", model_type)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("rows_in_dataset", len(data))
    mlflow.log_metric("accuracy", acc)

    with open("metrics.json", "w") as f:
        json.dump({"accuracy": acc}, f)

    model_info = mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name=MODEL_NAME,
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )
    joblib.dump(model, "model.pkl")
    print("Model saved to model.pkl and recorded in MLflow")

    if acc >= ACCURACY_THRESHOLD:
        client = MlflowClient()
        client.set_registered_model_alias(
            MODEL_NAME, "champion", model_info.registered_model_version
        )
        print(f"SUCCESS: version {model_info.registered_model_version} promoted to 'champion'")
    else:
        print(f"NOT promoted: accuracy {acc:.3f} < threshold {ACCURACY_THRESHOLD}")