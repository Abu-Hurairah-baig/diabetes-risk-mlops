import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri('sqlite:///mlflow.db')
client = MlflowClient()
MODEL_NAME = "diabetes-risk-model"

experiments = mlflow.search_experiments()
print("Experiments:", [e.name for e in experiments])
exp = next((e for e in experiments if e.name != "Default"), None)
if exp is None:
    raise SystemExit("No experiment found - run training first")

runs = mlflow.search_runs(exp.experiment_id, order_by=["metrics.f1 DESC"], max_results=1)
if runs.empty:
    raise SystemExit("No runs found - run training first")
run_id = runs.iloc[0]["run_id"]
print("Best run:", run_id)

artifacts = [f.path for f in client.list_artifacts(run_id)]
print("Artifacts:", artifacts)
if "model" not in artifacts:
    raise SystemExit("No 'model' folder in run artifacts - training used a different log_model path")

if MODEL_NAME not in [m.name for m in client.search_registered_models()]:
    mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    print("Registered version:", mv.version)
else:
    mv = client.get_latest_versions(MODEL_NAME)[0]
    print("Already registered, latest version:", mv.version)

client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
print("Alias 'champion' set to version", mv.version)
