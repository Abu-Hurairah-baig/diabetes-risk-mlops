import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri('sqlite:///mlflow.db')
client = MlflowClient()

MODEL_NAME = "diabetes-risk-model"
EXPERIMENT_NAME = "diabetes-experiment"  # <-- change to your real experiment name

# 1. What experiments do we even have?
print("Experiments:", [e.name for e in mlflow.search_experiments()])

# 2. Is the model registered?
models = [m.name for m in client.search_registered_models()]
print("Registered models:", models)

# 3. Find best run and register if needed
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise SystemExit(f"Experiment '{EXPERIMENT_NAME}' not found. "
                     "Fix EXPERIMENT_NAME using the list above.")

runs = mlflow.search_runs(experiment.experiment_id,
                          order_by=["metrics.f1 DESC"], max_results=1)
if runs.empty:
    raise SystemExit("No runs found in that experiment. Train the model first.")
run_id = runs.iloc[0]["run_id"]
print("Best run:", run_id)

if MODEL_NAME not in models:
    mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    print("Registered version:", mv.version)
else:
    mv = client.get_latest_versions(MODEL_NAME)[0]
    print("Already registered, latest version:", mv.version)

client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
print("Alias 'champion' ->", mv.version)