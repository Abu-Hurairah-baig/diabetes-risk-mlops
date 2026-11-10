import joblib
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri('sqlite:///mlflow.db')
model = mlflow.sklearn.load_model('models:/diabetes-risk-model@champion')
joblib.dump(model, 'champion.pkl')
print('SUCCESS: champion.pkl generated!')
