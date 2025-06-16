import mlflow, os

EXPERIMENT_NAME = "Default"
MODEL_NAME = "model"
TRACKING_URI = "https://dagshub.com/richardlois1/SML_Modelling.mlflow/"

mlflow.set_tracking_uri(TRACKING_URI)
client = mlflow.tracking.MlflowClient()

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
runs = client.search_runs([experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
run_id = runs[0].info.run_id

print("Latest run ID:", run_id)
os.makedirs("downloaded_model", exist_ok=True)
client.download_artifacts(run_id, MODEL_NAME, "downloaded_model")
print("Model downloaded to: downloaded_model/")
