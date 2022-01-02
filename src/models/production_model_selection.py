import os
import mlflow
import argparse
from pprint import pprint
from train_model import read_params
from mlflow.tracking import MlflowClient
from tensorflow.python.saved_model import signature_constants


key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY


def log_production_model(config_path):
    config = read_params(config_path)
    mlflow_config = config["mlflow_config"]
    model_name = mlflow_config["registered_model_name"]
    save_model_path = mlflow_config["save_model_path"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)
    runs = mlflow.search_runs(experiment_ids=1)

    max_recall = max(runs["metrics.val_recall"])
    max_recall_run_id = list(runs[runs["metrics.val_recall"] == max_recall]["run_id"])[0]

    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{model_name}'"):
        mv = dict(mv)
        if mv["run_id"] == max_recall_run_id:
            current_version = mv["version"]
            logged_model = mv["source"]
            pprint(mv, indent=4)
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Production"
            )
            mlflow.tensorflow.save_model(
                tf_saved_model_dir=os.path.join(logged_model, "tfmodel"),
                tf_meta_graph_tags=None,
                tf_signature_def_key=key,
                path=save_model_path
            )
        else:
            current_version = mv["version"]
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Staging"
            )


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = log_production_model(config_path=parsed_args.config)
