import warnings
warnings.filterwarnings('ignore')

import os
import sys
sys.path.insert(0, "../../")

import tempfile
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

import argparse
import numpy as np
from urllib.parse import urlparse

from src.data.load_data import read_params
from numpy import load
import joblib
import json

from keras import regularizers
import mlflow
import mlflow.tensorflow
from mlflow.tracking import MlflowClient

from tensorflow.python.saved_model import signature_constants


class Models(object):
    """
    Holds all modeling objects
    """
    preprocessing_pipeline = None
    prepared_features = np.array([])
    target_df = np.array([])
    test_prepared_features = np.array([])
    test_target_df = np.array([])
    model = None
    history = None
    best_score = None
    test_score = None
    train_loss = None
    train_acc = None
    train_prec = None
    train_recall = None
    train_auc = None
    val_loss = None
    val_acc = None
    val_prec = None
    val_recall = None
    val_auc = None
    checkpoint = None
    model_file = None
    weights_file = None
    clf_param = {}

    def __init__(self, epochs, batch_size, dropout, verbose,
                 checkpoint, model_file, weight_file):
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.verbose = verbose

    def get_score():
        """returns training and validation scores"""
        Models.train_loss = Models.history.history["loss"][-1]
        Models.train_acc = Models.history.history["binary_accuracy"][-1]
        Models.train_prec = Models.history.history["precision"][-1]
        Models.train_recall = Models.history.history["recall"][-1]
        Models.train_auc = Models.history.history["auc"][-1]
        Models.val_loss = Models.history.history["val_loss"][-1]
        Models.val_acc = Models.history.history["val_binary_accuracy"][-1]
        Models.val_prec = Models.history.history["val_precision"][-1]
        Models.val_recall = Models.history.history["val_recall"][-1]
        Models.val_auc = Models.history.history["val_auc"][-1]

    def save_best_model(checkpoint, model_file, weights_file):
        '''Saves best model as a h5 file
         Args:
           checkpoint - location and file name  holding saved preprocessing checkpoints
           (e.g. location/checkpoint_name.joblib)
           model_file - location and file name of saved model
           (e.g. location/model_file.json)
           weights_file - location and file name of saved model weights
           (e.g. location/weights_file.h5)
        '''
        # save parameters for estimator object for preprocessing data and best model
        Models.clf_param['preprocessing'] = joblib.load(Models.preprocessing_pipeline)

        # save model architecture and convert model to json format
        Models.model_json = Models.model.to_json()
        with open(model_file, 'w') as json_file:
            json_file.write(Models.model_json)
        print('Saved model to disk')

        # save model weights
        Models.model.save_weights(filepath=weights_file,
                                  overwrite=True,
                                  save_format="h5")
        print("Saved model weights to disk")

        # serialize object to disk
        joblib.dump(Models.clf_param, checkpoint)
        print('Saved model checkpoints to disk')

    def train_and_evaluate(loss, optimizer, metrics, epochs,
                           batch_size, dropout, verbose):
        """
        builds and runs keras model
        Args:
        loss- see: https://keras.io/api/losses/
               optimizer- see: https://keras.io/api/optimizers/
               metrics- see: https://keras.io/api/metrics/
               epochs- specifies number of epochs to run
               batch_size- specifies number of batches to run
               dropout- True, if dropout added to control model over fit \
               (see: https://keras.io/api/layers/regularization_layers/dropout/)
               verbose-1 for yes, else 0 for No
        """
        # instantiate sequential object
        Models.model = Sequential()

        if dropout:
            # define keras model
            Models.model.add(Dense(12, input_dim=8, activation='relu'))
            Models.model.add(Dropout(0.5))
            Models.model.add(Dense(8, activation='relu'))
            Models.model.add(Dropout(0.5))
            Models.model.add(Dense(1, activation='sigmoid'))

        else:
            # define keras model
            Models.model.add(Dense(12, kernel_regularizer=regularizers.l2(0.01),
                                   input_dim=8, activation='relu'))
            Models.model.add(Dense(8, kernel_regularizer=regularizers.l2(0.01),
                                   activation='relu'))
            Models.model.add(Dense(1, activation='sigmoid'))

            # compile keras model
            Models.model.compile(loss=loss,
                                 optimizer=optimizer,
                                 metrics=[tf.keras.metrics.BinaryAccuracy(),
                                          tf.keras.metrics.Precision(),
                                          tf.keras.metrics.Recall(),
                                          tf.keras.metrics.AUC()])

            print('Training started....')
            # fit keras model to data
            Models.history = Models.model.fit(Models.prepared_features, Models.target_df,
                                              epochs=epochs, batch_size=batch_size, verbose=verbose,
                                              validation_data=(Models.test_prepared_features, Models.test_target_df))
            print('Training completed')
            Models.get_score()
            tf.keras.models.save_model(Models.model, "./models")
            Models.save_best_model(checkpoint=Models.checkpoint,
                                   model_file=Models.model_file,
                                   weights_file=Models.weights_file)

    def train_data(config_path):
        """
        Gets saved params
        """
        config = read_params(config_path)
        train_and_evaluate_config = config["train_and_evaluate_config"]
        transform_data_config = config["transform_data_config"]
        loss = train_and_evaluate_config["loss"]
        optimizer = train_and_evaluate_config["optimizer"]
        metrics = train_and_evaluate_config["metrics"]
        epochs = train_and_evaluate_config["epochs"]
        batch_size = train_and_evaluate_config["batch_size"]
        dropout = train_and_evaluate_config["dropout"]
        verbose = train_and_evaluate_config["verbose"]
        transformed_data = transform_data_config["transformed_data"]
        test_transformed_data = transform_data_config["transformed_test_data"]
        transformed_target = transform_data_config["transformed_target"]
        test_transformed_target = transform_data_config["transformed_test_target"]
        Models.preprocessing_pipeline = transform_data_config["transform_pipeline"]

        save_model_config = config["save_model_config"]
        Models.checkpoint = save_model_config["checkpoint"]
        Models.model_file = save_model_config["model_file"]
        Models.weights_file = save_model_config["weights_file"]

        Models.prepared_features = load(transformed_data)
        Models.test_prepared_features = load(test_transformed_data)
        Models.target_df = load(transformed_target)
        Models.test_target_df = load(test_transformed_target)

        ############################### MLFLOW ###################################

        mlflow_config = config["mlflow_config"]
        experiment_name = mlflow_config["experiment_name"]
        artifact_repository = mlflow_config["artifact_repository"]
        run_name = mlflow_config["run_name"]
        remote_server_uri = mlflow_config["remote_server_uri"]
        tf_saved_model_dir = mlflow_config["tf_saved_model_dir"]
        artifact_path = mlflow_config["artifact_path"]
        registered_model_name = mlflow_config["registered_model_name"]

        experiment_name = experiment_name
        artifact_repository = artifact_repository
        run_name = run_name
        """
        - provide tracking uri and connect to our tracking server
        - defaults to http://localhost:5000 if not specified as below
        """
        mlflow.set_tracking_uri(remote_server_uri)

        # initialize mlflow client
        client = MlflowClient()
        """
        if experiment does not exist, we create new
        else we take existing experiment id and use to run experiments
        """
        try:
            # create experiment
            experiment_id = client.create_experiment(experiment_name,
                                                     artifact_location=artifact_repository)
        except:
            # get experiment id if already exists
            experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

        with tempfile.TemporaryDirectory() as tmp_dir, mlflow.start_run(experiment_id=experiment_id,
                                                                        run_name=run_name) as run:
            """get run_id"""
            run_id = run.info.run_uuid

            """brief notes about the run"""
            MlflowClient().set_tag(run_id,
                                   "mlflow.note.content",
                                   "This experiment explores different param setting for backorder prediction model")

            Models.train_and_evaluate(loss=loss, optimizer=optimizer, metrics=metrics,
                                      epochs=epochs, batch_size=batch_size, dropout=dropout, verbose=verbose)

            """log metrics and artifacts"""
            #mlflow.tensorflow.autolog(every_n_iter=1, log_models=True)  # enables model auto log

            fname = 'sample.txt'
            tmp_path = os.path.join(tmp_dir, fname)

            # create a text file to log
            with open(tmp_path, 'w') as f:
                f.write("sample")

            # log model scores and params locally
            scores_file = config["reports"]["scores"]
            params_file = config["reports"]["params"]

            with open(scores_file, "w") as f:
                scores = {
                    "train_loss": Models.train_loss,
                    "train_acc": Models.train_acc,
                    "train_prec": Models.train_prec,
                    "train_recall": Models.train_recall,
                    "train_auc": Models.train_auc,
                    "val_loss": Models.val_loss,
                    "val_acc": Models.val_acc,
                    "val_prec": Models.val_prec,
                    "val_recall": Models.val_recall,
                    "val_auc": Models.val_auc
                }
                json.dump(scores, f, indent=4)

            with open(params_file, "w") as f:
                params = {
                    "loss": loss,
                    "optimizer": optimizer,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "dropout": dropout
                }
                json.dump(params, f, indent=4)

            # track model scores and params with MLFLOW
            mlflow.log_param("loss", loss)
            mlflow.log_param("optimizer", optimizer)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("dropout", dropout)
            mlflow.log_metric("train_loss", Models.train_loss)
            mlflow.log_metric("train_acc", Models.train_acc)
            mlflow.log_metric("train_prec", Models.train_prec)
            mlflow.log_metric("train_recall", Models.train_recall)
            mlflow.log_metric("train_auc", Models.train_auc)
            mlflow.log_metric("val_loss", Models.val_loss)
            mlflow.log_metric("val_acc", Models.val_acc)
            mlflow.log_metric("val_prec", Models.val_prec)
            mlflow.log_metric("val_recall", Models.val_recall)
            mlflow.log_metric("val_auc", Models.val_auc)

            mlflow.log_artifact(tmp_path)
            tag = None
            key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

            # log model
            tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme
            if tracking_url_type_store != "file":
                mlflow.tensorflow.log_model(tf_saved_model_dir=tf_saved_model_dir,
                                            tf_meta_graph_tags=tag,
                                            tf_signature_def_key=key,
                                            artifact_path=artifact_path,
                                            registered_model_name=registered_model_name)
            else:
                mlflow.tensorflow.load_model(Models.model, "model")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    Models.train_data(config_path=parsed_args.config)
