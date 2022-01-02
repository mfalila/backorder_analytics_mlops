import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, "../../")

import tensorflow as tf
import tensorflow.keras as keras
#from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

import json
import yaml
#import joblib
#import pickle
#import mlflow
import argparse
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, \
    confusion_matrix, classification_report
from src.data.load_data import read_params
from numpy import load

from keras.utils import np_utils
from keras.models import model_from_json
from keras import regularizers


class Models(object):
    """
    Holds all modeling objects
    """
    prepared_features = np.array([])
    target_df = np.array([])
    test_prepared_features = np.array([])
    test_target_df = np.array([])
    model = None
    history = None
    best_score = None
    test_score = None

    def __init__(self, epochs, batch_size, dropout, verbose,
                 checkpoint, model_file, weight_file):
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.verbose = verbose
        #self.checkpoint=checkpoint
        #self.model_file=model_file
        #self.weights_file = weights_file

    def train_data(config_path):
        """
        Gets saved params
        """
        config = read_params(config_path)
        loss = config["train_and_evaluate_config"]["loss"]
        optimizer = config["train_and_evaluate_config"]["optimizer"]
        metrics = config["train_and_evaluate_config"]["metrics"]
        epochs = config["train_and_evaluate_config"]["epochs"]
        batch_size = config["train_and_evaluate_config"]["batch_size"]
        dropout = config["train_and_evaluate_config"]["dropout"]
        verbose = config["train_and_evaluate_config"]["verbose"]
        transformed_data = config["transform_data_config"]["transformed_data"]
        test_transformed_data = config["transform_data_config"]["transformed_test_data"]
        transformed_target = config["transform_data_config"]["transformed_target"]
        test_transformed_target = config["transform_data_config"]["transformed_test_target"]
        Models.prepared_features = load(transformed_data)
        Models.test_prepared_features = load(test_transformed_data)
        Models.target_df = load(transformed_target)
        Models.test_target_df = load(test_transformed_target)

        print("debug:1")
        print(Models.target_df)
        print(Models.prepared_features.shape)
        print(Models.target_df.shape)

        Models.train_and_evaluate(loss=loss, optimizer=optimizer, metrics=metrics, epochs=epochs,
                                  batch_size=batch_size, dropout=dropout, verbose=verbose)

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
                                 metrics=metrics)

            print('Training started....')
            print("Debug:2")
            print(Models.prepared_features)
            print(Models.target_df)

            # fit keras model to data
            Models.history = Models.model.fit(Models.prepared_features, Models.target_df,
                                              epochs=epochs, batch_size=batch_size, verbose=verbose)

            print('Training completed')
            print("Debug:3")
            print(Models.model)
            Models.get_score()

    def get_score():
        """returns training or test score"""
        print("Debug4")
        print(Models.model)
        _, score = Models.model.evaluate(Models.prepared_features, Models.target_df)
        train_loss = Models.history.history["loss"][-1]
        """save best score"""
        Models.best_score = score*100
        print('Train Score: %.2f' % (score*100))
        print('Train Loss: %.2f' % train_loss)
        """save test score"""
        _, score = Models.model.evaluate(Models.test_prepared_features, Models.test_target_df)
        Models.test_score = score * 100
        print('test Score: %.2f' % (score * 100))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    Models.train_data(config_path=parsed_args.config)
