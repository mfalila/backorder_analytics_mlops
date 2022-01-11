import yaml
import os
import json
import joblib
import numpy as np
import pandas as pd

import tensorflow as tf
from keras.models import model_from_json
np.set_printoptions(precision=8, suppress=False)

import src.data.load_data
#from src.data.load_data import read_params

params_path = "params.yaml"
schema_path = os.path.join("prediction_service", "schema_in.json")

config = src.data.load_data.read_params(params_path)
save_model_config = config["save_model_config"]
model_file = save_model_config["model_file"]
checkpoint = save_model_config["checkpoint"]
#model_weights = save_model_config["weights_file"]
#clf_checkpoint = save_model_config["checkpoint"]
model_weights = "models/clf_model_weights.h5"
clf_checkpoint = "data/processed/preprocessing_pipeline.joblib"



class NotInRange(Exception):
    def __init__(self, message="Values entered are not in expected range"):
        self.message = message
        super().__init__(self.message)


class NotInCols(Exception):
    def __init__(self, message="Not in cols"):
        self.message = message
        super().__init__(self.message)


def read_params(config_path=params_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


class Predict():
    transformed_features = None
    loaded_clf_checkpoint = None

    def __init__(self, data):
        self.data = data

    def load_checkpoint(file_name):
        """loads saved checkpoint file"""
        Predict.loaded_clf_checkpoint = joblib.load(file_name)
        return Predict.loaded_clf_checkpoint

    def transform_data(data):
        """takes dict from form, returns transformed feats"""
        # load the preprocessing pipeline and dataset
        Predict.load_checkpoint(clf_checkpoint)
        # get preprocessing parameters
        reloaded_full_pipeline = Predict.loaded_clf_checkpoint['preprocessing']
        # transform features
        Predict.transformed_features = reloaded_full_pipeline.transform(data)
        return Predict.transformed_features

    def predict(data):
        """takes transformed features, returns prediction"""
        # open saved json model in read mode
        json_file = open(model_file, 'r')
        # load the model
        json_model = json_file.read()
        # close file
        json_file.close()
        # convert the json model back to a sequential model
        loaded_model = model_from_json(json_model)
        # get model weights
        loaded_model.load_weights(model_weights)
        # compile loaded model
        loaded_model.compile(optimizer='adam', loss='categorical_crossentropy',
                             metrics=[tf.keras.metrics.Recall()])
        """ convert data to appropriate format for compatibility with tf"""
        transformed_arr = np.asarray(data).astype(np.float32)

        prediction = loaded_model.predict(transformed_arr)
        return prediction

"""
        try:
            if prediction == 1 or prediction == 0:
                return prediction
            else:
                raise NotInRange
        except NotInRange:
            return "unexpected result"
"""


def get_schema(schema_path=schema_path):
    with open(schema_path) as json_file:
        schema = json.load(json_file)
    return schema


def validate_input(dict_request):
    def _validate_cols(col):
        schema = get_schema()
        actual_cols = schema.keys()
        if col not in actual_cols:
            raise NotInCols

    def _validate_values(col, val):
        schema = get_schema()

        if not (schema[col]["min"] <= float(dict_request[col]) <= schema[col]["max"]) :
            raise NotInRange

    for col, val in dict_request.items():
        _validate_cols(col)
        _validate_values(col, val)

    return True


def form_response(dict_request):
    response = Predict.predict(dict_request)
    return response


def api_response(dict_request):
    try:
        if validate_input(dict_request):
            request_list = [dict_request]  # convert to list
            request_df = pd.DataFrame(data=request_list)
            transformed_features = Predict.transform_data(request_df)
            response = Predict.predict(transformed_features)
            response = response[0, 0]
            return response

    except NotInRange as e:
        response = {"the_expected_range": get_schema(), "response": str(e)}
        return response

    except NotInCols as e:
        response = {"the_expected_cols": get_schema().keys(), "response": str(e)}
        return response

    except Exception as e:
        response = {"response": str(e)}
        return response
