import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, "../../")

import argparse

import pandas as pd

from numpy import save
from load_data import read_params
import joblib

from sklearn.pipeline import FeatureUnion, Pipeline

from build_library.utils import FeatureSelector, CategoricalFeatsAdded,\
RemoveNegativeValues, SimpleImputerTransformer, \
CapOutliers, DelUnusedCols


class Data:
    """
    Loads data, samples training data if specified, assigns features_df and target_df
    """
    full_pipeline = None
    clf_param = {}

    def __init__(self, file, target, sample=False, n_samples=None, frac=None):
        self.file = file
        self. sample = sample
        self.n_samples = n_samples
        self.target = target
        self.frac = frac

    def get_data(file, target, sample=False, n_samples=None, frac=None):
        """Load train data with option to sample
        Args:
            frac= fraction(percent) of sample data to load for training
        """
        data = pd.read_csv(file)
        if sample:
            """Sample train data due to resource limitation"""
            data = data.groupby(target).apply(lambda x: x.sample(frac=frac))
            data = data.reset_index(drop=True)
        else:
            data = data
        print(data.shape, 'data loaded\n')
        return data

    def get_features(data):
        """Assign features dataframe"""
        features_df = data
        print(features_df.shape, 'features assigned\n')
        return features_df

    def get_target(data, target):
        """Assign target"""
        target_df = data[target].values
        print(target_df.shape, '...target rows loaded\n')
        return target_df


def transform_data(numpy_file_name, transformed_data):
    """
    Saves transformed numpy array as a .npy file
    """
    save(numpy_file_name, transformed_data)


def transform_and_saved_data(config_path, test=False):
    """
    input: data csv
    output: transformed data for modeling (a numpy array)
    """
    config = read_params(config_path)
    sample = config["transform_data_config"]["sample"]
    frac = config["transform_data_config"]["frac"]
    target = config["raw_data_config"]["target"]
    seed = config["raw_data_config"]["random_state"]
    categorical_features = config["transform_data_config"]["categorical_features"]
    numerical_features = config["transform_data_config"]["numerical_features"]
    transformed_data = config["transform_data_config"]["transformed_data"]
    transformed_target = config["transform_data_config"]["transformed_target"]
    transformed_test_data = config["transform_data_config"]["transformed_test_data"]
    transformed_test_target = config["transform_data_config"]["transformed_test_target"]
    transform_pipeline = config["transform_data_config"]["transform_pipeline"]

    if test:
        """transforms test data"""
        file = config["processed_data_config"]["test_data_csv"]
        transformed_data = transformed_test_data
        transformed_target = transformed_test_target
    else:
        """transforms train data"""
        file = config["processed_data_config"]["train_data_csv"]
        transformed_data = transformed_data
        transformed_target = transformed_target

    """load data"""
    data = Data.get_data(file, target=target, sample=sample, frac=frac)

    """get features"""
    features_df = Data.get_features(data)

    """get target"""
    target_df = Data.get_target(data, target=target)

    """Data transformation pipelines"""
    """define steps in the categorical pipeline"""
    categorical_pipeline = Pipeline(steps=[('cat_selector', FeatureSelector(categorical_features)),
                                           ('cat_feats_add', CategoricalFeatsAdded()),
                                           ('delete_unused', DelUnusedCols(features_df))])
    """define the steps in the numerical pipeline"""
    numerical_pipeline = Pipeline(steps=[('num_selector', FeatureSelector(numerical_features)),
                                         ('remove_negative_values', RemoveNegativeValues(features_df)),
                                         #('standard_trans', StandardScalerTransformer(features_df)),
                                         ('impute_missing', SimpleImputerTransformer(features_df)),
                                         ('cap_outliers', CapOutliers(features_df))])

    """combine numerical and categorical pipeline into one full big pipeline horizontally using FeatureUnion"""
    Data.full_pipeline = FeatureUnion(transformer_list=[('categorical_pipeline', categorical_pipeline),
                                                        ('numerical_pipeline', numerical_pipeline)])

    if not test:
        """save transformation pipeline"""
        Data.clf_param["preprocessing"] = Data.full_pipeline
        # serialize transformation pipeline object to disk
        joblib.dump(Data.clf_param, transform_pipeline)
        print("Saved preprocessing transform_pipeline to disk")

    """disable pandas chained_assignment warning"""
    pd.options.mode.chained_assignment = None

    """fit data to data transformation pipeline"""
    prepared_features = Data.full_pipeline.fit_transform(features_df)
    transform_data(transformed_data, prepared_features)
    transform_data(transformed_target, target_df)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    print()
    print("******** Transformed Train Data *************")
    transform_and_saved_data(config_path=parsed_args.config)
    print()
    print("******** Transformed Test Data *************")
    transform_and_saved_data(config_path=parsed_args.config, test=True)
