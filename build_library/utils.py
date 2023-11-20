import warnings
warnings.filterwarnings('ignore')

import argparse
import numpy as np
import pandas as pd

from src.data.load_data import read_params
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
import sklearn.preprocessing
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

args = argparse.ArgumentParser()
args.add_argument("--config", default="params.yaml")
##parsed_args = args.parse_args()
parsed_args, unknown = args.parse_known_args()
config_path = parsed_args.config

config = read_params(config_path)
seed = config["raw_data_config"]["random_state"]


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom transformer to extract columns passed as arguments
    """
    def __init__(self, feature_names):
        """Class constructor"""
        self.feature_names = feature_names

    def fit(self, features_df, target=None):
        """Returns self and nothing else"""
        return self

    def transform(self, features_df, target=None):
        """This method returns selected features"""
        return features_df[self.feature_names]


class DropMissing(BaseEstimator, TransformerMixin):
    """
    Takes df, drops all missing
    """
    def __init__(self, df):
        self.df = df

    def fit(self, df, target=None):
        """Returns self, nothing else."""
        return self

    def transform(self, df, target=None):
        """Drops missing data rows. """
        df.dropna(axis=0, how='any', inplace=True)
        return df


class CategoricalFeatsAdded(BaseEstimator, TransformerMixin):
    """
    A custom transformer to add engineered categorical features
    input: df
    output: checks if condition is met and adds indicator variable
    """
    def __init__(self, neg_inv_balance=True, low_inventory=True, \
                 low_intransit=True, high_forecast=True):
        """class constructor"""
        self.neg_inv_balance = neg_inv_balance
        self.low_inventory = low_inventory
        self.low_intransit = low_intransit
        self.high_forecast = high_forecast

    def fit(self, features_df, target=None):
        """ Returns self, nothing else is done here"""
        return self

    def transform(self, features_df, target=None):
        """ Creates aforementioned features and drops redundant ones"""

        if self.neg_inv_balance:
            """Check if needed"""
            ##features_df['neg_inv_balance'] = (features_df.national_inv < 0).astype(int)
            features_df['neg_inv_balance'] = ((features_df.national_inv).astype(int) < 0).astype(int)
            ##features_df['neg_inv_balance'] = (features_df.loc[features_df.national_inv] < 0).astype(int)

        if self.low_inventory:
            """check if needed"""
            features_df['low_inventory'] = ((features_df['national_inv']).astype(int) < \
                                            features_df['national_inv'].median()).astype(int)

        if self.low_intransit:
            """check if needed"""
            features_df['low_intransit'] = ((features_df['in_transit_qty']).astype(int) < \
                                            features_df['in_transit_qty'].mean()).astype(int)

        if self.high_forecast:
            """check if needed"""
            features_df['high_forcast'] = ((features_df['forecast_3_month']).astype(int) > \
                                           features_df['forecast_3_month'].mean()).astype(int)

        return features_df.values


class RemoveNegativeValues(BaseEstimator, TransformerMixin):
    """
    Takes df, converts all negative values to positive
    """
    def __init__(self, features_df):
        self.features_df = features_df

    def fit(self, features_df, target=None):
        """Returns self, does nothing else"""
        return self

    def transform(self, features_df, target=None):
        """Takes df, returns absolute values"""
        if ((features_df).astype(int) < 0).any().any(): #checks if any neg value in df
            features_df = features_df.abs()
        else:
            pass
        #features_df = features_df.abs()
        return features_df


class CategoricalImputerTransformer(BaseEstimator, TransformerMixin):
    """
    This transformer imputes missing values on categorical pipeline
    """
    def __init__(self, features_df, target=None):
        self.features_df = features_df

    def fit(self, features_df, target=None):
        return self

    def transform(self, features_df, target=None):
        imputer = SimpleImputer(missing_values=np.NaN,
                                strategy='most_frequent')
        # Fit data to the imputer object
        imputer = imputer.fit(features_df)
        # Impute the data
        imputed = imputer.transform(features_df)
        features_df = pd.DataFrame(data=imputed)
        return features_df


class SimpleImputerTransformer(BaseEstimator, TransformerMixin):
    """
    This transformer imputes missing values
    """
    def __init__(self, features_df, target=None):
        self.features_df = features_df
        self.target = target

    def fit(self, features_df, target=None):
        return self

    def transform(self, features_df, target=None):
        imputer = SimpleImputer(missing_values=np.NaN,
                                strategy='median')
        # Fit data to the imputer object
        imputer = imputer.fit(features_df)
        # Impute the data
        imputed = imputer.transform(features_df)
        features_df = pd.DataFrame(data=imputed)
        return features_df


class CapOutliers(BaseEstimator, TransformerMixin):
    """
    Takes df, caps outliers
    """
    def __init__(self, features_df):
        self.features_df = features_df

    def fit(self, features_df, Target=None):
        """Returns self, does nothing else"""
        return self

    def transform(self, features_df, q=0.90, target=None):
        for col in features_df.columns:
            if (((features_df[col].dtype) == 'float64') | ((features_df[col].dtype) == 'int64')):
                percentiles = features_df[col].quantile([0.01, q]).values
                features_df[col][features_df[col] <= percentiles[0]] = percentiles[0]
                features_df[col][features_df[col] >= percentiles[1]] = percentiles[1]
            else:
                features_df[col] = features_df[col]
        return features_df


class StandardScalerTransformer(BaseEstimator, TransformerMixin):
    """This transformer standardizes all numerical features"""
    def __init__(self, features_df, target=None):
        self.features_df = features_df
        self.target = target

    def fit(self, features_df, target=None):
        return self

    def transform(self, features_df, target=None):
        features = features_df
        scaler = StandardScaler().fit(features)
        features_df = scaler.transform(features)
        return features_df


class DelUnusedCols(BaseEstimator, TransformerMixin):
    """
    This transformer deletes unused columns from a data pipeline
    Col 0 holds an extra column for 'national_inv' added through the categorical feats. pipeline.
    This row is no longer needed after new categorical features leveraging the column are engineered
    """
    def __init__(self, features_df, target=None):
        self.features_df = features_df
        self.target = target

    def fit(self, features_df, target=None):
        return self

    def transform(self, features_df, target=None):
        a = features_df
        a = np.delete(a, [0, 1, 2], 1)
        features_df = a
        return features_df


class SplitData(object):
    """Takes prepared data, performs train test split"""
    prepared_features_df = pd.DataFrame()
    feats = None
    target = None
    train_feats = None
    test_feats = None
    train_target = None
    test_target = None

    def get_dataframe(prepared_features):
        """Takes prepared features array, returns dataframe"""
        SplitData.prepared_features_df = pd.DataFrame(data=prepared_features)

        # get features and target data
        SplitData.feats = SplitData.prepared_features_df.drop([0], axis=1)
        SplitData.target = SplitData.prepared_features_df[0]

    def split_data(test_frac):
        X_train, X_test, y_train, y_test = train_test_split(SplitData.feats, SplitData.target, \
                                                            test_size=test_frac, random_state=seed, \
                                                            stratify=SplitData.target)
        # save datasets
        SplitData.train_feats = X_train
        SplitData.test_feats = X_test
        SplitData.train_target = y_train
        SplitData.test_target = y_test

        print('\nTrain and test data assigned\n')
