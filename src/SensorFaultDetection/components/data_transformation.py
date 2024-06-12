import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from src.SensorFaultDetection.exception import CustomException
from src.SensorFaultDetection.logger import logging
from imblearn.combine import SMOTETomek
from sklearn.compose import ColumnTransformer
from src.SensorFaultDetection.utils import save_object


import os


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info("get_data_transformer_object")
            # define custom function to replace 'NA' with np.nan
            replace_na_with_nan = lambda X: np.where(X == 'na', np.nan, X)

            # define the steps for the preprocessor pipeline
            nan_replacement_step = ('nan_replacement', FunctionTransformer(replace_na_with_nan))
            imputer_step = ('imputer', SimpleImputer(strategy='constant', fill_value=0))
            scaler_step = ('scaler', RobustScaler())

            preprocessor = Pipeline(
                steps=[
                imputer_step,
                scaler_step
                ]
            )
            
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)



    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            train_df = train_df.fillna(train_df.mean())

            test_df = pd.read_csv(test_path)
            test_df = test_df.fillna(test_df.mean())
            
            logging.info("get_data_transformer_object1")
            object_columns = [col for col in train_df.columns if train_df[col].dtype == 'object']
            preprocessor = self.get_data_transformer_object()
            object_columns.append("Good/Bad")
            target_column_name = "Good/Bad"
            target_column_mapping = {+1: 0, -1: 1}
            
            logging.info("get_data_transformer_object2")

            #training dataframe
            input_feature_train_df = train_df.drop(columns=object_columns, axis=1)
            target_feature_train_df = train_df[target_column_name].map(target_column_mapping)

            logging.info("get_data_transformer_object3")
            
            #testing dataframe
            input_feature_test_df = test_df.drop(columns=object_columns, axis=1)
            target_feature_test_df = test_df[target_column_name].map(target_column_mapping)

            transformed_input_train_feature = preprocessor.fit_transform(input_feature_train_df)

            transformed_input_test_feature =preprocessor.transform(input_feature_test_df)

            smt = SMOTETomek(sampling_strategy="auto")
            

            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                transformed_input_train_feature, target_feature_train_df
            )

            # input_feature_test_final, target_feature_test_final = smt.fit_resample(
            #     transformed_input_test_feature, target_feature_test_df
            # )

            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final) ]
            test_arr = np.c_[ transformed_input_test_feature, np.array(target_feature_test_df) ]

            save_object(self.data_transformation_config.preprocessor_obj_file_path,
                        obj= preprocessor)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
