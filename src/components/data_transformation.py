# The main purpose of this data transformation is basically to do feature engineering, data cleaning, etc.

import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # Below one is to create all pickle files responsible fro converting categorical to numerical
    def get_data_transformer_object(self):
        """
        This method is responsible for creating the preprocessor object and saving it in the artifacts folder,
        main purpose is for data transformation

        """
        try:
            numerical_colums = ["writing_score", "reading_score"]
            categorcial_columns = ["gender", "race_ethnicity",
                                   "parental_level_of_education", "lunch", "test_preparation_course"]

            # below pipeline will be running on the training dataset
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            # below pipeline will be running on the training dataset, which will handle missing values and encode the values and
            # then standardization with standardscaler
            categorcial_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numerical Columns: {numerical_colums}")
            logging.info(f"Categorical Columns: {categorcial_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", numerical_pipeline, numerical_colums),
                    ("categorical_pipeling", categorcial_pipeline, categorcial_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("reading training and testing data completed")
            logging.info("obtaining preprocessing object")

            preprocessing_object = self.get_data_transformer_object()
            target_col_name = "math_score"
            numerical_colums = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(
                columns=[target_col_name], axis=1)
            target_feature_train_df = train_df[target_col_name]

            input_feature_test_df = test_df.drop(
                columns=[target_col_name], axis=1)
            target_feature_test_df = test_df[target_col_name]

            logging.info(
                f"Applying preprocessing obj on training df and testing df")

            input_feature_train_array = preprocessing_object.fit_transform(
                input_feature_train_df)
            input_feature_test_array = preprocessing_object.transform(
                input_feature_test_df)

            training_array = np.c_[
                input_feature_train_array, np.array(target_feature_train_df)]

            testing_array = np.c_[input_feature_test_array,
                                  np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_object
            )

            return (training_array, testing_array, self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys)
