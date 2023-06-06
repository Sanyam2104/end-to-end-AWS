import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd  
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, fix_imbalancing_data

@dataclass
class DataTransformationConfig:
    preprocessor_file_obj_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data trnasformation
        
        '''
        try:
            numerical_col = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
                        'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL',
                        'CNT_FAM_MEMBERS', 'month_on_book']
            
            categorical_col = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE',
                       'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 
                       'OCCUPATION_TYPE']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",MinMaxScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("label_encoder", OrdinalEncoder())
                ]
            )

            logging.info(f"Categorical columns: {categorical_col}")
            logging.info(f"Numerical columns: {numerical_col}")

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_col),
                ('cat_pipeline', cat_pipeline, categorical_col)
            ])

            return preprocessor            

        except Exception as e:
             raise CustomException(e,sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        train_df['OCCUPATION_TYPE'].fillna('NOT EMPLOYED', inplace=True)
        test_df['OCCUPATION_TYPE'].fillna('NOT EMPLOYED', inplace=True)

        logging.info("Read train and test data completed")

        logging.info("Obtaining preprocessing object")

        preprocessing_obj = self.get_data_transformer_object()

        target_column_name = 'TARGET'
        
        input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
        target_feature_train_df = train_df[target_column_name]

        input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
        target_feature_test_df = test_df[target_column_name]

        logging.info(
            f"Applying preprocessing object on training dataframe and testing dataframe."
        )

        input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
        input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

        train_arr = np.c_[
            input_feature_train_arr, np.array(target_feature_train_df)
        ]

        ### Calling SMOTE on train_arr now
        logging.info('SMOTE INITIALIZED')
        train_arr = fix_imbalancing_data(train_arr)

        test_arr = np.c_[
            input_feature_test_arr, np.array(target_feature_test_df)
        ]

        logging.info(f"Saved preprocessing object.")

        save_object(
            file_path = self.data_transformation_config.preprocessor_file_obj_path,
            obj = preprocessing_obj
        )

        print(test_arr[0])
        return (
            train_arr, 
            test_arr, 
            self.data_transformation_config.preprocessor_file_obj_path
        )




