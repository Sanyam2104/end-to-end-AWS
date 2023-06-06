import sys, os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')


            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            print("After Loading")
            print('features  : ',type(features))


            # data_scaled = preprocessor.transform(features)
            # print('data_scaled : ',data_scaled)
            # preds = model.predict(data_scaled)
            preds = model.predict(np.array([0.10526316, 0.0755814,  0.61490121, 0.03984975, 0.,0., 0.,1.,0.15789474,
                                    0.6,0.,1.,1.,4.,1,1.,1.,0.]).reshape(1, -1))

            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        CNT_CHILDREN: int,
        AMT_INCOME_TOTAL: int,
        DAYS_BIRTH: int,
        DAYS_EMPLOYED: int,
        FLAG_MOBIL: int,
        FLAG_WORK_PHONE: int,
        FLAG_PHONE: int,
        FLAG_EMAIL: int,
        CNT_FAM_MEMBERS: int,
        month_on_book: int,
        CODE_GENDER: str,
        FLAG_OWN_CAR: str,
        FLAG_OWN_REALTY: str,
        NAME_INCOME_TYPE: str,
        NAME_EDUCATION_TYPE: str,
        NAME_FAMILY_STATUS: str,
        NAME_HOUSING_TYPE: str,
        OCCUPATION_TYPE: str):

        self.CNT_CHILDREN = CNT_CHILDREN
        self.AMT_INCOME_TOTAL = AMT_INCOME_TOTAL
        self.DAYS_BIRTH = DAYS_BIRTH
        self.DAYS_EMPLOYED = DAYS_EMPLOYED
        self.FLAG_MOBIL = FLAG_MOBIL

        self.FLAG_WORK_PHONE = FLAG_WORK_PHONE
        self.FLAG_PHONE = FLAG_PHONE
        self.FLAG_EMAIL = FLAG_EMAIL
        self.CNT_FAM_MEMBERS = CNT_FAM_MEMBERS
        self.month_on_book = month_on_book

        self.CODE_GENDER = CODE_GENDER
        self.FLAG_OWN_CAR = FLAG_OWN_CAR
        self.FLAG_OWN_REALTY = FLAG_OWN_REALTY
        self.NAME_INCOME_TYPE = NAME_INCOME_TYPE
        self.NAME_EDUCATION_TYPE = NAME_EDUCATION_TYPE

        self.NAME_FAMILY_STATUS = NAME_FAMILY_STATUS
        self.NAME_HOUSING_TYPE = NAME_HOUSING_TYPE
        self.OCCUPATION_TYPE = OCCUPATION_TYPE



    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'CODE_GENDER' : [self.CODE_GENDER],
                'FLAG_OWN_CAR' : [self.FLAG_OWN_CAR],
                'FLAG_OWN_REALTY' : [self.FLAG_OWN_REALTY],
                'CNT_CHILDREN' : [self.CNT_CHILDREN],
                'AMT_INCOME_TOTAL' : [self.AMT_INCOME_TOTAL],
                'NAME_INCOME_TYPE' : [self.NAME_INCOME_TYPE],
                'NAME_EDUCATION_TYPE' : [self.NAME_EDUCATION_TYPE],
                'NAME_FAMILY_STATUS' : [self.NAME_FAMILY_STATUS],
                'NAME_HOUSING_TYPE' : [self.NAME_HOUSING_TYPE],
                'DAYS_BIRTH' : [-self.DAYS_BIRTH],
                'DAYS_EMPLOYED' : [-self.DAYS_EMPLOYED],
                'FLAG_MOBIL' : [self.FLAG_MOBIL],
                'FLAG_WORK_PHONE' : [self.FLAG_WORK_PHONE],
                'FLAG_PHONE' : [self.FLAG_PHONE],
                'FLAG_EMAIL' : [self.FLAG_EMAIL],
                'OCCUPATION_TYPE' : [self.OCCUPATION_TYPE],
                'CNT_FAM_MEMBERS' : [self.CNT_FAM_MEMBERS],
                'month_on_book' : [self.month_on_book]              
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
