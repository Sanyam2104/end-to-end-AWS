import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import dataframe_merger
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig


# this decorator directly define class variables that we genrally define
# within __init__ method
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw_data.csv')



class DataIngestion():
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        '''
           suppose your data is in some sql client DB, then this
           method will be used to write code to fetch data from there
        '''
        logging.info('Entering Data Ingestion Method')
        try:
            logging.info('Read the data from csv into dataframe')

            df_credit = pd.read_csv('notebook\\credit_record.csv')  
            df_application = pd.read_csv('notebook\\application_record.csv')
            print('CSVS LOADED')
            
            logging.info('Merging the csvs into one dataframe')

            # this will take both the csvs and do some processing and merge the datasets
            df_final = dataframe_merger(df_credit, df_application)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df_final.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Training Test split initiated')
            train_set, test_set = train_test_split(df_final, test_size=0.2, random_state=43)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Data Ingestion completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            CustomException(e, sys)


if __name__=="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    obj2 = DataTransformation()
    train_arr, test_arr, _ = obj2.initiate_data_transformation(train_data, test_data)

    model = ModelTrainer()
    print(model.initiate_model_train(train_arr, test_arr))



