import os
import sys
from dataclasses import dataclass

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import KFold,StratifiedKFold

from sklearn.metrics import accuracy_score, confusion_matrix,precision_score, recall_score, f1_score

from src.logger import logging
from src.exception import CustomException

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig():
    trained_model_file_path = os.path.join("artifacts", 'model.pkl')

class ModelTrainer():
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_train(self, train_arr, test_arr):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test=(
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            # I'm not using multiple models but below you can initialize models 
            # that you want to try 
            models = {
                'DecisionTreeClassifier': DecisionTreeClassifier(),
            }

            params = {'class_weight':['balanced'], 
                        'max_depth':[35,45], 
                        'splitter':['best'], 
                        'random_state':[42],
                        'min_samples_split':[38]
                    }

            logging.info('training the model')
            # model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, 
            #                                     X_test=X_test, y_test=y_test, 
            #                                     models=models, param=params)
            
            # best_model_score = max(sorted(model_report.values()))

            # ## To get best model name from dict

            # best_model_name = list(model_report.keys())[
            #     list(model_report.values()).index(best_model_score)
            # ]
            # best_model = models[best_model_name]
            
            best_model = DecisionTreeClassifier(class_weight='balanced', 
                                    max_depth=35, 
                                    splitter='best', 
                                    random_state=42,
                                    min_samples_split=38)
            best_model.fit(X_train,y_train)
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            logging.info('Evaluating the test data')
            predicted=best_model.predict(X_test)

            f1_scor = f1_score(y_test, predicted)
            return f1_scor
        
        except CustomException as e:
            raise CustomException(e, sys)