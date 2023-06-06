import os
import sys

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score, recall_score, f1_score


from collections import Counter
from imblearn.over_sampling import SMOTE

from src.exception import CustomException

def dataframe_merger(df_credit, df_application):
    credit_group = df_credit.groupby('ID')['MONTHS_BALANCE'].min()

    # Creating a month on book column which tells us the no. of month it has been since the account opened 
    # and maps the months balance in a manner that is understable easily.
    df_credit['open_month'] =  df_credit['ID'].apply(lambda x: credit_group.loc[x])
    df_credit['month_on_book'] = df_credit['MONTHS_BALANCE'] - df_credit['open_month']
    df_credit.drop(['MONTHS_BALANCE', 'open_month'], axis=1, inplace=True)

    # changing status as 0 and 1 , 0 - GOOD (PAID FULLY) ,  1 - BAD (DIDN'T PAID)
    df_credit.STATUS = df_credit.STATUS.map({'C':0, 'X':0, '0':0, '1':1, '2':1, '3':1, '4':1, '5':1,})

    # dropping duplicates if any
    df_credit.drop_duplicates(inplace=True)

    df_final = pd.merge(df_application, df_credit, on='ID', how='inner')
    df_final.drop(['ID'], axis=1, inplace=True)
    df_final.rename(columns={'STATUS':'TARGET'}, inplace=True)
    df_final.drop_duplicates(inplace=True)

    return df_final

def fix_imbalancing_data(df_final):
    # Balancing the dataset SMOTE

    features = df_final[:, :-1]
    label = df_final[:,-1]

    smote = SMOTE()
    
    try:
        # fit predictor and target variable
        x_smote, y_smote = smote.fit_resample(features, label)
        print('doneneenen')

        print('Original dataset shape', Counter(label))
        print('Resample dataset shape', Counter(y_smote))

        z = np.c_[x_smote, y_smote]

    except CustomException as e:
        raise CustomException(e, sys)
    
    return z


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except CustomException as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)   # Train model


            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = results_viewer(y_train, y_train_pred)

            test_model_score = results_viewer(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

            print('REPORT : : ', report)

        return report

    except Exception as e:
        raise CustomException(e, sys)
    

def results_viewer(actual, predicted):
    acc = accuracy_score(actual ,predicted)
    confusion_mat = confusion_matrix(actual ,predicted)
    pre_score = precision_score(actual ,predicted)
    recall = recall_score(actual ,predicted)
    f1 = f1_score(actual ,predicted)
    specificitactual  = confusion_mat[0,0] / (confusion_mat[0,0] + confusion_mat[0,1])

    print(f'Accuracy Score = {acc}\nPrecision Score = {pre_score}\nRecall Score = {recall}\nF1 Score = {f1}\nSpecificity Test = {specificitactual }\n\nConfusion Matrix = \n{confusion_mat}')

    return f1
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)