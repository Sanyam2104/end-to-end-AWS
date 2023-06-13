from flask import Flask, request, render_template
import pandas as pd
import numpy as np

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_data', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data=CustomData(
            CNT_CHILDREN = float(request.form.get('CNT_CHILDREN')),
            AMT_INCOME_TOTAL = float(request.form.get('AMT_INCOME_TOTAL')),
            DAYS_BIRTH = float(request.form.get('DAYS_BIRTH')),
            DAYS_EMPLOYED = float(request.form.get('DAYS_EMPLOYED')),
            CNT_FAM_MEMBERS = float(request.form.get('CNT_FAM_MEMBERS')),

            FLAG_MOBIL = float(request.form.get('FLAG_MOBIL')),
            FLAG_WORK_PHONE = float(request.form.get('FLAG_WORK_PHONE')),
            FLAG_PHONE = float(request.form.get('FLAG_PHONE')),
            FLAG_EMAIL = float(request.form.get('FLAG_EMAIL')),
            month_on_book = float(request.form.get('month_on_book')),

            CODE_GENDER = request.form.get('CODE_GENDER'),
            FLAG_OWN_CAR = request.form.get('FLAG_OWN_CAR'),
            FLAG_OWN_REALTY = request.form.get('FLAG_OWN_REALTY'),
            NAME_INCOME_TYPE = (request.form.get('NAME_INCOME_TYPE')),
            NAME_EDUCATION_TYPE = (request.form.get('NAME_EDUCATION_TYPE')),

            NAME_FAMILY_STATUS = (request.form.get('NAME_FAMILY_STATUS')),
            NAME_HOUSING_TYPE = (request.form.get('NAME_HOUSING_TYPE')),
            OCCUPATION_TYPE = (request.form.get('OCCUPATION_TYPE'))
        )

        pred_df = data.get_data_as_data_frame()
        print(type(pred_df))
        print('pred_df', pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])




if __name__=="__main__":
    app.run(host="0.0.0.0", port=8080)
