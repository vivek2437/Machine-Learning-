import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictdata",methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
        rh = float(request.form.get('RH'))
        temp = float(request.form.get('Temperature'))
        ws = float(request.form.get('Ws'))
        rain = float(request.form.get('Rain'))
        ffmc = float(request.form.get('FFMC'))
        isi = float(request.form.get('ISI'))
        classes = float(request.form.get('Classes'))
        region = float(request.form.get('Region'))
        dmc = float(request.form.get('DMC'))


        new_data_scaled=standard_scaler.transform([[rh, temp, ws, rain, ffmc,dmc, isi, classes, region]])
        result=ridge_model.predict(new_data_scaled)

        return render_template('home.html',results=result[0])

    else:
        return render_template('home.html')
    

if __name__=="__main__":
    app.run(host="0.0.0.0")