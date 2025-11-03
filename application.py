from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application 

# import Ridge_regressor and Standard Scaler pickle 
ridge_model = pickle.load(open("Models/ridge_model.pkl","rb"))
standard_scaler = pickle.load(open("Models/sc.pkl","rb"))

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=['GET','POST'])
def predict_datapoint(): 
    if request.method=="POST":
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        WS = float(request.form.get('WS'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled = standard_scaler.transform([[Temperature,RH,WS,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html',results=result[0])
    

    else: 
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True, port=5001)