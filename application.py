from flask import Flask,request,jsonify,render_template
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler 
import pickle
 
application = Flask(__name__)
app = application

## import regressor model and standard scaler pickle.
regessor_model = pickle.load(open(r"models\regressor.pkl","rb")) #"models\regressor.pkl" cause Invalid argument: because of escape characters. 
standard_scaler = pickle.load(open(r"models\scaler.pkl","rb"))

## Route for home page. 
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata",methods=["GET","POST"])
def predict_datapoint():
    if request.method == "POST":
        Weight= request.form.get("Weight")

        new_data_scaled = standard_scaler.transform([[Weight]])
        result=regessor_model.predict(new_data_scaled)

        return render_template("home.html",result=result[0])

    else:
        return render_template("home.html")

if __name__=="__main__":
    app.run(host="0.0.0.0")