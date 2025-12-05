import pickle 
from flask import Flask,request,app,jsonify,url_for,render_template,redirect,flash

import numpy as np
import pandas as pd

app = Flask(__name__)#It is the starting point of the app 
regmodel = pickle.load(open('regmodel.pkl','rb'))
scalar = pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')
# when user hist the /predict_api methods is post and the data is convert to json and assign to the data again
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__=="__main__":
    app.run(debug=True)