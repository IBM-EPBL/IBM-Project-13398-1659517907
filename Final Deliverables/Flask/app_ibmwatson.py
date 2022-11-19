import pandas as pd
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
import pickle
from sklearn.preprocessing import LabelEncoder
import requests
API_KEY = "hibrYF7JAuQ2jNQUxkCXaMAhsT_WY-PSaHSrHHJEephU"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]
header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

app = Flask(__name__)
filename = 'resale_model.sav'
model_rand = pickle.load(open(filename, 'rb'))

@app.route('/')
def index():
    return render_template('resaleintro.html') 

@app.route('/register')
def register():
    return render_template('Register.html') 

@app.route('/login')
def login():
    return render_template('Login.html') 

@app.route('/predict', methods=['GET','POST'])
def predict4():
    if request.method == 'POST':
        pricer=predictprice(request)
        ypred = '€{:.2f}'.format(pricer[0])
        return jsonify(result=ypred)
    else:
        return render_template('Predict.html') 

@app.route('/y_predict', methods=['POST'])
def y_predict():
    y_prediction = predictprice(request)
    return render_template('resalepredictDesign1.html',ypred = 'The resale value predicted is €{:.2f}'.format(y_prediction[0]))

@app.route('/json_predict', methods=['POST'])
def j_predict():
    y_prediction = predictprice(request)
    result='result'
    ypred = 'The resale value predicted is €{:.2f}'.format(y_prediction[0])
    return jsonify(result=ypred)

def predictprice(request):

    regyear = int(request.form.get('regyear'))
    regmonth = int(request.form.get('regmonth'))
    powerps = float(request.form['powerps'])
    kms = float(request.form['kms'])
    gearbox = request.form['gearbox']
    damage = request.form['dam']
    model = request.form.get('modeltype')
    brand = request.form.get('brand') 
    fuelType = request.form.get('fuel')   
    vehicletype = request.form.get('vehicletype')

    new_row = {'yearOfRegistration':regyear,'powerPS':powerps, 'kilometer':kms,
            'monthOfRegistration':regmonth, 'gearbox':gearbox, 'notRepairedDamage':damage,
            'model':model, 'brand':brand, 'fuelType':fuelType, 'vehicleType':vehicletype}
    print(new_row)

    new_df = pd.DataFrame(columns = ['vehicleType', 'yearOfRegistration', 'gearbox',
                                'powerPS', 'model', 'kilometer', 'monthOfRegistration', 'fuelType',
                                'brand', 'notRepairedDamage'])

    new_df = new_df.append(new_row,ignore_index =True)
    labels = ['gearbox', 'notRepairedDamage', 'model', 'brand', 'fuelType', 'vehicleType']
    mapper = {}
    for i in labels:
        mapper[i] = LabelEncoder()
        mapper[i].classes_=np.load(str('classes'+i+'.npy'), allow_pickle=True)
        tr = mapper[i].fit_transform(new_df[i])
        new_df.loc[:, i+'_Labels'] = pd.Series(tr, index=new_df.index)

    labeled = new_df[['yearOfRegistration'
                        ,'powerPS'
                        ,'kilometer'
                        ,'monthOfRegistration'
                        ]
                    +[x+'_Labels' for x in labels]]
    X = labeled.values
    payload_scoring = {"input_data": [{"fields": [['yearOfReg', 'powerPS', 'kilometer', 'monthOfRegistration','gearbox_labels', 'notRepairedDamage_labels', 'model_labels','brand_labels', 'fuelType_labels', 'vehicletype_labels']], "values": X}]}
    #payload_scoring = {"input_data": [{"fields": [array_of_input_fields], "values": [array_of_values_to_be_scored, another_array_of_values_to_be_scored]}]}
    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/1e6b2977-778a-489f-a753-d1d4fccdd3c9/predictions?version=2022-11-14', json=payload_scoring,
    headers={'Authorization': 'Bearer ' + mltoken})
    predictions = response_scoring.json()
    print("Scoring response")
    print(response_scoring.json())
    #y_prediction = model_rand.predict(X)
    y_prediction = predictions['predictions'][0]['values'][0][0]
    print(y_prediction)
    
	
	
    
    return y_prediction

if  __name__ == '__main__':
    print("Starting Flask Application")
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host='localhost', debug=False, threaded=False)
    print("Stopping Application")