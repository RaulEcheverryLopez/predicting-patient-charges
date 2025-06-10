from flask import Flask, request, render_template, jsonify
from pycaret.regression import load_model, predict_model
import pandas as pd
import numpy as np

app = Flask(__name__)

model = load_model('models/model.joblib')  # Cambiado para buscar el modelo en la carpeta models
cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns=cols)
    prediction = predict_model(model, data=data_unseen, round=0)
    prediction = int(prediction['prediction_label'][0])
    return render_template('home.html', pred='Expected Bill will be {}'.format(prediction))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0') 