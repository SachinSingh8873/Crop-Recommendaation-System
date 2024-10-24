from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the model and the scaler
model = pickle.load(open('crop_recommendation_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve input data from the form
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temperature = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        pH = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])

        # Preprocessing and prediction
        input_data = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)

        # Render template with prediction
        return render_template('index.html', prediction_text=f'{prediction[0]}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')


if __name__ == "__main__":
    app.run(debug=True)
