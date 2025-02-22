from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model("psy_prediction_model.h5")

# Load max_time from pickle file
with open("max_time.pkl", "rb") as file:
    max_time = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        num_weeks = int(request.form['num_weeks'])
        rom_values = [float(request.form[f'week_{i+1}']) for i in range(num_weeks)]

        pain_intensity = float(request.form['pain_intensity'])
        pain_x = float(request.form['pain_x'])
        pain_y = float(request.form['pain_y'])
        pain_z = float(request.form['pain_z'])
        age = int(request.form['age'])

        input_data = rom_values + [pain_intensity, pain_x, pain_y, pain_z, age]

        # Convert input to NumPy array and reshape
        input_array = np.reshape(np.array(input_data), (1, len(input_data), 1))

        # Make prediction
        prediction = model.predict(input_array)

        # Adjust predicted time based on age
        scaling_factor = 0.8
        age_factor = 1 + ((age - 18) * 0.005)  # Example: +0.5% per year over 18
        predicted_time = int(prediction[0][0] * max_time * scaling_factor * age_factor)

        return render_template('predict.html', prediction=predicted_time)

    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
