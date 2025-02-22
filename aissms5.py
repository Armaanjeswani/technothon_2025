import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

# Load the updated dataset
data = pd.read_csv(r"D:\Psy\psy_aissms\physiomize_data_final.csv")

# Convert categorical columns to numerical values if any
for col in data.select_dtypes(include=['object']).columns:
    data[col] = LabelEncoder().fit_transform(data[col])

# Drop target variable
X = data.drop(['Time for Complete ROM'], axis=1)
Y = data['Time for Complete ROM']

# Normalize target variable and save max_time
max_time = Y.max()

# Save max_time to a pickle file
with open("max_time.pkl", "wb") as file:
    pickle.dump(max_time, file)

Y = Y / max_time  # Normalize target variable

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Reshape for LSTM input
X_train = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test.values, (X_test.shape[0], X_test.shape[1], 1))

# Define LSTM model
model = Sequential([
    LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(units=100),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X_train, Y_train, epochs=100, batch_size=32)

# Save model
model.save("psy_prediction_model.h5")

# Load model for inference
loaded_model = load_model("psy_prediction_model.h5")

# Load max_time from pickle file
with open("max_time.pkl", "rb") as file:
    max_time = pickle.load(file)

# Scaling factor for prediction
scaling_factor = 0.8

# User input for prediction
num_weeks = int(input("Enter the number of weeks: "))
rom_values = [float(input(f"Enter ROM value for week {i+1}: ")) for i in range(num_weeks)]

pain_intensity = float(input("Enter pain intensity (1-10): "))
pain_x = float(input("Enter pain localization X: "))
pain_y = float(input("Enter pain localization Y: "))
pain_z = float(input("Enter pain localization Z: "))
age = int(input("Enter age of the patient: "))  # Age input

input_data = rom_values + [pain_intensity, pain_x, pain_y, pain_z, age]

# Convert input to NumPy array and reshape
input_array = np.reshape(np.array(input_data), (1, len(input_data), 1))

# Make prediction
prediction = loaded_model.predict(input_array)

# Adjust predicted time based on age (older patients take longer to recover)
age_factor = 1 + ((age - 18) * 0.005)  # Example: +0.5% time per year over 18
predicted_time = int(prediction[0][0] * max_time * scaling_factor * age_factor)

print("Predicted recovery time using loaded model:", predicted_time, "Weeks")
