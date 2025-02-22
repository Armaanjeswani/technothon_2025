import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

# Load the updated dataset
data = pd.read_csv("D:\Psy\psy_final\physiomize_data_updated.csv")

# Convert categorical columns to numerical values if any
for col in data.select_dtypes(include=['object']).columns:
    data[col] = LabelEncoder().fit_transform(data[col])

# Drop target variable
X = data.drop(['Time for Complete ROM'], axis=1)
Y = data['Time for Complete ROM']

# Normalize target variable
max_time = Y.max()
Y = Y / max_time

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Reshape for LSTM input
X_train = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test.values, (X_test.shape[0], X_test.shape[1], 1))

# Define LSTM model
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=100))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X_train, Y_train, epochs=100, batch_size=32)

# Save model
model.save("psy_prediction_model.h5")

# Load model for inference
loaded_model = load_model("psy_prediction_model.h5")

# Scaling factor for prediction
scaling_factor = 0.8

# User input for prediction
num_weeks = int(input("Enter the number of weeks: "))
rom_values = []
for i in range(num_weeks):
    rom = float(input(f"Enter ROM value for week {i+1}: "))
    rom_values.append(rom)

pain_intensity = float(input("Enter pain intensity (1-10): "))
pain_x = float(input("Enter pain localization X: "))
pain_y = float(input("Enter pain localization Y: "))
pain_z = float(input("Enter pain localization Z: "))

input_data = rom_values + [pain_intensity, pain_x, pain_y, pain_z]

# Convert input to NumPy array and reshape
input_array = np.reshape(np.array(input_data), (1, len(input_data), 1))

# Make prediction
prediction = loaded_model.predict(input_array)
predicted_time = int(prediction[0][0] * max_time * scaling_factor)

print("Predicted time using loaded model:", predicted_time, "Weeks")
