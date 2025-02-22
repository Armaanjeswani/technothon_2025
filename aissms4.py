import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from bokeh.io import output_file, show
from bokeh.plotting import figure, ColumnDataSource
from bokeh.layouts import column, layout, gridplot
import matplotlib.pyplot as plt

# Load the updated dataset
data = pd.read_csv("D:\Psy\psy_final\physiomize_data_updated.csv")

# Convert categorical columns to numerical values if any
for col in data.select_dtypes(include=['object']).columns:
    data[col] = LabelEncoder().fit_transform(data[col])

# Drop target variable
X = data.drop(['Time for Complete ROM'], axis=1)
Y = data['Time for Complete ROM']

# Feature Scaling
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)
joblib.dump(scaler_X, "scaler_X.pkl")

scaler_Y = MinMaxScaler()
Y = scaler_Y.fit_transform(Y.values.reshape(-1, 1)).flatten()
joblib.dump(scaler_Y, "scaler_Y.pkl")

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Reshape for LSTM input
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Define LSTM model
model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(LSTM(units=64))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile model
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)

# Train model
model.fit(X_train, Y_train, validation_split=0.1, epochs=100, batch_size=32, callbacks=[early_stopping, reduce_lr])

# Save model
model.save("psy_prediction_model.h5")

# Load model for inference
loaded_model = load_model("psy_prediction_model.h5")

# Model Evaluation
Y_pred = loaded_model.predict(X_test).flatten()
Y_pred = scaler_Y.inverse_transform(Y_pred.reshape(-1, 1)).flatten()
Y_test = scaler_Y.inverse_transform(Y_test.reshape(-1, 1)).flatten()

mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_pred)

print(f"Model Evaluation:\nMAE: {mae}\nMSE: {mse}\nRMSE: {rmse}\nR2 Score: {r2}")

# Plot error graph using Bokeh
error = Y_test - Y_pred
output_file("error_plot.html")
p = figure(title="Prediction Error", x_axis_label="Index", y_axis_label="Error")
p.line(list(range(len(Y_test))), error, legend_label="Error", line_width=2)
show(p)
