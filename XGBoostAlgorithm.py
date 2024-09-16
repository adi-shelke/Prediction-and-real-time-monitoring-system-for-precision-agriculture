import joblib
import pandas as pd

# Load the XGBoost model and label encoder
xgboost_model = joblib.load('xgboost_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Prompt user for input values
N_input = float(input("Enter Nitrogen (N) value: "))
P_input = float(input("Enter Phosphorus (P) value: "))
K_input = float(input("Enter Potassium (K) value: "))
temperature_input = float(input("Enter Temperature value: "))
humidity_input = float(input("Enter Humidity value: "))
ph_input = float(input("Enter pH value: "))
rainfall_input = float(input("Enter Rainfall value: "))

# Create a DataFrame with the input values
input_data = pd.DataFrame({
    'N': [N_input],
    'P': [P_input],
    'K': [K_input],
    'temperature': [temperature_input],
    'humidity': [humidity_input],
    'ph': [ph_input],
    'rainfall': [rainfall_input]
})

# Predict the crop using the loaded model
predicted_label = xgboost_model.predict(input_data)

# Decode the predicted label to get the crop name
predicted_crop = label_encoder.inverse_transform(predicted_label)

print(f"The recommended crop for the given soil parameters is: {predicted_crop[0]}")
