from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the XGBoost model and label encoder
xgboost_model = joblib.load('xgboost_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        
        # Extract input values from the request data
        input_data = pd.DataFrame({
            'N': [data['N']],
            'P': [data['P']],
            'K': [data['K']],
            'temperature': [data['temperature']],
            'humidity': [data['humidity']],
            'ph': [data['ph']],
            'rainfall': [data['rainfall']]
        })
        
        
        # Make prediction
        predicted_label = xgboost_model.predict(input_data)
        predicted_crop = label_encoder.inverse_transform(predicted_label)
        
        # Return the prediction as a JSON response
        return jsonify({'recommended_crop': predicted_crop[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/test', methods=['POST','GET'])
def test():
    return jsonify({"message":"success"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
