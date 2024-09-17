from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

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
        
        # Get probabilities for each crop
        probabilities = xgboost_model.predict_proba(input_data)
        
        # Get top 3 crops and their probabilities
        top_n = 3
        top_n_indices = np.argsort(probabilities, axis=1)[:, -top_n:][:, ::-1]  # Sort and get indices of top 3 crops
        top_n_probs = np.take_along_axis(probabilities, top_n_indices, axis=1)  # Get probabilities of top 3 crops
        
        # Decode the crop labels to their original string values
        top_n_crops = label_encoder.inverse_transform(top_n_indices[0])
        
        # Prepare the result with crop names and suitability percentages
        recommendations = [
            {'crop': crop, 'suitability': round(prob * 100, 2)} 
            for crop, prob in zip(top_n_crops, top_n_probs[0])
        ]
        
        # Return the top 3 recommendations as a JSON response
        return jsonify({'recommendations': recommendations})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/test', methods=['POST', 'GET'])
def test():
    return jsonify({"message": "success"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
