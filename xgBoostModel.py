import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")

# Data preprocessing
features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

# Train-test split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, target_encoded, test_size=0.2, random_state=2)

# Train XGBoost classifier
xgboost_model = XGBClassifier()
xgboost_model.fit(Xtrain, Ytrain)

# Evaluate the XGBoost model
predicted_values_xb = xgboost_model.predict(Xtest)
xg_accuracy = accuracy_score(Ytest, predicted_values_xb)
print("XGBoost's Accuracy is:", xg_accuracy * 100)
print(classification_report(Ytest, predicted_values_xb))

# Save the trained XGBoost model and label encoder
joblib.dump(xgboost_model, 'xgboost_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("XGBoost model and Label Encoder saved successfully.")
