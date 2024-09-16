import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")

# Display basic information
print(df.head())
df.info()

# Check for missing values
print(df.isnull().sum())

# Plot histogram of temperature
plt.hist(df['temperature'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Temperature')
plt.ylabel('Frequency')
plt.title('Distribution of Temperature')
plt.show()

# Scatter plot of temperature vs. rainfall
plt.scatter(df['temperature'], df['rainfall'], color='green', alpha=0.5)
plt.xlabel('Temperature')
plt.ylabel('Rainfall')
plt.title('Scatter Plot of Temperature vs. Rainfall')
plt.show()

# Box plot of temperature for each crop label
plt.figure(figsize=(10, 6))
sns.boxplot(x='label', y='temperature', data=df)
plt.xlabel('Crop Label')
plt.ylabel('Temperature')
plt.title('Box Plot of Temperature for Each Crop Label')
plt.xticks(rotation=45)
plt.show()

# Correlation matrix heatmap
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Count plot of each crop label
plt.figure(figsize=(10, 6))
sns.countplot(x='label', data=df, palette='Set3')
plt.xlabel('Crop Label')
plt.ylabel('Count')
plt.title('Count of Each Crop Label')
plt.xticks(rotation=45)
plt.show()

# Box plot comparison of N, P, K values
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['N', 'P', 'K']], palette='Set3')
plt.title('Comparison of N, P, K')
plt.xlabel('Nutrient')
plt.ylabel('Value')
plt.show()

# Bar plot comparison of N, P, K values across different crops
dfm = df.melt(id_vars=['label'], value_vars=['N', 'P', 'K'], var_name='Nutrient', value_name='Value')
plt.figure(figsize=(12, 8))
sns.barplot(data=dfm, x='label', y='Value', hue='Nutrient', dodge=True)
plt.xlabel('Crops')
plt.ylabel('Values')
plt.title('Comparison of N, P, K across different crops')
plt.xticks(rotation=90)
plt.legend(title='Nutrient', title_fontsize='14')
plt.tight_layout()
plt.show()

# Data preprocessing
features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

# Train-test split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, target_encoded, test_size=0.2, random_state=2)

# Decision Tree classifier
DecisionTree = DecisionTreeClassifier(criterion="entropy", random_state=2, max_depth=5)
DecisionTree.fit(Xtrain, Ytrain)
predicted_values = DecisionTree.predict(Xtest)
dt_accuracy = accuracy_score(Ytest, predicted_values)
print("Decision Tree's accuracy is:", dt_accuracy * 100)

# Cross-validation for Decision Tree
dt_cross_val_score = cross_val_score(DecisionTree, features, target, cv=5)
print("Decision Tree cross-validation scores:", dt_cross_val_score)

# XGBoost classifier
XB = XGBClassifier()
XB.fit(Xtrain, Ytrain)
predicted_values_xb = XB.predict(Xtest)
xg_accuracy = accuracy_score(Ytest, predicted_values_xb)
print("XGBoost's Accuracy is:", xg_accuracy * 100, "%")

# Classification report for XGBoost
print(classification_report(Ytest, predicted_values_xb))

# Accuracy comparison plot
acc = [dt_accuracy, xg_accuracy]
model = ['Decision Tree', 'XGBoost']
plt.figure(figsize=[10, 5], dpi=100)
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
sns.barplot(x=acc, y=model, palette='dark')
plt.show()

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

# Use the trained Decision Tree model to make a prediction
predicted_label = DecisionTree.predict(input_data)

# Decode the predicted label
predicted_crop = label_encoder.inverse_transform(predicted_label)
print(f"The recommended crop for the given soil parameters is: {predicted_crop[0]}")