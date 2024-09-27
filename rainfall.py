import pandas as pd

# Load the CSV file
file_path = 'crop_recommendation.csv'  # Replace with the actual file path
data = pd.read_csv(file_path)

# Find the maximum and minimum values of the 'rainfall' column
max_rainfall = data['rainfall'].max()
min_rainfall = data['rainfall'].min()

# Print the results
print(f"Maximum Rainfall: {max_rainfall}")
print(f"Minimum Rainfall: {min_rainfall}")
# Find the maximum and minimum values of the 'pH' column
max_ph = data['ph'].max()
min_ph = data['ph'].min()

# Print the results
print(f"Maximum pH: {max_ph}")
print(f"Minimum pH: {min_ph}")

