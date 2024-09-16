import requests

url = 'http://localhost:5000/predict'
data = {
    "N": 77,
    "P": 57,
    "K": 21,
    "temperature": 24,
    "humidity": 73,
    "ph": 6.5,
    "rainfall": 80.0
}

response = requests.post(url, json=data)
print(response.json())
