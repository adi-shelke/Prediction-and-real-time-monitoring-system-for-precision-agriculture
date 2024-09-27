const data = {
    N: 77,
    P: 57,
    K: 21,
    temperature: 24,
    humidity: 73,
    ph: 6.5,
    rainfall: 80.0
};

// Define the API endpoint
const url = 'http://prediction-and-real-time-monitoring.onrender.com:5000/predict';

// Send POST request using fetch
fetch(url, {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
})
    .then(response => response.json())  // Parse the response as JSON
    .then(data => {
        console.log('Prediction Response:', data);
    })
    .catch(error => {
        console.error('Error:', error.message);
    });
