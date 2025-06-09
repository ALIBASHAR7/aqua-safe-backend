from flask_cors import CORS
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os # Ensure this line is present and not commented out!


app = Flask(__name__)

# IMPORTANT: During deployment, you'll update this with your deployed frontend's URL.
# For now, allowing all origins is useful for initial testing, but not recommended for production.
# Replace '*' with your GitHub Pages URL (e.g., 'https://yourusername.github.io') for security.
CORS(app, origins=["http://localhost:3000", "https://aqua-safe-frontend.onrender.com", "https://*.github.io"]) # Add your GitHub Pages URL here


# Load the model
model = joblib.load('../models/water_quality_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
    except Exception as e:
        return jsonify({'error': f'Invalid input format: {str(e)}'}), 400
    
    # Check if all required fields are present
    required_fields = ['pH', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing field: {field}'}), 400

    # Convert input data to the format expected by the model
    try:
        input_features = np.array([[  # Ensure data is in the right shape
            data['pH'],
            data['Hardness'],
            data['Solids'],
            data['Chloramines'],
            data['Sulfate'],
            data['Conductivity'],
            data['Organic_carbon'],
            data['Trihalomethanes'],
            data['Turbidity']
        ]], dtype=float)  # Convert to float for model compatibility
    except ValueError as e:
        return jsonify({'error': f'Invalid input data: {str(e)}'}), 400
    except KeyError as e: # Add KeyError handling for robustness
        return jsonify({'error': f'Missing or misspelled input key: {str(e)}'}), 400
    
    # Perform the prediction
    try:
        prediction = model.predict(input_features)
    except Exception as e:
        return jsonify({'error': f'Model prediction error: {str(e)}'}), 500

    # Return the prediction result
    return jsonify({'Potability': int(prediction[0])})  # Convert to integer for JSON response

if __name__ == '__main__':
    print("Starting Flask app...")
    # For local development, use app.run
    # For Render, Gunicorn will handle running the app, so this block won't be used there.
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
