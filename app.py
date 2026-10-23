from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the entire trained pipeline (preprocessor + model)
model_pipeline = joblib.load('diabetes_logistic_regression_model.joblib')

# Define the features that the model expects
EXPECTED_FEATURES = [
    'gender', 'age', 'hypertension', 'heart_disease', 
    'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json(force=True)
        
        # Handle both single dictionary and list of dictionaries
        if isinstance(data, dict):
            data = [data]
            
        # Convert input data to a Pandas DataFrame
        input_df = pd.DataFrame(data)
        
        # Check if all expected features exist in the input dataframe
        missing_cols = [col for col in EXPECTED_FEATURES if col not in input_df.columns]
        if missing_cols:
            return jsonify({
                'error': f'Missing fields: {missing_cols}',
                'required_fields': EXPECTED_FEATURES
            }), 400
            
        # Ensure correct column ordering matching the training stage
        input_df = input_df[EXPECTED_FEATURES]
        
        # Predict using the pre-fitted pipeline
        predictions = model_pipeline.predict(input_df)
        probabilities = model_pipeline.predict_proba(input_df)[:, 1]
        
        # Structure the final response
        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append({
                'diabetes_prediction': int(pred),
                'probability': float(prob)
            })
            
        return jsonify({'predictions': results}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app on local port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
