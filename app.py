```python
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained preprocessing + model pipeline
MODEL_PATH = "diabetes_logistic_regression_model.joblib"

EXPECTED_FEATURES = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]

try:
    model_pipeline = joblib.load(MODEL_PATH)
except (OSError, Exception) as error:
    raise RuntimeError(f"Failed to load model: {error}") from error


@app.route("/predict", methods=["POST"])
def predict():
    # Validate JSON input
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({
            "error": "Invalid or missing JSON request body."
        }), 400

    # Accept either one record or a list of records
    if isinstance(data, dict):
        data = [data]
    elif not isinstance(data, list) or not data:
        return jsonify({
            "error": "Input must be a JSON object or a non-empty list of objects."
        }), 400

    # Validate individual records
    if not all(isinstance(record, dict) for record in data):
        return jsonify({
            "error": "Every item in the input list must be a JSON object."
        }), 400

    # Check required features
    missing_fields = [
        feature
        for feature in EXPECTED_FEATURES
        if any(feature not in record for record in data)
    ]

    if missing_fields:
        return jsonify({
            "error": "Required fields are missing.",
            "missing_fields": missing_fields,
            "required_fields": EXPECTED_FEATURES,
        }), 400

    try:
        # Create DataFrame with the expected feature order
        input_df = pd.DataFrame(data, columns=EXPECTED_FEATURES)

        # Generate predictions
        predictions = model_pipeline.predict(input_df)

        # Generate probabilities when supported by the model
        if not hasattr(model_pipeline, "predict_proba"):
            return jsonify({
                "error": "The loaded model does not support probability predictions."
            }), 500

        probabilities = model_pipeline.predict_proba(input_df)

        # Find the probability column corresponding to class 1
        classes = list(model_pipeline.classes_)

        if 1 not in classes:
            return jsonify({
                "error": "The model does not contain class 1."
            }), 500

        positive_class_index = classes.index(1)

        # Format results
        results = [
            {
                "diabetes_prediction": int(prediction),
                "probability": float(probability[positive_class_index]),
            }
            for prediction, probability in zip(predictions, probabilities)
        ]

        return jsonify({
            "predictions": results
        }), 200

    except (ValueError, TypeError) as error:
        return jsonify({
            "error": "Invalid input data.",
            "details": str(error),
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")

        return jsonify({
            "error": "An internal error occurred while generating predictions."
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )
```
