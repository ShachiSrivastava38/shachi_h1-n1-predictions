import joblib
from flask import Flask, jsonify, render_template_string, request
import pandas as pd

app = Flask(__name__)

# Load the trained RandomForest model
MODEL_PATH = "model.pkl"  # Rename your .pkl file to model.pkl or adjust path
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Error loading model from {MODEL_PATH}: {e}")

# Explicit feature list expected by the model
FEATURE_NAMES = [
    "Unique_Id",
    "H1N1_Worry",
    "H1N1_Awareness",
    "Antiviral_Medication",
    "Contact_Avoidance",
    "Bought_Face_Mask",
    "Wash_Hands_Frequently",
    "Avoid_Large_Gatherings",
    "Reduced_Outside_Home_Cont",
    "Avoid_Touch_Face",
    "Dr_Recc_H1N1_Vacc",
    "Dr_Recc_Seasonal_Vacc",
    "Chronic_Medic_Condition",
    "Cont_Child_Undr_6_Mnths",
    "Is_Health_Worker",
    "Has_Health_Insur",
    "Is_H1N1_Vacc_Effective",
    "Is_H1N1_Risky",
    "Sick_From_H1N1_Vacc",
    "Is_Seas_Vacc_Effective",
    "Is_Seas_Risky",
    "Sick_From_Seas_Vacc",
    "Age_Bracket",
    "Qualification",
    "Race",
    "Sex",
    "Income_Level",
    "Marital_Status",
    "Housing_Status",
    "Employment",
    "Census_Msa",
    "No_Of_Adults",
    "No_Of_Children",
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>H1N1 Vaccine Prediction API</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f4f7f6; }
        .container { max-width: 700px; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        h2 { color: #333; }
        .form-group { margin-bottom: 12px; }
        label { display: inline-block; width: 220px; font-weight: bold; font-size: 0.9em; }
        input { padding: 6px; width: 200px; }
        button { margin-top: 15px; padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { margin-top: 20px; font-weight: bold; font-size: 1.1em; color: #28a745; }
    </style>
</head>
<body>
<div class="container">
    <h2>H1N1 Vaccine Model Inference</h2>
    <form method="POST" action="/predict">
        {% for feature in features %}
        <div class="form-group">
            <label for="{{ feature }}">{{ feature }}:</label>
            <input type="text" id="{{ feature }}" name="{{ feature }}" value="0" required>
        </div>
        {% endfor %}
        <button type="submit">Predict</button>
    </form>
    {% if prediction is not none %}
    <div class="result">
        Prediction: {{ prediction }}
    </div>
    {% endif %}
</div>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE, features=FEATURE_NAMES, prediction=None)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded properly"}), 500

    try:
        # Check if request is JSON or Web Form
        if request.is_json:
            data = request.get_json()
            df = pd.DataFrame([data])
        else:
            data = request.form.to_dict()
            df = pd.DataFrame([data])

        # Ensure all columns exist and maintain exact order
        for col in FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0

        df = df[FEATURE_NAMES]

        # Convert object columns to numeric where possible
        df = df.apply(pd.to_numeric, errors="ignore")

        # Get prediction
        prediction = model.predict(df)[0]

        if request.is_json:
            return jsonify({"prediction": int(prediction)})

        return render_template_string(
            HTML_TEMPLATE, features=FEATURE_NAMES, prediction=int(prediction)
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)