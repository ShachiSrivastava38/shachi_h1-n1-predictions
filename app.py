import logging
import os
import urllib.request
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template_string, request, url_for

app = Flask(__name__)

# Configure basic logging to stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Load the trained RandomForest model
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)
except Exception as e:
    model = None
    logger.exception("Error loading model from %s: %s", MODEL_PATH, e)

# Ensure static directory exists and a vaccine image is present for the UI
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
VACCINE_PNG_PATH = os.path.join(STATIC_DIR, "vaccine.png")
VACCINE_SVG_PATH = os.path.join(STATIC_DIR, "vaccine.svg")

if not os.path.exists(VACCINE_SVG_PATH) and not os.path.exists(VACCINE_PNG_PATH):
    try:
        urllib.request.urlretrieve(
            "https://upload.wikimedia.org/wikipedia/commons/6/6a/Vaccine_injection.jpg",
            VACCINE_PNG_PATH,
        )
        logger.info("Downloaded default vaccine image to %s", VACCINE_PNG_PATH)
    except Exception:
        logger.exception("Failed to download default vaccine image")

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
        .container { max-width: 900px; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin: auto; }
        h2 { color: #333; }
        .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .form-group { margin-bottom: 12px; }
        label { display: block; font-weight: bold; font-size: 0.85em; margin-bottom: 3px; }
        input { padding: 6px; width: 90%; border: 1px solid #ccc; border-radius: 4px; }
        button { margin-top: 20px; padding: 12px 24px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 1em; width: 100%; }
        button:hover { background: #0056b3; }
        .result { margin-top: 20px; padding: 15px; font-weight: bold; font-size: 1.2em; color: #155724; background-color: #d4edda; border-color: #c3e6cb; border-radius: 4px; text-align: center; }
        .header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; }
        .header img { max-width: 140px; border-radius: 8px; }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h2>H1N1 Vaccine Model Inference</h2>
        <img src="{{ url_for('static', filename=image_file) }}" alt="Vaccine Image" />
    </div>
    <form method="POST" action="/predict">
        <div class="form-grid">
            {% for feature in features %}
            <div class="form-group">
                <label for="{{ feature }}">{{ feature }}:</label>
                <input type="text" id="{{ feature }}" name="{{ feature }}" value="0" required>
            </div>
            {% endfor %}
        </div>
        <button type="submit">Predict</button>
    </form>
    {% if prediction is not none %}
    <div class="result">
        Prediction Output: {{ prediction }}
    </div>
    {% endif %}
</div>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    image_file = "vaccine.svg" if os.path.exists(VACCINE_SVG_PATH) else "vaccine.png"
    return render_template_string(
        HTML_TEMPLATE,
        features=FEATURE_NAMES,
        prediction=None,
        image_file=image_file,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded properly. Ensure model.pkl exists."}), 500

    try:
        # Check payload type
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        # Build DataFrame with proper column ordering
        df = pd.DataFrame([data])

        # Fill missing columns with 0
        for col in FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0

        df = df[FEATURE_NAMES]

        # Explicitly convert strings to numeric values where applicable
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Predict outcome
        prediction = model.predict(df)[0]
        prediction_val = int(prediction)

        if request.is_json:
            return jsonify({"prediction": prediction_val})

        image_file = "vaccine.svg" if os.path.exists(VACCINE_SVG_PATH) else "vaccine.png"
        return render_template_string(
            HTML_TEMPLATE,
            features=FEATURE_NAMES,
            prediction=prediction_val,
            image_file=image_file,
        )

    except Exception as e:
        logger.exception("Prediction error: %s", e)
        return jsonify({"error": str(e)}), 400


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH if model is not None else None,
    })


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host=host, port=port, debug=debug, use_reloader=False)
