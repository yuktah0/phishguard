
import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from feature_extractor import (
    FEATURE_COLUMNS,
    extract_url_features,
    get_missing_features,
)

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model_exports"

MODEL_PATH = MODEL_DIR / "phishguard_xgboost.joblib"
CONFIG_PATH = MODEL_DIR / "phishguard_config.json"

model = joblib.load(MODEL_PATH)

with open(CONFIG_PATH, encoding="utf-8") as file:
    config = json.load(file)

if list(FEATURE_COLUMNS) != list(model.feature_names_in_):
    raise RuntimeError(
        "Feature columns do not match the trained model's feature order."
    )

SHAP_DIR = BASE_DIR / "static"
SHAP_DIR.mkdir(exist_ok=True)

shap_explainer = shap.TreeExplainer(model)
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/check", methods=["POST"])
def check_url():
    url = request.form.get("url", "").strip()

    if not url:
        return render_template(
            "index.html",
            error="Please enter a URL.",
        )

    try:
        features = extract_url_features(url)
        missing = get_missing_features(features)

        if missing:
            return render_template(
                "index.html",
                error=(
                    "Some required features could not be retrieved: "
                    + ", ".join(missing)
                    + ". Please try again later."
                ),
                submitted_url=url,
            )

        input_df = pd.DataFrame(
            [[features[name] for name in FEATURE_COLUMNS]],
            columns=FEATURE_COLUMNS,
        )

        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]

        classes = list(model.classes_)
        phishing_index = classes.index(1)
        legitimate_index = classes.index(0)

        phishing_probability = float(probabilities[phishing_index])
        legitimate_probability = float(probabilities[legitimate_index])

        result = (
            "Phishing" if prediction == 1 else "Legitimate"
        )
        

        confidence = float(
            probabilities[classes.index(prediction)]
        ) * 100
        # Generate a local SHAP explanation for this URL.
        shap_values = shap_explainer(input_df)

        plt.figure()
        shap.plots.bar(
            shap_values[0],
            max_display=12,
            show=False,
        )
        plt.title("Features Influencing This Prediction")
        plt.tight_layout()

        shap_path = SHAP_DIR / "shap_explanation.png"
        plt.savefig(shap_path, dpi=150, bbox_inches="tight")
        plt.close()

        return render_template(
            "index.html",
            submitted_url=url,
            result=result,
            confidence=round(confidence, 2),
            phishing_probability=round(
                phishing_probability * 100, 2
            ),
            legitimate_probability=round(
                legitimate_probability * 100, 2
            ),
        )

    except (ValueError, RuntimeError) as exc:
        return render_template(
            "index.html",
            error=str(exc),
            submitted_url=url,
        )
    except Exception:
        app.logger.exception("URL analysis failed")
        return render_template(
            "index.html",
            error=(
                "Analysis failed unexpectedly. "
                "Check the terminal for details and try again."
            ),
            submitted_url=url,
        )


if __name__ == "__main__":
    app.run(debug=True)
