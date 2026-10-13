import shap
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from data_preprocessing import load_and_prepare_data

# Load data
X_train, X_test, y_train, y_test = load_and_prepare_data()

# XGBoost model
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)

# Train model
model.fit(X_train, y_train)

# SHAP explainer
explainer = shap.TreeExplainer(model)

# Select 3 test samples
sample_indices = [0, 1, 2]

for i in sample_indices:

    sample = X_test.iloc[[i]]

    # Prediction
    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][prediction]

    print("\n-----------------------------")
    print("Test Sample:", i)
    print("Actual Class:", y_test.iloc[i])
    print("Predicted Class:", prediction)
    print("Confidence:", round(probability * 100, 2), "%")
    print("-----------------------------")

    # SHAP explanation
    shap_values = explainer(sample)

    # Waterfall plot
    shap.plots.waterfall(
        shap_values[0],
        max_display=15,
        show=False
    )

    plt.title(
        f"Local SHAP Explanation - Test Sample {i}"
    )

    plt.tight_layout()

    plt.savefig(
        f"shap_local_sample_{i}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()