import shap
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from data_preprocessing import load_and_prepare_data


# Load data
X_train, X_test, y_train, y_test = load_and_prepare_data()


# Create XGBoost model
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


# Train the model
model.fit(X_train, y_train)


# Create SHAP explainer
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)


# SHAP summary plot
plt.figure(figsize=(10, 8))

shap.summary_plot(
    shap_values,
    X_test,
    show=False
)

plt.title("SHAP Feature Importance - XGBoost")

plt.tight_layout()

plt.show()