# 🛡️ PhishGuard — ML-Based Phishing URL Detection

PhishGuard is a machine learning project that investigates phishing URL detection through comparative model evaluation, feature reduction, and Explainable AI (XAI). It combines research notebooks with a Flask web application that classifies URLs using XGBoost and provides SHAP-based explanations of individual predictions.

## 🎯 Objectives

- Compare datasets with different feature dimensionalities.
- Evaluate machine learning models across original and reduced feature sets.
- Investigate the trade-off between classification performance, feature extraction feasibility, and computational efficiency.
- Build a web application for interactive URL classification.
- Improve interpretability using SHAP explanations.

## 🔬 Research Methodology

### 1. Dataset and Feature-Set Comparison

The research investigated two dataset configurations containing **48 and 111 features**. Reduced-feature variants containing **25 and 50 features**, respectively, were also evaluated.

| Configuration | Features |
|---|---:|
| Original compact dataset | 48 |
| Reduced compact dataset | 25 |
| Original high-dimensional dataset | 111 |
| Reduced high-dimensional dataset | 50 |

This produced four experimental configurations for comparative evaluation.

### 2. Model Evaluation

**Four machine learning models were evaluated across all four configurations.** Classification performance was examined using metrics including accuracy, precision, recall, F1-score, and ROC-AUC.

The comparison also considered practical deployment requirements:

- Feasibility and complexity of extracting features from live URLs.
- Dependence on external DNS, WHOIS, HTTP, and TLS lookups.
- Computational overhead and time required for analysis.
- Suitability for integration into an interactive web application.

### 3. Feature Selection and Final Configuration

The research explored feature reduction to investigate whether a smaller feature set could retain useful predictive information while simplifying the overall pipeline.

After comparative evaluation and consideration of deployment practicality, the **111-feature dataset reduced to 50 selected features** was chosen for the final prototype.

### 4. Final Model Performance

The deployed XGBoost classifier recorded the following evaluation results:

| Metric | Score |
|---|---:|
| Accuracy | 96.62% |
| Precision | 93.66% |
| Recall | 96.91% |
| F1-score | 95.26% |
| ROC-AUC | 99.47% |

These results describe performance under the research evaluation setup and do not guarantee equivalent performance on previously unseen URLs.

## 🌐 Web Application

The research model was integrated into a Flask application for interactive URL analysis.

### Workflow

1. The user submits a URL.
2. The application validates the input.
3. The feature extraction module derives the required URL and domain-related features.
4. The extracted features are arranged in the order expected by the trained model.
5. The XGBoost classifier generates a prediction and class probabilities.
6. SHAP generates a local feature-contribution visualization.
7. The interface displays the results.

### Features

- URL input and validation.
- Automated extraction of model-required features.
- Legitimate or phishing classification.
- Prediction confidence and class probabilities.
- SHAP-based explanations for individual predictions.
- Graceful error handling when required features cannot be retrieved.

## 🧠 Explainable AI with SHAP

SHAP (SHapley Additive exPlanations) is used to examine how individual features contribute to the model's output.

The research incorporates feature analysis, while the web application displays a local explanation for the submitted URL. This helps users inspect the model's behaviour rather than relying solely on its predicted class.

SHAP explains the model's output; it does not independently establish whether a website is malicious.

## 🛠️ Tech Stack

- **Language:** Python
- **Data analysis:** Pandas, NumPy
- **Machine learning:** Scikit-learn, XGBoost
- **Explainable AI:** SHAP
- **Web framework:** Flask
- **Frontend:** HTML, CSS
- **Visualization:** Matplotlib
- **Feature extraction:** Requests, dnspython, python-whois
- **Model persistence:** Joblib
- **Research:** Jupyter Notebooks

## 📂 Repository Structure

```text
PhishGuard/
├── data/                   # Research datasets
├── model_exports/          # Trained model and configuration
├── notebooks/              # Research and model evaluation
├── output_images/          # Research and application figures
├── templates/              # Flask HTML templates
├── v1/                     # Earlier implementation
├── app.py                  # Flask application
├── feature_extractor.py    # URL and domain feature extraction
├── shap_explainer.py       # SHAP explanation functionality
├── requirements.txt        # Python dependencies
└── README.md
```

## ⚙️ Installation and Usage

### Prerequisites

- Python 3.11
- Git

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd PhishGuard
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

### 5. Open the website

Visit:

```text
http://127.0.0.1:5000
```

Ensure the required trained model and configuration files are present in `model_exports/` before starting the application.

## ⚠️ Limitations

- Classification performance on a test dataset may differ from performance on real-world URLs.
- External lookups may fail or introduce latency.
- Live feature extraction must remain consistent with the definitions used during training.
- Prediction probabilities are model estimates, not guarantees of safety.
- SHAP explanations describe model behaviour rather than proving causation.

PhishGuard is an academic prototype and should not be used as the sole basis for cybersecurity decisions.

## 🚀 Future Improvements

- Validate live feature extraction against the original dataset definitions.
- Improve handling of DNS and domain-information lookup failures.
- Evaluate feature reduction and model performance on additional datasets.
- Improve inference efficiency and local SHAP visualizations.
- Expand testing on previously unseen URLs.

---

*PhishGuard explores the intersection of machine learning, cybersecurity, feature engineering, and explainable AI through research and practical implementation.*
