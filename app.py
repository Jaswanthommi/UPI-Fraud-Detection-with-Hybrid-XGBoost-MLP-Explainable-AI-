from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os
import shap
from tensorflow.keras.models import load_model

# ===============================
# Flask App Initialization
# ===============================
app = Flask(__name__)

# ===============================
# Absolute Base Directory
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===============================
# Load Saved Models & Objects
# ===============================
xgb_model = joblib.load(
    os.path.join(BASE_DIR, "models", "xgboost_upi_fraud.pkl")
)

mlp_model = load_model(
    os.path.join(BASE_DIR, "models", "mlp_upi_fraud.h5")
)

scaler = joblib.load(
    os.path.join(BASE_DIR, "models", "scaler.pkl")
)

feature_columns = joblib.load(
    os.path.join(BASE_DIR, "models", "feature_columns.pkl")
)

# ===============================
# Initialize SHAP Explainer
# ===============================
# TreeExplainer works efficiently with XGBoost models
# It uses the tree structure directly, no background data needed
shap_explainer = shap.TreeExplainer(xgb_model)

# ===============================
# Home Route
# ===============================
@app.route('/')
def home():
    return render_template('index.html')

# ===============================
# Prediction Route
# ===============================
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')

    # -------- Safe Input Collection --------
    input_data = {
        "txn_type": request.form.get('txn_type', 'P2P'),
        "txn_amount": float(request.form.get('txn_amount', 0)),
        "day_of_week": request.form.get('day_of_week', 'Monday'),
        "hour_of_day": int(request.form.get('hour_of_day', 0)),
        "device_change_flag": int(request.form.get('device_change_flag', 0)),
        "geo_distance_from_home": float(request.form.get('geo_distance_from_home', 0)),
        "avg_daily_txn_amt": float(request.form.get('avg_daily_txn_amt', 0)),
        "txn_frequency_day": int(request.form.get('txn_frequency_day', 0)),
        "failed_txn_ratio_7d": float(request.form.get('failed_txn_ratio_7d', 0)),
        "suspicious_merchant_flag": int(request.form.get('suspicious_merchant_flag', 0)),
        "upi_app_used": request.form.get('upi_app_used', 'GooglePay'),
        "txn_channel": request.form.get('txn_channel', 'Mobile'),
        "is_night_txn": int(request.form.get('is_night_txn', 0)),
        "account_age_days": int(request.form.get('account_age_days', 0)),
        "fraud_probability_score": float(request.form.get('fraud_probability_score', 0))
    }

    # -------- Convert to DataFrame --------
    df = pd.DataFrame([input_data])

    # -------- One-Hot Encoding --------
    df = pd.get_dummies(df)

    # -------- Add Missing Columns --------
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # -------- Keep Correct Feature Order --------
    df = df[feature_columns]

    # -------- Feature Scaling --------
    scaled_data = scaler.transform(df)

    # -------- Model Predictions --------
    xgb_prob = xgb_model.predict_proba(scaled_data)[:, 1]
    mlp_prob = mlp_model.predict(scaled_data).ravel()

    # -------- Weighted Soft Voting --------
    final_prob = (0.6 * xgb_prob) + (0.4 * mlp_prob)
    final_prob = float(np.asarray(final_prob).ravel()[0])

    # -------- Final Decision --------
    prediction = "Fraudulent 🚨" if final_prob > 0.5 else "Legitimate ✅"

    # -------- SHAP Explainability --------
    # Compute SHAP values for the XGBoost model
    shap_values = shap_explainer.shap_values(scaled_data)
    
    # Handle different SHAP return formats
    if isinstance(shap_values, list):
        # Binary classification: list of arrays [negative_class, positive_class]
        shap_values_array = np.array(shap_values[1]).flatten()
    else:
        # Single array (already computed for positive class)
        shap_values_array = np.array(shap_values).flatten()
    
    # Create feature importance dictionary
    feature_importance = {}
    for i, feature_name in enumerate(feature_columns):
        feature_importance[feature_name] = float(shap_values_array[i])
    
    # Sort features by absolute SHAP value (most important first)
    sorted_features = sorted(
        feature_importance.items(), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )
    
    # Get top 10 most important features
    top_features = sorted_features[:10]
    
    # Prepare SHAP data for visualization
    # Create a list of dictionaries with feature name and SHAP value together
    top_features_list = [{'name': f[0], 'value': float(f[1])} for f in top_features]
    max_abs_shap = max([abs(f['value']) for f in top_features_list]) if top_features_list else 1.0
    
    # Add absolute value and bar width percentage to each feature
    for feature in top_features_list:
        feature['abs_value'] = abs(feature['value'])
        feature['bar_width'] = round((feature['abs_value'] / max_abs_shap * 100), 1) if max_abs_shap > 0 else 0
    
    # Count positive and negative features
    positive_count = sum(1 for f in top_features_list if f['value'] > 0)
    negative_count = sum(1 for f in top_features_list if f['value'] < 0)
    
    shap_data = {
        'features': top_features_list,
        'base_value': float(shap_explainer.expected_value),
        'max_abs_shap': max_abs_shap,
        'positive_count': positive_count,
        'negative_count': negative_count
    }

    return render_template(
        'result.html',
        prediction=prediction,
        probability=round(float(final_prob), 3),
        shap_data=shap_data,
        input_data=input_data
    )

# ===============================
# Run Server
# ===============================
if __name__ == '__main__':
    app.run(debug=True)
