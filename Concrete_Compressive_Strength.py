#!/usr/bin/python3

# Title: Concrete Compressive Strength Prediction with Gradient Boosting, SVR, and MLP + XAI (LIME & SHAP)

# ----------------------------------------
# Import necessary libraries
# ----------------------------------------
import pandas as pd
import numpy as np
import os

# Models
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Explainability
import lime.lime_tabular
import shap

# Visualization
import matplotlib
matplotlib.use('Agg')  # ✅ Added for better plotting outside notebooks
import matplotlib.pyplot as plt

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# for model deployment
import streamlit as st
import joblib

# ----------------------------------------
# STEP 1: Load and Prepare Data
# ----------------------------------------
data = pd.read_csv("Concret_data/Concrete_Data.csv") 
data.columns = data.columns.str.strip().str.replace('"', '')
print("📋 Cleaned Columns:", data.columns.tolist())

# Features and target
X = data.drop(columns=["Concrete compressive strength(MPa, megapascals)"])
y = data["Concrete compressive strength(MPa, megapascals)"]

# Save unscaled versions for explainability
X_train_df, X_test_df, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standard scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_df)
X_test = scaler.transform(X_test_df)

# ----------------------------------------
# STEP 2: Define and Train Models
# ----------------------------------------
models = {
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
    "Support Vector Regressor": SVR(kernel='rbf', C=100, epsilon=0.2),
    "MLP Regressor": MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu', max_iter=1000, random_state=42)
}

trained_models = {}

for name, model in models.items():
    print(f"🚀 Training {name}...")
    model.fit(X_train, y_train)
    trained_models[name] = model
    if name == "Gradient Boosting":
        joblib.dump(model, "Gradient_Boosting_model.pkl")
        

joblib.dump(X.columns.tolist(), "feature_names.pkl")
joblib.dump(scaler, "scaler.pkl")  # Save the scaler for deployment

# ----------------------------------------
# streamlit Predict
def predict():
    st.title('🎓Concret Strength Prediction Model')
    st.markdown("🔢 Enter **numeric, comma-separated** values matching the model's features.")
    st.caption("You need to enter **exactly 8 values**, in the same order used in training.")
    input_feature = st.text_input("📥 Input Features:")
    with st.expander("🧱 Need a sample input?"):
        st.code("540.0,0.0,0.0,162.0,2.5,1040.0,676.0,28", language="text")
        st.caption("🔎 Paste this into the input box above to test a valid prediction for compressive strength.")
    if st.button("Predict"):
        try:
            # Load artifacts
            model = joblib.load("Gradient_Boosting_model.pkl")
            feature_names = joblib.load("feature_names.pkl")
            scaler = joblib.load("scaler.pkl")

            # Parse input and check feature count
            features = list(map(float, input_feature.strip().split(",")))
            if len(features) != 8:
                st.error("⚠️ Please enter exactly 8 numeric values.")
                return

            # Scale the input features
            scaled_features = scaler.transform([features])

            # Make prediction
            prediction = model.predict(scaled_features)
            st.success(f"✅ Predicted Compressive Strength: {prediction[0]:.2f} MPa")
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
# ----------------------------------------
# STEP 3: Evaluation
# ----------------------------------------
def evaluate_model(model, name):
    y_pred = model.predict(X_test)
    print(f"\n📈 Evaluation for {name}:")
    print(f"  MAE:  {mean_absolute_error(y_test, y_pred):.3f}")
    print(f"  MSE:  {mean_squared_error(y_test, y_pred):.3f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
    print(f"  R²:   {r2_score(y_test, y_pred):.3f}")

# ----------------------------------------
# STEP 4: LIME Explanation (✅ fixed with unscaled data)
# ----------------------------------------
def lime_explanation(model, name):
    print(f"\n🔍 LIME Explanation for {name}...")
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train_df),
        feature_names=X.columns.tolist(),
        mode="regression"
    )
    exp = explainer.explain_instance(X_test_df.iloc[0].values, model.predict, num_features=8)  # ✅ Use .values
    fig = exp.as_pyplot_figure()
    plt.title(f"LIME Explanation - {name}")
    plt.tight_layout()
    plt.show()
    plt.pause(3)
    plt.close()

# ----------------------------------------
# STEP 5: SHAP Explanation (✅ fixed with unscaled data and legacy plot fallback)
# ----------------------------------------
def shap_explanation(model, name):
    print(f"\n🔍 SHAP Explanation for {name}...")

    try:
        if name == "Gradient Boosting":
            explainer = shap.Explainer(model, X_train_df)  # ✅ Use unscaled
            shap_values = explainer(X_test_df.iloc[:1])
            shap.plots.waterfall(shap_values[0])
        else:
            background = shap.sample(X_train_df, 100, random_state=42)  # ✅ Use unscaled
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_test_df.iloc[:1].values)
            shap.plots._waterfall.waterfall_legacy(shap_values[0], feature_names=X.columns.tolist())  # ✅ Legacy plot
    except Exception as e:
        print(f"⚠️ SHAP explanation failed for {name}: {e}")

# ----------------------------------------
# STEP 6: Run Pipeline
# ----------------------------------------
if __name__ == "__main__":
    
    for name, model in trained_models.items():
        evaluate_model(model, name)
        lime_explanation(model, name)
        shap_explanation(model, name)

    predict()