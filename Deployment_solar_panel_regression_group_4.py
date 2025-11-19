# -*- coding: utf-8 -*-
"""For Deployment Solar_Panel_Regression_Group_4.ipynb"""

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import streamlit as st

# =========================================================
# 1️⃣ LOAD CSV AND TRAIN MODEL
# =========================================================

df = pd.read_csv('solarpowergeneration.csv')

# Fill missing values
df['average-wind-speed-(period)'] = df['average-wind-speed-(period)'].fillna(
    df['average-wind-speed-(period)'].mean()
)

# Separate X and y
X = df.drop(['power-generated'], axis=1)
y = df['power-generated']

# ---------- Manual Split (80% Train / 20% Test) ----------
np.random.seed(42)
n_samples = len(X)
indices = np.random.permutation(n_samples)

train_size = int(0.80 * n_samples)
train_idx = indices[:train_size]
test_idx = indices[train_size:]

X_train = X.iloc[train_idx]
y_train = y.iloc[train_idx]

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

# ---------- Fit ACTUAL STANDARD SCALER ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- Train Model ----------
model = GradientBoostingRegressor(
    learning_rate=0.1,
    max_depth=3,
    n_estimators=100,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# ---------- Save Model + Scaler ----------
with open('gradient_boosting_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)


# =========================================================
# 2️⃣ STREAMLIT UI
# =========================================================

st.set_page_config(
    page_title="Solar Panel Regression App",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
        .main { background-color: #f4f7fa; }
        .title-text { font-size: 40px; font-weight: 800; color: #1B7F79; text-align: center; padding-bottom: 10px; }
        .sub-text { text-align: center; font-size: 20px; color: #333; margin-top: -15px; padding-bottom: 20px; }
        .input-card { background: white; padding: 25px; border-radius: 15px; box-shadow: 0px 4px 12px rgba(0,0,0,0.1); }
        .prediction-box { background: #1B7F79; color: white; padding: 18px; border-radius: 12px; 
                          text-align: center; font-size: 24px; font-weight: 700; margin-top: 20px; }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='title-text'>⚡ Solar Panel Regression App</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Gradient Boosting based Power Generation Prediction</p>", unsafe_allow_html=True)


# =========================================================
# 3️⃣ LOAD MODEL + SCALER
# =========================================================

@st.cache_resource
def load_artifacts():
    with open("gradient_boosting_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler

model, scaler = load_artifacts()


# =========================================================
# 4️⃣ INPUT UI
# =========================================================

st.markdown("<div class='input-card'>", unsafe_allow_html=True)
st.markdown("### 🌤 Enter Environmental Parameters")
st.markdown("Provide values to predict solar power generation.")

cols = st.columns(3)
features = list(X.columns)

user_inputs = {}
for i, feature in enumerate(features):
    with cols[i % 3]:

        # Special 15-decimal field
        if feature == "distance-to-solar-noon":
            user_inputs[feature] = st.number_input(
                feature.title(),
                value=0.0,
                format="%.15f"
            )
        else:
            user_inputs[feature] = st.number_input(
                feature.title(),
                value=0.0
            )

st.markdown("</div>", unsafe_allow_html=True)

user_df = pd.DataFrame([user_inputs])


# =========================================================
# 5️⃣ APPLY SCALER + PREDICT
# =========================================================

if st.button("🔍 Predict Power Generation", use_container_width=True):

    raw_values = user_df.to_numpy().flatten()

    # If all zeros → prediction = zero
    if np.allclose(raw_values, 0.0):
        st.info("All inputs are zero — predicted power = 0 kW")
        st.markdown("<div class='prediction-box'>🌞 Predicted Power: <br>0.00 kW</div>", 
                    unsafe_allow_html=True)

    else:
        scaled_input = scaler.transform(user_df)
        prediction = model.predict(scaled_input)[0]

        st.markdown(
            f"<div class='prediction-box'>🌞 Predicted Power: <br>{prediction:.2f} kW</div>",
            unsafe_allow_html=True
        )
