# -*- coding: utf-8 -*-
"""For Deployment Solar_Panel_Regression_Group_4.ipynb"""

import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor
import streamlit as st

df = pd.read_csv('solarpowergeneration.csv')

df['average-wind-speed-(period)'] = df['average-wind-speed-(period)'].fillna(df['average-wind-speed-(period)'].mean())

df_features = df.drop(['power-generated'], axis=1)

scaled_df = df_features.copy()
for col in scaled_df.select_dtypes(include=[np.number]).columns:
    mean_val = scaled_df[col].mean()
    std_val = scaled_df[col].std()
    if std_val != 0:
        scaled_df[col] = (scaled_df[col] - mean_val) / std_val
    else:
        scaled_df[col] = 0.0

y = df['power-generated']
x = scaled_df

x_train = x.iloc[0:2336]
y_train = y.iloc[0:2336]
x_test = x.iloc[2336:2920]
y_test = y.iloc[2336:2920]

# Train the model
model = GradientBoostingRegressor(learning_rate=0.1, max_depth=3, n_estimators=100, random_state=42)
model.fit(x_train, y_train)

with open('gradient_boosting_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# --------------------------------------------------------
# BEAUTIFUL MODERN UI LAYOUT
# --------------------------------------------------------
import streamlit as st
import pickle
import pandas as pd

# Page Config
st.set_page_config(
    page_title="Solar Panel Regression App",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS Styling
st.markdown("""
    <style>
        .main {
            background-color: #f4f7fa;
        }
        .title-text {
            font-size: 40px;
            font-weight: 800;
            color: #1B7F79;
            text-align: center;
            padding-bottom: 10px;
        }
        .sub-text {
            text-align: center;
            font-size: 20px;
            color: #333333;
            margin-top: -15px;
            padding-bottom: 20px;
        }
        .input-card {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        }
        .prediction-box {
            background: #1B7F79;
            color: white;
            padding: 18px;
            border-radius: 12px;
            text-align: center;
            font-size: 24px;
            font-weight: 700;
            margin-top: 20px;
        }
    </style>
""",
    unsafe_allow_html=True
)

st.markdown("<h1 class='title-text'>⚡ Solar Panel Regression App</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Gradient Boosting Regression based Power Generation Prediction</p>", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    with open("gradient_boosting_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Means and Stds
means = {
    "distance-to-solar-noon": 0.0,
    "temperature": 0.0,
    "wind-direction": 0.0,
    "wind-speed": 0.0,
    "sky-cover": 0.0,
    "visibility": 0.0,
    "humidity": 0.0,
    "average-wind-speed-(period)": 0.0,
    "average-pressure-(period)": 0.0
}

stds = {
    "distance-to-solar-noon": 1.0,
    "temperature": 1.0,
    "wind-direction": 1.0,
    "wind-speed": 1.0,
    "sky-cover": 1.0,
    "visibility": 1.0,
    "humidity": 1.0,
    "average-wind-speed-(period)": 1.0,
    "average-pressure-(period)": 1.0
}

# --------------------------------------------------------
# INPUT FORM IN A CARD
# --------------------------------------------------------
st.markdown("<div class='input-card'>", unsafe_allow_html=True)
st.markdown("### 🌤 Enter Environmental Parameters")
st.markdown("Provide values for the solar panel environment to predict power output.")

cols = st.columns(3)

user_input = {}
feature_list = list(means.keys())

for i, feature in enumerate(feature_list):
    with cols[i % 3]:

        if feature == "distance-to-solar-noon":
            user_input[feature] = st.number_input(
                feature.replace("-", " ").title(),
                value=round(float(means[feature]), 15),
                format="%.15f"
            )
        else:
            user_input[feature] = st.number_input(
                feature.replace("-", " ").title(),
                value=float(means[feature])
            )

st.markdown("</div>", unsafe_allow_html=True)

user_df = pd.DataFrame([user_input])

# --------------------------------------------------------
# PROCESSING & PREDICTION
# --------------------------------------------------------
def manual_standard_scale(df):
    scaled = df.copy()
    for col in scaled.columns:
        mean_val = means[col]
        std_val = stds[col]
        scaled[col] = (scaled[col] - mean_val) / std_val if std_val != 0 else 0
    return scaled

scaled_user_df = manual_standard_scale(user_df)

# Prediction Button
if st.button("🔍 Predict Power Generation", use_container_width=True):
    prediction = model.predict(scaled_user_df)[0]

    st.markdown(
        f"<div class='prediction-box'>🌞 Predicted Power: <br>{prediction:.2f} kW</div>",
        unsafe_allow_html=True
    )
