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
# PAGE CONFIG
# --------------------------------------------------------
st.set_page_config(page_title="Solar Panel Regression App", layout="wide")

st.title("⚡ Solar Panel Regression App")
st.subheader("Power Generation Prediction")

# --------------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------------
@st.cache_resource
def load_model():
    with open("gradient_boosting_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# --------------------------------------------------------
# MEANS & STDS
# --------------------------------------------------------
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
# USER INPUT SECTION
# --------------------------------------------------------
def user_input_features():
    inputs = {}
    for feature in means.keys():

        # SPECIAL HANDLING FOR distance-to-solar-noon
        if feature == "distance-to-solar-noon":
            inputs[feature] = st.number_input(
                f"{feature.replace('-', ' ').title()}",
                value=round(float(means[feature]), 15),
                format="%.15f"
            )
        else:
            inputs[feature] = st.number_input(
                f"{feature.replace('-', ' ').title()}",
                value=float(means[feature])
            )

    return pd.DataFrame([inputs])

user_df = user_input_features()

# --------------------------------------------------------
# SCALING (IN BACKEND ONLY)
# --------------------------------------------------------
def manual_standard_scale(df):
    scaled = df.copy()
    for col in scaled.columns:
        mean_val = means[col]
        std_val = stds[col]
        if std_val != 0:
            scaled[col] = (scaled[col] - mean_val) / std_val
        else:
            scaled[col] = 0.0
    return scaled

scaled_user_df = manual_standard_scale(user_df)

# --------------------------------------------------------
# PREDICTION
# --------------------------------------------------------
if st.button("Predict Power Generation"):
    prediction = model.predict(scaled_user_df)[0]
    st.success(f"🌞 **Predicted Power Generated:** {prediction:.2f} kW")
