import streamlit as st
import pickle
import pandas as pd

# Load trained model
with open("logistic_regression_model.pkl", "rb") as file:
    model = pickle.load(file)

# Manual training means and stds
training_means = {
    'Pclass': 2.308642,
    'Sex': 0.647587,
    'Age': 29.735277,
    'SibSp': 0.523008,
    'Parch': 0.381594,
    'Fare': 32.204208,
    'Cabin': 6.716049,
    'Embarked': 1.536476
}

training_stds = {
    'Pclass': 0.836071,
    'Sex': 0.477990,
    'Age': 13.002218,
    'SibSp': 1.102743,
    'Parch': 0.806057,
    'Fare': 49.693429,
    'Cabin': 2.460739,
    'Embarked': 0.791503
}

# Label encodings
sex_label = {'male': 1, 'female': 0}
embarked_label = {'S': 2, 'C': 0, 'Q': 1}
cabin_label = {'U': 8, 'C': 2, 'B': 1, 'D': 3, 'E': 4, 'A': 0, 'F': 5, 'G': 6, 'T': 7}

# Title
st.title("🚢 Titanic Survival Predictor")
st.markdown("Predict the likelihood of survival for a Titanic passenger based on their details.")

# Layout: use columns for compact UI
col1, col2, col3 = st.columns(3)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex_str = st.selectbox("Sex", list(sex_label.keys()))
    age = st.slider("Age", 0, 100, 30)

with col2:
    sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
    parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
    fare = st.number_input("Fare", 0, 600, 30)

with col3:
    cabin_str = st.selectbox("Cabin Deck", list(cabin_label.keys()))
    embarked_str = st.selectbox("Port of Embarkation", list(embarked_label.keys()))

# Predict button
if st.button("Predict"):
    # Encode inputs
    sex = sex_label[sex_str]
    cabin = cabin_label[cabin_str]
    embarked = embarked_label[embarked_str]

    # Raw input
    input_data = {
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Cabin': cabin,
        'Embarked': embarked
    }

    # Manual standard scaling
    scaled_input = {feature: (input_data[feature] - training_means[feature]) / training_stds[feature]
                    for feature in input_data}

    # Create DataFrame
    scaled_df = pd.DataFrame([scaled_input])

    # Predict
    prediction = model.predict(scaled_df)
    prob = model.predict_proba(scaled_df)[0][1]

    # Output
    if prediction[0] == 1:
        st.success(f"✅ The passenger is likely to **survive** with a probability of `{prob:.2f}`.")
    else:
        st.error(f"❌ The passenger is likely to **not survive** with a probability of `{1 - prob:.2f}`.")

# =========================================================
#  FOOTER
# =========================================================

st.markdown("<p class='footer-text'>App Created by <b>Manish Maltare</b></p>", unsafe_allow_html=True)
