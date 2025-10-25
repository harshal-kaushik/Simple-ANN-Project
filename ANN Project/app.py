import numpy as np
import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder
import pickle

## load the encoder and scalar
with open('label_encoder.pkl', 'rb') as file:
    labels_encoder_gender = pickle.load(file)
with open('one_hot_encoder.pkl', 'rb') as file:
    one_hot_encoder = pickle.load(file)
with open('scalar.pkl', 'rb') as file:
    scalar = pickle.load(file)

# load the training model
model = tf.keras.models.load_model('model.h5')


## streamlit app
import streamlit as st
import pandas as pd

# Page setup
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered",
)

# --- Custom CSS for a professional look ---
st.markdown("""
    <style>
        body {
            background: linear-gradient(120deg, #f6f9fc, #e9effd);
        }
        .main {
            background-color: white;
            border-radius: 15px;
            padding: 30px 40px;
            box-shadow: 0px 4px 25px rgba(0, 0, 0, 0.08);
        }
        .title {
            font-size: 2rem;
            font-weight: 600;
            color: #1a237e;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #555;
            text-align: center;
            font-size: 1rem;
            margin-bottom: 30px;
        }
        .prediction-box {
            text-align: center;
            padding: 18px;
            border-radius: 10px;
            margin-top: 25px;
            font-size: 1.1rem;
            font-weight: 500;
        }
        .churn {
            background-color: #ffebee;
            color: #c62828;
            border: 1px solid #ef9a9a;
        }
        .no-churn {
            background-color: #e8f5e9;
            color: #2e7d32;
            border: 1px solid #a5d6a7;
        }
        .stButton>button {
            background-color: #1a237e;
            color: white;
            border: none;
            padding: 10px 18px;
            border-radius: 8px;
            font-weight: 500;
        }
        .stButton>button:hover {
            background-color: #283593;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# --- Main container ---
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.markdown("<div class='title'>Customer Churn Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict the likelihood of a customer leaving the bank 📊</div>", unsafe_allow_html=True)

# --- Input Section ---
st.subheader("Customer Details")

col1, col2 = st.columns(2)
with col1:
    geography = st.selectbox('Geography', one_hot_encoder.categories_[0])
    age = st.slider('Age', 18, 100, 30)
    credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=600)
    tenure = st.slider('Tenure (Years)', 0, 10, 5)
    has_credit_card = st.selectbox('Has Credit Card', [0, 1])

with col2:
    gender = st.selectbox('Gender', labels_encoder_gender.classes_)
    balance = st.number_input('Balance', min_value=0.0, value=0.0)
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)
    num_of_products = st.slider('Number of Products', 1, 4, 1)
    is_active_member = st.selectbox('Is Active Member', [0, 1])

# --- Data preparation ---
input_data = {
    'CreditScore': credit_score,
    'Gender': labels_encoder_gender.transform([gender])[0],
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_credit_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}

geo_encoded = one_hot_encoder.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded.toarray(), columns=one_hot_encoder.get_feature_names_out(['Geography']))
input_df = pd.DataFrame([input_data])
final_input = pd.concat([input_df.reset_index(drop=True), geo_encoded_df.reset_index(drop=True)], axis=1)
final_input_scaled = scalar.transform(final_input)

# --- Predict Button ---
if st.button("Predict Churn"):
    prediction = model.predict(final_input_scaled)
    prediction = prediction[0][0]

    if prediction >= 0.5:
        st.markdown("<div class='prediction-box churn'>⚠️ The customer is likely to churn.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='prediction-box no-churn'>✅ The customer is not likely to churn.</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
