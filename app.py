import streamlit as st
import pandas as pd
import pickle

# ======================
# Load model
# ======================

loaded = pickle.load(
    open("models/churn_model_final.pkl","rb")
)

model = loaded['model']
scaler = loaded['scaler']
threshold = loaded['threshold']
feature_columns = loaded['feature_columns']

# ======================
# Page setup
# ======================

st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📉"
)

st.title("📉 Telecom Customer Churn Prediction")

# ======================
# Inputs
# ======================

gender = st.selectbox(
    "Gender",
    ["Male","Female"]
)

seniorcitizen = st.selectbox(
    "Senior Citizen",
    [0,1]
)

partner = st.selectbox(
    "Partner",
    ["Yes","No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes","No"]
)

tenure = st.slider(
    "Tenure",
    0,72,12
)

phoneservice = st.selectbox(
    "Phone Service",
    ["Yes","No"]
)

multiplelines = st.selectbox(
    "Multiple Lines",
    ["Yes","No","No phone service"]
)

internetservice = st.selectbox(
    "Internet Service",
    ["DSL","Fiber optic","No"]
)

onlinesecurity = st.selectbox(
    "Online Security",
    ["Yes","No","No internet service"]
)

onlinebackup = st.selectbox(
    "Online Backup",
    ["Yes","No","No internet service"]
)

deviceprotection = st.selectbox(
    "Device Protection",
    ["Yes","No","No internet service"]
)

techsupport = st.selectbox(
    "Tech Support",
    ["Yes","No","No internet service"]
)

streamingtv = st.selectbox(
    "Streaming TV",
    ["Yes","No","No internet service"]
)

streamingmovies = st.selectbox(
    "Streaming Movies",
    ["Yes","No","No internet service"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month",
     "One year",
     "Two year"]
)

paperlessbilling = st.selectbox(
    "Paperless Billing",
    ["Yes","No"]
)

paymentmethod = st.selectbox(
    "Payment Method",
[
"Electronic check",
"Mailed check",
"Bank transfer (automatic)",
"Credit card (automatic)"
]
)

monthlycharges = st.slider(
    "Monthly Charges",
    0,
    150,
    70
)

totalcharges = st.number_input(
    "Total Charges",
    value=1000
)

service_count = sum([
onlinesecurity=="Yes",
onlinebackup=="Yes",
deviceprotection=="Yes",
techsupport=="Yes",
streamingtv=="Yes",
streamingmovies=="Yes"
])

# ======================
# Dataframe
# ======================

input_data = pd.DataFrame({

'gender':[gender],
'SeniorCitizen':[seniorcitizen],
'Partner':[partner],
'Dependents':[dependents],
'tenure':[tenure],
'PhoneService':[phoneservice],
'MultipleLines':[multiplelines],
'InternetService':[internetservice],
'OnlineSecurity':[onlinesecurity],
'OnlineBackup':[onlinebackup],
'DeviceProtection':[deviceprotection],
'TechSupport':[techsupport],
'StreamingTV':[streamingtv],
'StreamingMovies':[streamingmovies],
'Contract':[contract],
'PaperlessBilling':[paperlessbilling],
'PaymentMethod':[paymentmethod],
'MonthlyCharges':[monthlycharges],
'TotalCharges':[totalcharges],
'service_count':[service_count]

})

if st.button("Predict"):

    input_encoded = pd.get_dummies(input_data)

    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    input_scaled = scaler.transform(
        input_encoded
    )

    probability = model.predict_proba(
        input_scaled
    )[0][1]

    prediction = (
        1 if probability >= threshold
        else 0
    )

    if prediction==1:

        st.error(
            f"⚠️ Customer likely to churn\n\nRisk: {probability*100:.2f}%"
        )

    else:

        st.success(
            f"✅ Customer likely to stay\n\nConfidence: {(1-probability)*100:.2f}%"
        )