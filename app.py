import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.write("""
# Cross-sell Prediction App
This app predicts if the existing customers will buy vehicle insurance!
Data obtained from the [kaggle](https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction).

""")

st.sidebar.header('User Input Features')


def user_input_features():

    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    age = st.sidebar.slider('Age (years)', 20, 85, 52)
    dl = st.sidebar.selectbox('Have driving license?', ('Yes', 'No'))
    region_code = st.sidebar.slider('Region code', 0, 52, 25)
    previously_insured = st.sidebar.selectbox(
        'Previously insured?', ('Yes', 'No'))
    vehicle_age = st.sidebar.selectbox(
        'Vehicle Age', ('< 1 Year', '1-2 Year', '> 2 Years'))
    vehicle_damage = st.sidebar.selectbox('Vehicle Damage', ('Yes', 'No'))
    annual_premium = st.sidebar.slider('Annual Premium', 2630, 540165, 271397)
    policy_sales_channel = st.sidebar.slider(
        'Policy Sales Channel', 1, 163, 80)
    vintage = st.sidebar.slider('Vintage', 10, 299, 150)

    def change_text_to_num(x):
        if x == "Yes":
            return 1
        return 0

    data = {'Gender': gender,
            'Age': age,
            'Driving_License': change_text_to_num(dl),
            'Region_Code': region_code,
            'Previously_Insured': change_text_to_num(previously_insured),
            'Vehicle_Age': vehicle_age,
            'Vehicle_Damage': vehicle_damage,
            'Annual_Premium': annual_premium,
            'Policy_Sales_Channel': policy_sales_channel,
            'Vintage': vintage}
    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
raw_data = pd.read_csv('dummy.csv')
insurance = raw_data.drop(columns=['id'])
df = pd.concat([input_df, insurance], axis=0)

# Encoding of ordinal features


def preprocessor(x):
    encode = ["Gender", 'Vehicle_Age', 'Vehicle_Damage', 'Driving_License']

    for col in encode:
        dummy = pd.get_dummies(x[col], prefix=col)
        x = pd.concat([x, dummy], axis=1)
        del x[col]

    return x


numer = ['Age', 'Annual_Premium',
         'Policy_Sales_Channel', 'Vintage', 'Region_Code']


# Reads in saved Scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))

processed_data = preprocessor(df)  # Handling categorical variables
processed_data.iloc[:1, :5] = scaler.transform(processed_data[:1][numer])
processed_input_data = processed_data[:1]

# Displays the user input features
st.subheader('User Input features')

st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('xgboost_model.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(np.array(processed_input_data))
prediction_proba = load_clf.predict_proba(np.array(processed_input_data))

pred = ['No', 'Yes']

st.subheader('Will the customer buy?')
st.write(pred[int(prediction)])

st.subheader('Prediction Probability')
st.write(prediction_proba)
