import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

st.title("Insurance Fraud Detection System", anchor=None)

model = pickle.load(open('model.pkl', 'rb'))
model_single = pickle.load(open('model_single.pkl', 'rb'))

# Returns all the categorical columns from dataframe passed.
def get_categorical_columns(df):
    categorical_columns = []
    threshold = 1000
    object_columns = df.select_dtypes(include=['object']).columns

    unique_values = df[object_columns].nunique()
    for k, v in unique_values.items():
        if v <= threshold:
            categorical_columns.append(k)

    return categorical_columns

# Preprocessing
def preprocess_data(df):
    # Dropping rows with NA values. 
    st.write(f"{len(df)} Rows")
    if len(df) > 1:
        df = df.dropna()

    # Removing "Cust" from CustomerId and "Location" from IncidentAddress.
    df['CustomerID'] = df['CustomerID'].str.replace('Cust', '')
    df['IncidentAddress'] = df['IncidentAddress'].str.replace('Location ', '').astype('int64').astype('int32')

    # Converting dates to numeric values.
    df['DateOfIncident'] = pd.to_datetime(df['DateOfIncident'], format='%d-%m-%Y')
    df['DateOfIncident'] = df['DateOfIncident'].astype('int64').astype('int32')
    df['DateOfPolicyCoverage'] = pd.to_datetime(df['DateOfPolicyCoverage'], format='%d-%m-%Y')
    df['DateOfPolicyCoverage'] = df['DateOfPolicyCoverage'].astype('int64').astype('int32')

    # Splitting train and test data. 
    df = df.drop(['VehicleAttribute', 'VehicleAttributeDetails'], axis=1)

    # One-hot encoding on the categorical columns
    categorical_columns = get_categorical_columns(df)
    df = pd.get_dummies(df, columns=categorical_columns)

    return df

def predict(InsuredAge, InsuredZipCode, CapitalGains, CapitalLoss, IncidentTime, NumberOfVehicles, BodilyInjuries, AmountOfInjuryClaim, AmountOfPropertyClaim, AmountOfVehicleDamage, InsurancePolicyNumber, CustomerLoyaltyPeriod, Policy_Deductible, PolicyAnnualPremium, UmbrellaLimit):
    prediction = model_single.predict([[InsuredAge, InsuredZipCode, CapitalGains, CapitalLoss, IncidentTime, NumberOfVehicles, BodilyInjuries, AmountOfInjuryClaim, AmountOfPropertyClaim, AmountOfVehicleDamage, InsurancePolicyNumber, CustomerLoyaltyPeriod, Policy_Deductible, PolicyAnnualPremium, UmbrellaLimit]])
    if prediction == 0:
        return 'No Fraud'
    else:
        return 'Fraud'

def main1():
    uploaded_file = st.file_uploader("Choose your CSV file")
    
    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file, na_values=["?", "MISSEDDATA", "NA", "-1", "MISSINGVAL", "-5", "MISSINGVALUE"])
        st.dataframe(dataframe)
            
        test_data = preprocess_data(dataframe)

        predictions = model.predict(test_data)
        
        if type(predictions) == np.ndarray:
            customer_ids = ['Cust' + str(customer_id) for customer_id in test_data.loc[predictions == 1, 'CustomerID'].tolist()]
            
            predict_df = pd.DataFrame(columns=["CustomerID", "Fraud"])
            predict_df["CustomerID"] = customer_ids
            predict_df["Fraud"] = True

            st.success(f"There are {len(customer_ids)} Fraud Insurance claims found in your data.")

            dataframe = dataframe.merge(predict_df[['CustomerID', 'Fraud']], on='CustomerID', how='left')
            dataframe['Fraud'] = dataframe['Fraud'].fillna(False)
            st.dataframe(dataframe)

        elif predictions == 0:
            st.success("It's not a Fraud.")
        elif predictions == 1:
            st.success("It's a Fraud.")

def main2():
    InsuredAge = st.text_input('Insured Age')
    InsuredZipCode = st.text_input('Insured Zip Code')
    CapitalGains = st.text_input('Capital Gains')
    CapitalLoss = st.text_input('Capital Loss')
    IncidentTime = st.text_input('Incident Time')
    NumberOfVehicles = st.text_input('Number Of vechiles')
    BodilyInjuries = st.text_input('Bodily Injuries')
    AmountOfInjuryClaim = st.text_input('Amount of Injury Claim')
    AmountOfPropertyClaim = st.text_input('Amount Of Property Claim')
    AmountOfVehicleDamage = st.text_input('Amount Of Vehicle Damage')
    InsurancePolicyNumber = st.text_input('Insurance Policy Number')
    CustomerLoyaltyPeriod = st.text_input('Customer Loyalty Period')
    Policy_Deductible = st.text_input('Policy Deductible')
    PolicyAnnualPremium = st.text_input('Policy Annual Premium')
    UmbrellaLimit = st.text_input('Umbrella Limit')
    result = ""
    if st.button("Predict"):
        result = predict(InsuredAge, InsuredZipCode, CapitalGains,CapitalLoss,IncidentTime,NumberOfVehicles,BodilyInjuries,AmountOfInjuryClaim,AmountOfPropertyClaim,AmountOfVehicleDamage,InsurancePolicyNumber,CustomerLoyaltyPeriod,Policy_Deductible,PolicyAnnualPremium,UmbrellaLimit)
        st.success('This is {}.'.format(result))

genre = st.radio(
    "Select the type of data input",
    (
        'File (Use this when there are multiple records)', 
        'Values (Use this when there is single records)')
    )

if genre == 'File (Use this when there are multiple records)':
    main1()
else:
    main2()