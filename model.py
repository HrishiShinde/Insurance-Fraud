import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

print("Training started!")
label_encoder = LabelEncoder()

# Returns all the categorical columns from dataframe passed.
def get_categorical_columns(df):
    categorical_columns = []
    threshold = 8000
    object_columns = df.select_dtypes(include=['object']).columns

    unique_values = df[object_columns].nunique()
    for k, v in unique_values.items():
        if v <= threshold:
            categorical_columns.append(k)

    return categorical_columns

# Preprocessing
def preprocess_data(df):
    # Dropping rows with NA values. 
    df = df.dropna()

    # Removing "Cust" from CustomerId and "Location" from IncidentAddress.
    df['CustomerID'] = df['CustomerID'].str.replace('Cust', '')
    df['IncidentAddress'] = df['IncidentAddress'].str.replace('Location ', '').astype('int64').astype('int32')

    # Converting dates to numeric values.
    df['DateOfIncident'] = pd.to_datetime(df['DateOfIncident'], format='%d-%m-%Y')
    df['DateOfIncident'] = df['DateOfIncident'].astype('int64').astype('int32')
    df['DateOfPolicyCoverage'] = df['DateOfPolicyCoverage'].astype('int64').astype('int32')

    # Splitting train and test data. 
    x = df.drop(['ReportedFraud', 'VehicleAttribute', 'VehicleAttributeDetails'], axis=1)
    y = df['ReportedFraud']

    # Perform one-hot encoding on the categorical columns
    categorical_columns = get_categorical_columns(x)
    x = pd.get_dummies(x, columns=categorical_columns)

    y = label_encoder.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test


# Importing data.
train_data = pd.read_csv("Data/Train_Claim_Master.csv", na_values=["?", "MISSEDDATA", "NA", "-1", "MISSINGVAL", "-5", "MISSINGVALUE"])

# Preprocessing and splitting data between test and train.
x_train, x_test, y_train, y_test = preprocess_data(train_data)

# Using Random Forest Classifier.
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
predrfc = rfc.predict(x_test)

pickle.dump(rfc, open('model.pkl','wb'))

print("Model Trained!")

 