import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv("C://Users//Yashwanth//Downloads//insurance.csv")

# Encode categorical columns with get_dummies
df_encoded = pd.get_dummies(df, drop_first=True)

# Define features and target
X = df_encoded.drop(columns='charges', axis=1)
Y = df_encoded['charges']

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train the Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predict on the test data
Y_pred = regressor.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(Y_test, Y_pred)
# st.write(f"Mean Squared Error on Test Data: {mse}")

# Streamlit form to accept user input for predictions
st.title("Insurance Cost Prediction")

st.write("Enter the following details to predict the insurance cost:")

# Input fields for user
age = st.number_input("Age", min_value=18, max_value=100, value=21)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=27.99)
children = st.number_input("Number of children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

# Encode the inputs (one-hot encode as done with the training set)
sex_encoded = 0 if sex == "male" else 1
smoker_encoded = 0 if smoker == "yes" else 1
region_encoded = {"southeast": 0, "southwest": 1, "northeast": 2, "northwest": 3}[region]

# Prepare the input data for prediction with the same columns as the training data
input_data = pd.DataFrame({
    'age': [age],
    'bmi': [bmi],
    'children': [children],
    'sex_female': [sex_encoded],  # Encoding for 'sex'
    'smoker_yes': [smoker_encoded],  # Encoding for 'smoker'
    'region_southwest': [1 if region_encoded == 1 else 0],  # Encoding for 'region'
    'region_northeast': [1 if region_encoded == 2 else 0],
    'region_northwest': [1 if region_encoded == 3 else 0],
})

# Ensure the input_data has the same columns as the training data
input_data = input_data.reindex(columns=X.columns, fill_value=0)

# Add a button for prediction
if st.button('Predict Insurance Cost'):
    # Make the prediction
    prediction = regressor.predict(input_data)
    
    # Display the result
    st.write(f"The predicted insurance cost is: USD {prediction[0]:.2f}")
