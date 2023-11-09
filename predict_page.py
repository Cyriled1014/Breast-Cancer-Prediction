import streamlit as st
import joblib
import pandas as pd

model= joblib.load('best_model.joblib')
pipeline = joblib.load('pipeline.joblib')

st.title('Breast Cancer Classification Prediction')

feature_names = ["concave points_worst" , "concave points_mean", "perimeter_worst", "radius_worst", "area_worst", "texture_worst", "concavity_worst","perimeter_se", "texture_mean", "radius_mean"]


# Create a dictionary to store user inputs

# Define feature names and their types (numeric or categorical)
feature_names = {
"concave points_worst": "numeric", 
"concave points_mean" : "numeric", 
"perimeter_worst" : "numeric",
"radius_worst" : "numeric", 
"area_worst": "numeric", 
"texture_worst": "numeric", 
"concavity_worst" : "numeric",
"perimeter_se": "numeric", 
"texture_mean": "numeric", 
"radius_mean" : "numeric"
}

# Create a dictionary for user inputs
user_inputs = {}

# Loop through the features names and create appropriate input fields

for feature, feature_type in feature_names.items():

    user_input = st.number_input(f"{feature}", min_value = 0.000001, max_value = 10000.0)


    user_inputs[feature] = user_input


# Convert the user_inputs dictionary into a list of lists (2D array-like format)
user_input_data = pd.DataFrame([user_inputs],  columns=feature_names)


# Transform user input data using the preprocessor
transformed_data = pipeline.transform(user_input_data)


# Make predictions using the model
prediction = model.predict(transformed_data)


if prediction == 1:
    st.write("The Breast Cancer type is Malignant")
elif prediction == 0:
    st.write("The Brest Cancer type is Benign")




