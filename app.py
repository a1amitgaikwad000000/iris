import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("iris_model.pkl", "rb"))

st.title("🌸 Iris Flower Prediction App")

st.write("Enter the flower measurements to predict the species")

# Input fields
sepal_length = st.number_input("Sepal Length", min_value=0.0)
sepal_width = st.number_input("Sepal Width", min_value=0.0)
petal_length = st.number_input("Petal Length", min_value=0.0)
petal_width = st.number_input("Petal Width", min_value=0.0)

# Prediction button
if st.button("Predict"):
    
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    prediction = model.predict(input_data)

    species = ["Setosa", "Versicolor", "Virginica"]

    st.success(f"Predicted Flower: {species[prediction[0]]}")
