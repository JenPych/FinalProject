import streamlit as st
import pickle

ask = st.text_input("Which model would you like to use? LR for Linear Regression or SVR for Support Vector Regression")


if ask == "LR":
    with open('modelLR.pickle', 'rb') as file:
        model = pickle.load(file)

elif ask == "SVR":
    with open('modelSVR.pickle', 'rb') as file:
        model = pickle.load(file)
else:
    print("error")

st.write("Ads in TV and Radio")
tv = st.number_input("Enter the cost of ads on TV: ")
radio = st.number_input("Enter the cost of ads on radio: ")
both = tv * radio

if st.button("Predict"):
    y_pred = model.predict([[tv, radio, both]])
    st.write(f'Predicted Sales = {y_pred}')
