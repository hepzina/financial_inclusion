import streamlit as st
import joblib
import numpy as np

col1, col2, col3 = st.columns(3)

with col2:
    st.header("Financial inclusion model")

cols=["country","year","cellphone_access","gender_of_respondent","education_level","relationship_with_head","job_type"]

with open("model_rf.pkl", "rb") as file:
    my_model, encoders = joblib.load(file)

country=st.selectbox("Choose country",["Kenya","Rwanda","Tanzania","Uganda"])
encoded_coun = encoders["country"].transform([country])[0]

year = st.number_input("Enter a year", min_value=1900, max_value=2100, step=1)

cell=st.selectbox("Cellphone access",["Yes", "No"])
encoded_cell = encoders['cellphone_access'].transform([cell])[0]

gender=st.selectbox("What your gender",["Male","Female"])
encoded_gen = encoders["gender_of_respondent"].transform([gender])[0]

edu=st.selectbox("What your gender",['No formal education','Other/Dont know/RTA','Primary education','Secondary education','Tertiary education','Vocational/Specialised training'])
encoded_edu= encoders["education_level"].transform([edu])[0]

rela=st.selectbox("Relation_with_head",['Child','Head of Household','Other non-relatives','Other relative','Parent','Spouse'])
encoded_rela = encoders["relationship_with_head"].transform([rela])[0]

job=st.selectbox("Job type",['Dont Know/Refuse to answer','Farming and Fishing','Formally employed Government','Formally employed Private','Government Dependent','Informally employed','No Income','Other Income','Remittance Dependent','Self employed'])
encoded_job = encoders['job_type'].transform([job])[0]


# Combine all features into an array
input_data = np.array([[encoded_coun, year, encoded_cell, encoded_gen,encoded_edu,encoded_rela,encoded_job]])


# Predict and display result
if st.button("Predict"):
    prediction = my_model.predict(input_data)
    if prediction == 0:
        st.write("Does not have  a bank account")
    else:
        st.write("Has a bank account")
