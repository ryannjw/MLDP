import streamlit as st
import joblib
import numpy as np
import pandas as pd 
model = joblib.load('model.pkl')
# APP Name 
st.title('Resale Price Prediction')

#Define the input fields
towns = ['Tampines', 'Bedok','Punggol', ]
flat_types = ['2 Room', '3 Room', '4 Room', '5 Room']
storeys_range = ['01 TO 03', '04 TO 06', '07 TO 09',]

town_selected = st.selectbox('Select Town', towns)
flat_type_selected = st.selectbox('Select Flat Type', flat_types)
storey_range_selected = st.selectbox('Select Storey Range', storeys_range)
floor_area_sqm_selected = st.slider('Select Floor Area (sqm)', min_value=30, max_value=200, value=70)
       



#prediction buttons
if st.button("predict HDP Price"):
    #creat dict for input features
    # input_data = pd.DataFrame({
    #     'town': [town_selected],
    #     'flat_type': [flat_type_selected],
    #     'storey_range': [storey_range_selected],
    #     'floor_area_sqm': [floor_area_sqm_selected]
    # })
    # #Convert input data to DataFrame

    df_input = pd.DataFrame({
        'town': [town_selected],
        'flat_type': [flat_type_selected],
        'storey_range': [storey_range_selected],
        'floor_area_sqm': [floor_area_sqm_selected]
    })

    df_input = pd.get_dummies(df_input, columns=['town', 'flat_type', 'storey_range'])
    df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)  

    y_unseen = model.predict(df_input)[0]
    st.success(f'Predicted Resale Price: ${y_unseen:,.2f}')

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://www.shutterstock.com/image-photo/great-white-shark-close-shot-1899493687");
        background-size: cover;
    }}
    </style>""",
    unsafe_allow_html=True)