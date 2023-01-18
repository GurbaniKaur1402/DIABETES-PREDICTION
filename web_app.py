import pandas as pd
import numpy as np
import pickle
import streamlit as st

from deploy import BloodPressure, DiabetesPedigreeFunction, Glucose, Insulin, Pregnancy, SkinThickness

loaded_model=pickle.load(open("savedmodel.sav",'rb'))

def diabetes_prediction(input_data):

    input_data_as_numpy_array=np.asarray(input_data)

    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        print('The person is not diabetic')
    else:
        print('The person is diabetic')


def main():
    st.title('Diabetes Prediction Web App')

    Pregnancy=st.text_input('Number of Pregnancies')
    Glucose=st.text_input(' Blood Glucose')
    BloodPressure=st.text_input('Blood Pressure')
    SkinThickness=st.text_input('Skin THickness value')
    Insulin=st.text_input('Value of Insulin')
    BMI=st.text_input('BMI value')
    DiabetesPedigreeFunction=st.text_input('value of diabetes pedigree function')
    Age=st.text_input('age of the person')

    diagnosis=''

    if st.button('Diabetes Test Result'):
        diagnosis=diabetes_prediction([Pregnancy,Glucose,BloodPressure,SkinThickness,Insulin,DiabetesPedigreeFunction,Age])


    st.success(diagnosis)



if __name__=='__main__':
    main()

    
