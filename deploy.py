import pandas as pd
import numpy as np
import pickle


load_model=pickle.load(open("savedmodel.sav",'rb'))
input_data=[]
Pregnancy = int(input("Enter Pregnancy-"))
input_data.append(Pregnancy)
Glucose=int (input("Enter Glucose-"))
input_data.append(Glucose)
BloodPressure=int(input("Enter BloodPressure-"))
input_data.append(BloodPressure)
SkinThickness=int(input("Enter SkinThickness-"))
input_data.append(SkinThickness)
Insulin=int(input("Insulin-"))
input_data.append(Insulin)
BMI=int(input("Enter BMI-"))
input_data.append(BMI)
DiabetesPedigreeFunction=int(input("Enter DiabetesPedigreeFunction-"))
input_data.append(DiabetesPedigreeFunction)
Age=int(input("Enter Age-"))
input_data.append(Age)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
print(input_data_as_numpy_array)
print (input_data_reshaped)
prediction = load_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')

  



