import streamlit as st 
import config 
import numpy as np 
from xgboost import XGBClassifier
from utils import open_scaler,average_school_rating,get_home_price
from sklearn.preprocessing import OneHotEncoder

#Load the model 

model=XGBClassifier()
model.load_model(r"output\housing_xgboost_model.json")


home_category=np.load('utils_output\homeType_catergory.npy',allow_pickle=True).tolist()
school_district_catergory=np.load('utils_output\school_district_level_catergory.npy',allow_pickle=True).tolist()


encoder_home=OneHotEncoder(sparse_output=False,categories=home_category)
encoder_sd=OneHotEncoder(sparse_output=False,categories=school_district_catergory)

encoder_home.fit(np.array(home_category[0][0]).reshape(-1,1))
encoder_sd.fit(np.array(school_district_catergory[0][0]).reshape(-1,1))

st.title("Estimate Your Home Value")

yearBuild= st.number_input(min_value=1905,max_value=2023,label="Enter the year in which the house was build ")
latitude = st.number_input(min_value=30.085,max_value=30.517,label="Enter the latitude of the house ")
longtitude = st.number_input(min_value=-98.0204,max_value=-97.570,label="Enter the longitude of the house ")
garageSpace = st.number_input(min_value=0,label="Enter the number of garage spaces  ")
numOfPatioAndPorchFeatures = st.number_input(min_value=0,label="Enter the count of porch and patio feature ")
lotSizeSqFt = st.number_input(min_value=0,label="Enter the lot size in sq feet ")
averageSchoolRating= st.number_input(min_value=0,label="Enter the School Rating.")
MedianStudentsPerTeacher = st.number_input(min_value=0,label="Enter the median students per teacher number ")

numofBedrooms = st.number_input(min_value=0,label="Enter the number of bedrooms ")
numofBathrooms= st.number_input(min_value=0,label="Enter the number of bathrooms ")

hasspa =st.selectbox("Has Spa",(True,False))

home_type   = st.selectbox("HomeType", ('Single Family','Condo','Townhouse','Multiple Occupancy','Residential','Apartment', 
                            'Mobile / Manufactured', 'MultiFamily','Vacant Land','Other'))

submit=st.button(label="Estimate")

if submit:
    latitude_tr = open_scaler(r"utils_output\latitude_scaler.pkl",latitude)
    longitude_tr = open_scaler(r"utils_output\longitude_scaler.pkl",longtitude)
    garageSpace_tr = open_scaler(r"utils_output\garageSpaces_scaler.pkl",garageSpace)
    numOfPatioAndPorchFeatures_tr=open_scaler(r"utils_output\numOfPatioAndPorchFeatures_scaler.pkl",numOfPatioAndPorchFeatures)
    lotSizeSqFt_tr= open_scaler(r"utils_output\lotSizeSqFt_scaler.pkl",lotSizeSqFt)
    averageSchoolRating_tr=open_scaler(r"utils_output\avgSchoolRating_scaler.pkl",averageSchoolRating)
    MedianStudentsPerTeacher_tr=open_scaler(r"utils_output\MedianStudentsPerTeacher_scaler.pkl",MedianStudentsPerTeacher)
    numofBathrooms_tr=open_scaler(r"utils_output\numOfBathrooms_scaler.pkl",numofBathrooms)
    numofBedrooms_tr=open_scaler(r"utils_output\numOfBedrooms_scaler.pkl",numofBedrooms)
    homeage_tr = open_scaler(r"utils_output\home_age_scaler.pkl",2024-yearBuild)

    hasSpa_tr =[1 if hasspa==True else 0 ][0]

    home_type_ohe= encoder_home.transform([[home_type]]).tolist()[0]
    school_rating_ohe=encoder_sd.transform([[average_school_rating(averageSchoolRating)]]).tolist()[0]

    

    feature_list=[]

    feature_list.extend([latitude_tr,longitude_tr,garageSpace_tr,numOfPatioAndPorchFeatures_tr,lotSizeSqFt_tr,\
                         averageSchoolRating_tr,MedianStudentsPerTeacher_tr, numofBathrooms_tr,numofBedrooms_tr,hasSpa_tr,homeage_tr])
    
    feature_list=feature_list+ home_type_ohe + school_rating_ohe

 

    prediction=model.predict(np.array(feature_list).reshape(1,-1))

    home_price=get_home_price(prediction)

    st.text_area(label="$$ Estimated Home Price is",value = home_price)