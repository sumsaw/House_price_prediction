
from xgboost import XGBClassifier
data_file_path=r"input\train.csv"
save_model=r"output\housing_xgboost_model.json"


one_hot_columns=['homeType','school_district_level']

normalize_columns=['latitude','longitude','garageSpaces','numOfPatioAndPorchFeatures',\
                   'lotSizeSqFt','avgSchoolRating','MedianStudentsPerTeacher','home_age',\
                    'numOfBedrooms','numOfBathrooms']


xgb_model=XGBClassifier(booster= 'dart',reg_lambda=3.707134105471464e-06,

    alpha= 0.002438214547890573,
    subsample= 0.9707991870452112,
    colsample_bytree= 0.8191172587547764,
    max_depth= 9,
    min_child_weight= 10,
    eta= 0.03267796954607347,
    gamma= 2.5322158336494526e-08,
    grow_policy= 'depthwise',
    sample_type= 'weighted',
    normalize_type= 'forest',
    rate_drop= 6.6227807052151025e-06,
    skip_drop= 0.001013246858551301,
    objective='multi:softmax',num_class=5)