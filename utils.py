
import config
import pickle
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

def get_home_price(prediction):
    if prediction==0:
        return '0-250000'
    elif prediction==1:
        return '250000-350000'
    elif prediction==2:
        return '350000-450000'
    elif prediction==3:
        return '450000-650000'
    elif prediction==4:
        return '650000+'
    

def  average_school_rating(value):
    if value ==6:
        return 'par'
    elif value > 6: 
        return 'above_par'
    elif value<6:
        return 'below_par'
    
def open_scaler(scaler_file_path,column):
    with open(scaler_file_path,'rb') as f:
        sc=pickle.load(f)
        column=sc.transform([[column]])
    
    return column.tolist()[0][0]


    
def train_model(file_path):
    """
    This function takes the preprocessed dataset and then returns the trained model and its corresponding artifacts like 
    standard scalar and One-hot-encoders
    
    """
    hash_map={}
    hash_map['target']={'0-250000':0, '250000-350000':1, '350000-450000':2,'450000-650000':3,'650000+':4}

    df=pd.read_csv(file_path)
    df=df.sample(frac=1).reset_index(drop=True)

    x_train=df.drop('target',axis=1)
    y_train=df['target']


    for column in config.normalize_columns:
        sc=StandardScaler()
        sc.fit(x_train[column].values.reshape(-1,1))
        x_train[column]=sc.transform(x_train[column].values.reshape(-1,1))

        with open(f'utils_output/{column}_scaler.pkl','wb') as f:
            pickle.dump(sc,f)

    print("Intialize the XGboost Model")
    model=config.xgb_model

    print(x_train.columns)
    print(len(x_train.columns))

    print("Training........")
    model.fit(x_train,y_train)

    print("Prediction......")
    predictions=model.predict(x_train)

    print(f"Macro_F1_score: {f1_score(y_train,predictions,average='macro')}")

    feature_imp=pd.DataFrame(model.get_booster().get_fscore(),index=['feature_imp_score']).T.sort_values(by='feature_imp_score',ascending=False)

    print(feature_imp)

    #Save the model 

    model.save_model(config.save_model)


def train_n_fold(file_path):
    """
    Function which takes preprocessed data and does n-fold validation on it 
    """
    df=pd.read_csv('input\data_preprocessd.csv')

    df['kfold']=-1
    df=df.sample(frac=1).reset_index(drop=True)

    kf=StratifiedKFold(n_splits=config.n_fold)

    for f,(t_,v_) in enumerate(kf.split(X=df.drop('target',axis=1),y=df['target'])):
        df.loc[v_,'kfold']=f


    for fold in range(config.n_fold):
        df_train=df[df.kfold!=fold].reset_index(drop=True)
        df_valid=df[df.kfold==fold].reset_index(drop=True)

        x_train=df_train.drop(["kfold","target"],axis=1)
        y_train=df_train["target"]

        x_valid=df_valid.drop(["kfold","target"],axis=1)
        y_valid=df_valid['target']

        for column in config.normalize_columns:
            sc=StandardScaler()
            sc.fit(x_train[[column]])
            x_train[column]=sc.transform(x_train[[column]])
            x_valid[column]=sc.transform(x_valid[[column]])

        model=config.xgb_model

        print(f"Training for fold : {fold+1}")

        model.fit(x_train,y_train)

        y_vpred= model.predict(x_valid)
        y_tpred= model.predict(x_train)

        t_f1_score=f1_score(y_train,y_tpred,average='macro')
        v_f1_score=f1_score(y_valid,y_vpred,average='macro')

        print(f"****** fold: {fold+1}, train_f1_score: {t_f1_score}, valid_f1_score : {v_f1_score} ******")


def pre_process_data(file_path):

    """ THis function takes the raw dataset and outputs the pre-processed dataset"""

    #read the csv file 

    hash_map={}
    hash_map['target']={'0-250000':0, '250000-350000':1, '350000-450000':2,'450000-650000':3,'650000+':4}

    df=pd.read_csv(file_path)
    df=df.sample(frac=1).reset_index(drop=True)

    #Feature engineering 
    df['target']=df['priceRange'].map(hash_map['target'])
    df['hasspa_encoded']=df['hasSpa'].map({True:1, False:0})
    df['home_age']=2024-df['yearBuilt']
    df['school_district_level']=df['avgSchoolRating'].apply(average_school_rating)

    #One hot Encoding 

    for column in config.one_hot_columns:
        onehot_encoder=OneHotEncoder(sparse_output=False,categories='auto')
        onehot_encoder.fit(df[[column]])
        feature=onehot_encoder.transform(df[[column]])
        categories=onehot_encoder.categories_
        np.save(f'utils_output/{column}_catergory.npy',categories)
        df=pd.concat([df,pd.DataFrame(feature,columns=onehot_encoder.get_feature_names_out())],axis=1)
    
    #drop the non-needed columns 
    df.drop(labels=['uid','city','yearBuilt','homeType','school_district_level','description','priceRange','hasSpa'],axis=1,inplace=True)

    df.to_csv(r'input\data_preprocessd.csv',index=False)

    



