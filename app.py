from flask import Flask,render_template
import requests
import pickle
import pandas as pd
import numpy as np
import sklearn
import lightgbm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import joblib
from zipfile import ZipFile





print('The scikit-learn version is {}.'.format(sklearn.__version__))
print(lightgbm.__version__)

app = Flask(__name__)

url = 'https://github.com/Ashishsarathe1989/Santander-Customer-Satisfaction/blob/main/LGBM_model.zip?raw=true'
with open("LGBM_model.zip",'wb') as output_file: output_file.write(requests.get(url).content)
url = 'https://github.com/Ashishsarathe1989/Santander-Customer-Satisfaction/blob/main/Remove_const_Fetaure.pkl?raw=true'
with open("Remove_const_Fetaure.pkl",'wb') as output_file: output_file.write(requests.get(url).content)
url = 'https://github.com/Ashishsarathe1989/Santander-Customer-Satisfaction/blob/main/Remove_Qassi_Fetaure.pkl?raw=true'
with open("Remove_Qassi_Fetaure.pkl",'wb') as output_file: output_file.write(requests.get(url).content)
url = 'https://github.com/Ashishsarathe1989/Santander-Customer-Satisfaction/blob/main/correlated_features.pkl?raw=true'
with open("correlated_features.pkl",'wb') as output_file: output_file.write(requests.get(url).content)
url = 'https://github.com/Ashishsarathe1989/Santander-Customer-Satisfaction/blob/main/Scaler.pkl?raw=true'
with open("Scaler.pkl",'wb') as output_file: output_file.write(requests.get(url).content)
url = 'https://github.com/Ashishsarathe1989/Santander-Customer-Satisfaction/blob/main/onehot.pkl?raw=true'
with open("onehot.pkl",'wb') as output_file: output_file.write(requests.get(url).content)
url = 'https://github.com/Ashishsarathe1989/Santander-Customer-Satisfaction/blob/main/cat_col.pkl?raw=true'
with open("cat_col.pkl",'wb') as output_file: output_file.write(requests.get(url).content)

                                                          
#loading model from downloaded pickle file


with ZipFile("LGBM_model.zip", 'r') as zip: LGBM_zip= zip.extractall()
with open("LGBM_model.pkl", 'rb') as file:  LGBM = pickle.load(file)
with open("Remove_const_Fetaure.pkl", 'rb') as file:  ConstanttVarinace = pickle.load(file)
with open("Remove_Qassi_Fetaure.pkl", 'rb') as file:  Quassiconst = pickle.load(file)
with open("correlated_features.pkl", 'rb') as file:  correlated_features = pickle.load(file)
with open("Scaler.pkl", 'rb') as file:  Scaler = pickle.load(file)
with open("onehot.pkl", 'rb') as file:  onehot = pickle.load(file)
with open("cat_col.pkl", 'rb') as file:  cat_col = pickle.load(file)

#LGBM = joblib.load('LGBM_model/LGBM_model.pkl')
#ConstanttVarinace = joblib.load('Remove_const_Fetaure.pkl')
#Quassiconst = joblib.load('Remove_Qassi_Fetaure.pkl')
#correlated_features = joblib.load('correlated_features.pkl')
#Scaler = joblib.load('Scaler.pkl')
#onehot = joblib.load('onehot.pkl')
#cat_col = joblib.load('cat_col.pkl')



     

@app.route("/")
def Index():
     return render_template('Index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    
    
    file = request.files['file']
    fileName = file.filename
    
    file.save(secure_filename(file.filename))
    
  	 
    df=pd.read_csv(fileName)
      
    x_test=df.drop(["ID"],axis = 1)
    
    #remove_cols= ConstanttVarinace+ Quassiconst+ correlated_features
    #Drop Contant Feature
    constant_ = ConstanttVarinace.get_support()
    
    constant_feature=np.where(constant_==False)
  
    x_test.drop(x_test.columns[constant_feature],axis=1,inplace=True)
    
    #Drop quassi Contant Feature
    
    #Drop Contant Feature
    constant_ = Quassiconst.get_support()
    
    constant_feature=np.where(constant_==False)
  
    x_test.drop(x_test.columns[constant_feature],axis=1,inplace=True)
    
    #Coreleayted Feature
    
    x_test.drop(correlated_features, axis=1, inplace=True)
    
    #onehotencoding
    
    temp_test = onehot.transform(x_test[cat_col].values).toarray()
    
    ohe_df_test = pd.DataFrame(temp_test,columns=onehot.get_feature_names())# Get feature name test

    x_test = pd.concat([x_test, ohe_df_test], axis=1).drop(x_test[cat_col], axis=1)#Remove categorical columns (will replace with one-hot encoding) Test data
    
    #adding new features
    
    feature=x_test.columns.values
    x_test["sum"]=x_test[feature].sum(axis=1)
    x_test["mean"]=x_test[feature].mean(axis=1)
    x_test["min"]=x_test[feature].min(axis=1)
    x_test["max"]=x_test[feature].max(axis=1)
    x_test["std"]=x_test[feature].std(axis=1)
    x_test["med"]=x_test[feature].median(axis=1)
   

    age_below_23= []
    for i in (x_test['var15']):
      if i < 23:
        age_below_23.append(1)
      else:
        age_below_23.append(0)

    x_test['age_below_23']= age_below_23
    
    x_test.fillna(x_test.mean())
   
    x_test_std = Scaler.transform(x_test)
    
    
    
    y_test_pred_lgbm = LGBM.predict_proba(x_test_std)[:,1]
    
    #make prediction
    prediction=""
    for i in y_test_pred_lgbm:
      if i > 0.5:
         prediction="Unsatisfied"
      else:
         prediction="Satisfied"
        
    print(prediction)        
       
    return render_template('Index.html', prediction_text='Customer {} with bank service.'.format(prediction)) 

    

if __name__=='__main__':
    app.run(debug=True)