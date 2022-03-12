#!/usr/bin/env python
# coding: utf-8

# ##Introduction -:
# Santander Customer Satisfaction is a competition posted by Santander Bank on Kaggle.As for any 
# business customer satisfaction is important to grow the buisness.Similarly, Santander was asking 
# Kagglers to find happy and unhappy customer using machine learning algorithm.

# ##Business problem: -
# 1.we need to predict whether a customer is satisfied with their services based on the features provided 
# by the company or not . This will help them to take appropriate action to improve the customer 
# satisfaction to stay longer with bank before the customer leave .
# 
# 2.To build a machine learning algorithm using the training data set and predict the total satisfied and 
# unsatisfied customers in the Santander test data set.

# In[1]:


#Importing  important libraries 
import warnings
import pickle
warnings.filterwarnings("ignore")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve, auc
from collections import Counter


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


#Redaing Train data
train_data=pd.read_csv('/content/drive/MyDrive/santander-customer-satisfaction/train.csv')
#Redaing Test  Data 
test_data = pd.read_csv('/content/drive/MyDrive/santander-customer-satisfaction/test.csv')


# In[ ]:


train_data.head(5)


# *   There are 76020 data points 370 Independet feature    and    1 dependent Target variable.
# 
# *   0 represent Happy Customer 
# *   1 represent Unhappy Customer
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# In[ ]:


test_data.head(5)


# In[ ]:


print(test_data.shape)


# There are 75818  data points 370 feature in test data

# In[ ]:


colum_=np.array(train_data.columns.to_list())
feature=[col for col in train_data.columns if col not in ["ID","TARGET"]]
print(feature)


# #Exploratory data analysis

# ###Check the Train-test Distribution
# Before we doing our work, we might be extremely interested in the distribution of the dataset. The division of train set and test set should be as balanced as possible in all kinds of aspects.
# if not then have to make it balance or can say normally distributed  using  Log Transformation,Box-Cox Transformation.

# In[ ]:


plt.figure(figsize=(18,10))
sns.kdeplot(train_data[feature].mean(axis=0), fill=True,color='green',label="Train")
sns.kdeplot(test_data[feature].mean(axis=0), fill=True,color='red',label="TEST")
plt.title('Finding Distibution of mean value per column in  between train and test data')
plt.legend()
plt.show()


# In[ ]:


plt.figure(figsize=(18,10))
sns.kdeplot(train_data[feature].mean(axis=1), fill=True,color='green',label="Train")
sns.kdeplot(test_data[feature].mean(axis=1), fill=True,color='red',label="TEST")
plt.title('Finding Distibution of mean value per row in  between train and test data')
plt.legend()
plt.show()


# ####We can see data distitbution of both(tain  and test) almost balanced ,so during testing our resault will be not bisaed.
# 

# ###TARGET (Dependent varibale)

# In[ ]:


#Target Variable Analysis 
train_data["TARGET"].value_counts()


# In[ ]:


Traget_0=(100.* train_data["TARGET"].value_counts()[0] / 76020).round(1)
Traget_1=(100.* train_data["TARGET"].value_counts()[1] / 76020).round(1)


# In[ ]:


print("Happy Customer = " + str(Traget_0)+"%")
print("Unhappy Customer = "+str(Traget_1)+"%")


# In[ ]:


sns.countplot(x="TARGET",hue="TARGET",data=train_data);
 
# Show the plot
plt.show()


# ####We Can see Dataset is higly Imbalance .
# ####So before feeding into model we have to make it balance othewise result will be bias .
# ####we will fix this during doing Feature engeering

# Lets Check data distribution according to customer type

# In[ ]:


#Checking Null value in train dataset 
train_data.columns[train_data.isnull().any()]


# ###There are no null or missing value in dataset .

# Select Top 20 feature using SelectKBest for further data analysis.
# 
# Refrence: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html

# In[ ]:


x=train_data.drop(["ID","TARGET"],axis = 1)
y=train_data["TARGET"]


# In[ ]:


from sklearn.feature_selection import SelectKBest, f_classif
# Create and fit selector
selector = SelectKBest(f_classif, k=20) #k is number of top feature
selector.fit(x, y)
mask = selector.get_feature_names_out(input_features=None)
print("Top 15 Feature",mask)


# ###Constant Feature

# In[ ]:


train_data.iloc[:, [21, 22 , 56 , 57 , 58 , 59 , 80 , 84, 85 ,131, 132, 133 ,134, 155 ,161 ,162 ,179, 180, 189 ,192, 220, 222, 234 ,238 ,244 ,248 ,261 ,262 ,303 ,307 ,315, 319 ,327, 349]].head(5)


# In[ ]:


Top_Feature=['var15','ind_var5', 'ind_var8_0', 'ind_var12_0', 'ind_var13_0',
 'ind_var13_corto', 'ind_var13', 'ind_var30', 'ind_var39_0', 'num_var4',
 'num_var5' ,'num_var8_0' ,'num_var13_0' ,'num_var13', 'num_var30', 'num_var35',
 'num_var42', 'saldo_var30' ,'var36', 'num_meses_var5_ult3']


# In[ ]:


Top_20_Feature_With_target=train_data[Top_Feature+['TARGET']]
print(list(Top_20_Feature_With_target.columns)) #we get 21 feature including target variable


# In[ ]:


Top_10_Feature_With_target=['var15', 'ind_var5', 'ind_var30' ,'num_var4', 'num_var5' ,'num_var30',
 'num_var35', 'num_var42' ,'var36', 'num_meses_var5_ult3','TARGET']


# * Using SelectKBest i got 20 important feature ['var15','ind_var5', 'ind_var8_0', 'ind_var12_0', 'ind_var13_0',
#  'ind_var13_corto', 'ind_var13', 'ind_var30', 'ind_var39_0', 'num_var4',
#  'num_var5' ,'num_var8_0' ,'num_var13_0' ,'num_var13', 'num_var30', 'num_var35'
#  'num_var42', 'saldo_var30' ,'var36', 'num_meses_var5_ult3']
# 
# * Also got to know Features [ 21,  22 , 56 , 57 , 58 , 59 , 80 , 84,  85 ,131, 132, 133 ,134, 155 ,161 ,162 ,179, 180,
#  189 ,192, 220, 222, 234 ,238 ,244 ,248 ,261 ,262 ,303 ,307 ,315, 319 ,327, 349] are constant  (These all are feature  colum's index )
# 
# *   we can remove 34 constant feature during feature preprocessing  as they  are not contibuting much for classfication.

# ####Find quasi-constant features
# Let's first find out the number of quasi-constant features.

# In[ ]:


vt = VarianceThreshold(threshold=0.02)
v_threshold = vt.fit(x)
#The get_support returns a Boolean vector where True means that the variable does not have zero variance so we will remove False vaibale
Quasi_feature = v_threshold.get_support()
k=np.where(Quasi_feature==False)
print("quasi-constant features={}".format(len(k[0])))


# *   
# Quasi-constant features are those that show the same value for the great majority of the observations.
# *   We assume that features with a higher variance may contain more useful information
# *  Using a feature with zero-variance or little variane  only adds to model complexity, not to its predictive power ,So we will remove feature  with lower variacnce.
# 
# *   There are 106 quasi_Fconstant featute while doing feature engrring we wiil remove this.
# 

# 
# ##Univaraite analysis
# ###we will select 1 feature at time to understand data's insight 

# In[ ]:


unique_colm=[]
for k in colum_:
  res=k.split("_")[0] #indexing zero becaue we want first  word after spliting e.g imp_op_var39_comer_ult3 geiing only imp
  unique_colm.append(res)
res=(unique_colm)
print(res)


# In[ ]:


plt.figure(figsize=(10,5))
sns.histplot(res)
plt.show()


# ####we ploted prefix column value and found rendring  num prefix column is  high
# ####We will do analysis  on all the keyword appear on x-axis in above graph
# ####Here prefix num rendring is high so let start analyis with num
# 

# In[ ]:


Num_colm=[]
for i in colum_:
  res=i.split("_")[0] 
  if res =='num':
    Num_colm.append(i)
print(Num_colm)
print("Number of "'num'" prefix column is = {}".format(len(Num_colm)))


# ###  Take 10 random num feature 

# In[ ]:


def plot_Distibution(happy_dt,unhappy_dt,features):

  plt.figure(figsize=(20,15))
  i=0
  for feature in features:
    i+=1
    plt.subplot(2,5,i)
    sns.kdeplot(happy_dt[feature],color='green')
    sns.kdeplot(unhappy_dt[feature],color='red')
    plt.xlabel(feature,fontsize=9)
  plt.show()
happy_dt=train_data.loc[train_data["TARGET"]==0]  
unhappy_dt=train_data.loc[train_data["TARGET"]==1]  
feature=Num_colm [0:10]
plot_Distibution(happy_dt,unhappy_dt,feature)


# ####As we can see there are some plot which have happy and unhappy customer both ,also  sligtly gaussion distribution ,so we can consider these feature might be usefull ,also have some plot which has 0 variance so we can remove such feature.

# ###Prefix num_op (Number of transction)

# In[ ]:


substring="num_op"
strings_with_substring = [string for string in Num_colm if substring in string]
features=strings_with_substring[0:10]
plot_Distibution(happy_dt,unhappy_dt,features)    


# ‘num_op’ likey is shortened word for importe which is Spanish find num_op is number of transactions done  by  customer having seen above 4 graph have zero varinace , ('num_op_var40_hace2',
#  'num_op_var40_hace3',
#  'num_op_var40_ult1',
#  'num_op_var40_ult3') so we can remove these feature ,
#  and except this we are getting some insight about happy and unhappy custome data maigt be this feature would be usefull.

# In[ ]:


sns.set_style("whitegrid")
sns.FacetGrid(train_data,hue="TARGET",height=8).map(sns.histplot,"num_op_var40_hace3").add_legend()
plt.title('Finding Insight between number of operations and Target')
plt.show()


# ##num_var4 Bank Product

# In[ ]:


sns.set_style("whitegrid")
sns.FacetGrid(train_data,hue="TARGET",height=8).map(sns.histplot,"num_var4").add_legend()
plt.title('Finding Insight betwwen Number of product and Target')
plt.show()


# According this comment  https://www.kaggle.com/cast42/santander-customer-satisfaction/exploring-features/comments#115223)
# ### num_var4 is the number of products
# Total number of bank product is 7 ,by using above plot we can see with bank product few customer are unhappy .
# using  1 product's customer volume  is very high(happy customer) >using  2 product's customer volume  is high(happy customer)>using 3 product's customer volume  is less (happy customer)
# 

# ##num_var35 analysis

# In[ ]:


sns.set_style("whitegrid")
sns.FacetGrid(train_data,hue="TARGET",height=8).map(sns.histplot,"num_var35").add_legend()
plt.title('Finding Insight betwwen Number of product and Target')
plt.show()


# ######By seeing this can not say anything from which product its belong to but we observerd this we are getting happy and unhappy customer data distributiion  hence this feature may usefull

# ###Prefix "imp" analysis

# In[ ]:


imp_colm=[]
for i in colum_:
  res=i.split("_")[0] 
  if res =='imp':
    imp_colm.append(i)
print(imp_colm)
print("Number of "'imp'" prefix column is = {}".format(len(imp_colm)))


# In[ ]:


features_imp=imp_colm[0:10]# taking 10  prefix imp feature to check data distibution
plot_Distibution(happy_dt,unhappy_dt,features_imp)


# "imp" is short from of import (inferred from literature review) ,total number of "imp" prefix column is 49 .by above plot we can see most of data distribution happening  at zero .

# ###Prefix 'Saldo’ feature

# In[ ]:


saldo_colm=[]
for i in colum_:
  res=i.split("_")[0] 
  if res =='saldo':
    saldo_colm.append(i)
print(saldo_colm)
print("Number of "'saldo'" prefix column is = {}".format(len(saldo_colm)))


# In[ ]:


features_saldo=saldo_colm[0:10]# taking 10  prefix saldo feature to check data distibution
plot_Distibution(happy_dt,unhappy_dt,features_saldo)


# Total number of "Saldo" prefix column is 71 .by above plot we can see most of data distribution happening  at zero.
# we will consider same as it data for model trainin.

# ##Var3(Natonlaity of customer)

# In[ ]:


train_data["var3"].value_counts()


# In[ ]:


train_data.loc[train_data.var3==-999999].shape


# In[ ]:


train_data.loc[train_data.var3==-999999].head(5)


# In[ ]:


print(100*(74165/76020)) # % of Appering 2 in dataset belong to Var3
print(100*(116/76020) ) # % of Appering -999999 in dataset belong to Var3


# ####By refering some artical said that this feature contain natonlaity of customer and -999999 look like outlier, replace -999999 in var3 column with  value 2 and we aslo did some analysis in va3 ,most comman value 2 its appering 97% on train dataset so we can replace with 2
# 
# ### See https://www.kaggle.com/cast42/santander-customer-satisfaction/debugging-var3-999999
# 

# Var15 (Customer Age)

# In[ ]:


train_data["var15"].describe()


# We can see age  range of customer  5 to 105

# In[ ]:


sns.set_style("whitegrid")
sns.FacetGrid(train_data,hue="TARGET",height=6,aspect=2.5).map(plt.hist,"var15").add_legend()
plt.title('Finding Insight between Number of Age and Target')
plt.show()


# In[ ]:


Unappy_Customer=train_data.loc[train_data['TARGET']==1, 'var15']
Happy_Customer=train_data.loc[train_data['TARGET']==0, 'var15']


# In[ ]:


plt.figure(figsize=(30,10))
sns.kdeplot(Happy_Customer,color='green')
sns.kdeplot(Unappy_Customer,color='red')
x_ = np.arange(5, 106, 1)

plt.title('Finding Insight between Customer  Age and Target')
plt.xlabel("Customer age range")
plt.xticks(x_)
plt.show()


# 
# 
# *   Green plot represent :-Happy customer 
# *   Red plot represent:-Unhappy customer
# 
# 
# 
# *   By above histogram and pdf (Unhappy cutsomer age range) graph we can see most of  unhappy customer  are older (more than  40  age group)
# *  By above histogram and pdf (Happy cutsomer age range) graph we can see most of  happy customer  are younger (less than  30  age group)
# 
# *   By above histogram  we can see below age of 23 are totally  happy customer  we did not find any unhappy customer till age to 23
# *   So  we can make new feature during feature engnerring age<23 is totally happy  customer or not
# 
# 
# 
# 

# ###Bivariate Analysis.
# ####Bivariate analysis is the analysis of any concurrent relationship of  two variavle .
# ####we will plot pairplot to see distribution  among  all the important feature or attribute.

# In[ ]:


#plot pait plot to test bi-virates relationship
plt.figure(figsize=(8,5))
g = sns.PairGrid(train_data[Top_10_Feature_With_target], hue="TARGET", aspect=0.8, diag_sharey=False)
g.map_lower(sns.scatterplot)
g.map_diag(sns.histplot)
plt.suptitle('Finding data distribution among Important feature and Target')
plt.show()


# By seeing  above grap we can say  var15 have repaltionship with other feature where we found  happy and unhappy customer both.
# also we found num_var25 and num_meses_var5_ult3 have some relationship with respect to target variabl.
# 
# Although we do not get more infromation about data and also seems data is  not 
# separable,however we get some  data  pattern such kind of pattern consider as categorcal in nature with prefix "num" let dig out more to make sure about data nature.
# 
# 

# In[ ]:


train_data[Num_colm].nunique()


# we can see max "num" count is 172 ,so we can say its kind of categorical in nature.

# ##Check features with correlation

# In[ ]:


fig, ax1 = plt.subplots(figsize=(30,30))
sns.heatmap(train_data.corr(),vmin= -1, vmax=1.0,  fmt='.2f',
                 linewidths=.5,  cbar_kws={"shrink": .90},ax=ax1)
plt.show()


# There are many corelation feature seems in above plot ,but due to density of graph  we cant not read this graph properly or seems hard to read ,so we will check with top feature .
# Features with high correlation are more linearly dependent and hence have almost the same effect on the dependent variable. So, when two features have high correlation, we can drop one of the  features.

# ###Lets check feature corelationship with 20 important feature as we got from using SelectKBest 

# In[ ]:


fig, ax1 = plt.subplots(figsize=(25,30))
sns.heatmap(Top_20_Feature_With_target.corr(),vmin= -1, vmax=1.0, annot=True, fmt='.2f',
                 linewidths=.5,  cbar_kws={"shrink": .90},ax=ax1)
plt.show()


# Now we can see high corealted featute ,we will remove higly corelated feature during feature  engineering.

# ###PCA visualization

# In[ ]:


# initializing the pca
from sklearn import decomposition
pca = decomposition.PCA()


# In[ ]:


from sklearn.preprocessing import StandardScaler
standardized_data = StandardScaler().fit_transform(x)
print(standardized_data.shape)


# In[ ]:


# configuring the parameteres
# the number of components = 2
pca.n_components = 2
data=standardized_data
label=y
pca_data = pca.fit_transform(data)
# even tried this  code but still getting same resault  (pca_data = pca.fit_transform(data))

# pca_reduced will contain the 3-d projects of simple data
print("shape of pca_reduced.shape = ", pca_data.shape)


# In[ ]:


# attaching the label for each 2-d data point 
pca_data = np.vstack((pca_data.T, label)).T

# creating a new data fram which help us in ploting the result data
pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal", "2nd_principal","label"))
sns.FacetGrid(pca_df,hue="label", height=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
plt.show()


# Now we can visualize most of the  data being overlap its  means we can not find that the which  feature are  independent.so its does not seem usefull.
# 

# Create New Feature
# 
# 1.   Based on var15 make a new feature whether the age of happy customer is below 23 or not.
# 2.   We can also  add the statistical indicators(mean,median,std,min,max)
# 3. Using  PCA to reduce dimension and without PCA .
# 4. Apply  one hot encoding to Num_colm and create new datasets.
# 

# ##Feature Preprocessing

# ###Removing constant features
# This dataset contains constant features. I know this from previous analysis so I will quickly remove these features to reduce the data size.
# 
# Such kind  of feature do not give any contibution during classification process.

# In[ ]:


x_train=train_data.drop(["ID","TARGET"],axis = 1)
y_train=train_data["TARGET"]
x_test=test_data.drop(["ID"],axis = 1)
print(x_train.shape)
print(x_test.shape)


# In[ ]:


def Remove_const_Fetaure(x_train,x_test):
  vt = VarianceThreshold(threshold=0)
  v_threshold = vt.fit(x_train) 
  #The get_support returns a Boolean vector where True means that the variable does not have zero variance so we will remove False vaibale
  constant_ = v_threshold.get_support()
  constant_feature=np.where(constant_==False)
  x_train.drop(x_train.columns[constant_feature],axis=1,inplace=True)
  x_test.drop(x_test.columns[constant_feature],axis=1,inplace=True)
  print("Total constant feature {}".format(len(constant_feature[0])))
  print("Remaining columns after removing constant feature on train dataset {}".format(x_train.shape[1]))
  print("Remaining columns after removing constant feature on test dataset {}".format(x_test.shape[1]))


# In[ ]:


Remove_const_Fetaure(x_train,x_test)


# ###Removing Quassi constant features

# Using a feature with  little variane(Quassi) only adds to model complexity, not to its predictive power ,So we will remove feature with lower variacnce.

# In[ ]:


def Remove_quassiconst_Fetaure(x_train,x_test):
  vt = VarianceThreshold(threshold=0.02)
  v_threshold = vt.fit(x_train) 
  #The get_support returns a Boolean vector where True means that the variable does not have zero variance so we will remove False vaibale
  constant_ = v_threshold.get_support()
  constant_feature=np.where(constant_==False)
  x_train.drop(x_train.columns[constant_feature],axis=1,inplace=True)
  x_test.drop(x_test.columns[constant_feature],axis=1,inplace=True)
  print("Total quassi constant feature {}".format(len(constant_feature[0])))
  print("Remaining columns after removing quassi constant feature on train dataset {}".format(x_train.shape[1]))
  print("Remaining columns after removing quassi constant feature on test dataset {}".format(x_test.shape[1]))


# In[ ]:


Remove_quassiconst_Fetaure(x_train,x_test)


# ###Removing Duplicate Features using Transpose
# By refering some blog come to know this dataset also have duplicate feature(columns) .we have no built-in Python method that can remove duplicate features. However, we have a method that can help us identify duplicate rows in a pandas dataframe. We will use this method to first take a transpose of our dataset as shown below:
# 

# In[ ]:


train_features_T = x_train.T
print("tarin data shape after tarnspose",train_features_T.shape)
test_features_T = x_test.T
print("test data shape after tarnspose",test_features_T.shape)


# In the script above we take the transpose of our training data and store it in the train_features_T dataframe same with test data. Our initial training set contains 76020 rows and 370 columns, if you take a look at the shape of the transposed training set, you will see that it contains 370 rows and 76020 columns.
# 
# Luckily, in pandas we have duplicated() method which can help us find duplicate rows from the dataframe. Remember, the rows of the transposed dataframe are actually the columns or the features of the actual dataframe.
# 
# Let's find the total number of duplicate features in our dataset using the sum() method, chained with the duplicated() method as shown below.

# In[ ]:


print("Total duplicate feature in training data = {}".format(train_features_T.duplicated().sum()))
print("Total duplicate feature in test data = {}".format(test_features_T.duplicated().sum()))


# Finally, we can drop the duplicate rows using the drop_duplicates() method. If you pass the string value first to the keep parameter of the drop_duplicates() method, all the duplicate rows will be dropped except the first copy. In the next step we will remove all the duplicate rows and will take transpose of the transposed training set to get the original training set that doesn't contain any duplicate column. Execute the following script

# In[ ]:


x_train_ = train_features_T.drop_duplicates(keep='first').T
x_test_ = test_features_T.drop_duplicates(keep='first').T
print("new training set without duplicate features:",x_train_.shape)
print("new testing set without duplicate features:",x_test_.shape)


# ###Removing correlated  feature
# If the two variables move in the same direction, then those variables are said to have a positive correlation. If they move in opposite directions, then they have a negative correlation.Features with high correlation are more linearly dependent and hence have almost the same effect on the dependent variable. So we can drop one of the  features.
# From the heatmap of features in EDA its clear that feature are highly corellated so we are keeping threshold 0.80 That means if two features have correlation greater than 0.80, then second feature will be dropped and first one will be kept.

# In[ ]:


#Correlation 
cor_feature = abs(x_train_.corr())
#Selecting highly correlated features
correlated_features = set()
for i in range(len(cor_feature.columns)):
    for j in range(i):
        if abs(cor_feature.iloc[i, j]) > 0.8:
            colname = cor_feature.columns[i]
            correlated_features.add(colname)


# In[ ]:


print("Total we have {} correlated  feature".format(len(correlated_features)))


# In[ ]:


x_train.drop(correlated_features, axis=1, inplace=True)
x_test.drop(correlated_features, axis=1, inplace=True)


# In[ ]:


(x_train.shape) ,(x_test.shape)


# After removing coreallaion feature left with 166 feature  in train data and 167 in test data.

# ##Feature Engineering

# In[ ]:


#While doing EDA we find  natonlaity of customer  -999999 look like outlier
# Replace -999999 in var3 column with most common value 2 
# See https://www.kaggle.com/cast42/santander-customer-satisfaction/debugging-var3-999999
# for details
x_train = x_train.replace(-999999,2)
x_train.loc[x_train.var3==-999999].shape


# Create new dataset with statistical indicators(sum,mean,median,std,min,max)

# ###One Hot Encoding
# The maximum number of unique values found for a feature with ‘num’ keyword is 172(total datapoints is around 70k) and minimum was 2.

# In[ ]:


Num_colm_one=[]
for i in x_train.columns:
  res=i.split("_")[0] 
  if res =='num':
    Num_colm_one.append(i)
print(Num_colm_one)
print("Number of "'num'" prefix column is = {}".format(len(Num_colm_one)))


# In[ ]:


#getting all columns with less than or equal to 10 nunique values
cat_col = []
for col in Num_colm_one:
  if (x_train[col].nunique()<=10)  & (x_train[col].nunique()>2):
    cat_col.append(col)
print("There are %i columns which have less than or equal to 10 and greater than 2 number of unique values. \nWe will create new datasets which use one hot encoding"%(len(cat_col)))


# In[ ]:


#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(x_train[cat_col].values) 

temp_train = enc.transform(x_train[cat_col].values).toarray()# Transform train data to  onehot encoding  

temp_test = enc.transform(x_test[cat_col].values).toarray()# Transform test data to  onehot encoding  
ohe_df = pd.DataFrame(temp_train,columns=enc.get_feature_names())# Get feature name  train
ohe_df_test = pd.DataFrame(temp_test,columns=enc.get_feature_names())# Get feature name test

x_train = pd.concat([x_train, ohe_df], axis=1).drop(x_train[cat_col], axis=1)#Remove categorical columns (will replace with one-hot encoding) Train data

x_test = pd.concat([x_test, ohe_df_test], axis=1).drop(x_test[cat_col], axis=1)#Remove categorical columns (will replace with one-hot encoding) Test data


# In[ ]:


x_train.shape, x_test.shape


# In[ ]:


feature=x_train.columns.values
for df in [x_train,x_test]:
  df["sum"]=df[feature].sum(axis=1)
  df["mean"]=df[feature].mean(axis=1)
  df["min"]=df[feature].min(axis=1)
  df["max"]=df[feature].max(axis=1)
  df["std"]=df[feature].std(axis=1)
  df["med"]=df[feature].median(axis=1)
x_train[x_train.columns[216:]].head(10)


# In[ ]:


x_test[x_test.columns[216:]].head(10)


# ###Creating new datasets (Age below 23 or not )

# In[ ]:


def age_23(tarin_,test_):
  
  for data in (tarin_,test_):

      age_below_23=[]
      for age in (data["var15"]):
          if age < 23 :
            age_below_23.append(1)
          else:
            age_below_23.append(0)
        
      data["age_below_23"]=age_below_23

      print("New feature added below 23 age and shape",data.shape)
      print("-------------------------------")


# In[ ]:


age_23(x_train,x_test)


# we are considring age below 23 is happy customer represented by 1 and unhappy customer above 23 represent by 0.

# ####Standard Scaling

# In[ ]:


data_train = x_train.fillna(x_test.mean())
data_test = x_test.fillna(x_test.mean())


# In[ ]:


(data_train.shape) ,(data_test.shape)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std=sc.fit_transform(data_train)
X_test_std = sc.transform(data_test)


# ###Using PCA for  Dimensionality Reduction

# In[ ]:


# fit and transform data
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)


# ##Modeling

# ###Using Cross validation to find best param with PCA

# In[ ]:


def find_best_params_pca(model,params,cv=10,n_jobs=-1):
  """
  Funcion which implements random Seacrh CV and returns best model
  """
  random_cv = RandomizedSearchCV(model,param_distributions=params,scoring='roc_auc',n_jobs=n_jobs,cv=cv,verbose=2)
  random_cv.fit(X_train_pca,y_train)
  print("The best auc score was %.3f"%(random_cv.best_score_))
  print("The best params were: %s"%(random_cv.best_params_))
  return random_cv.best_estimator_


# ###Using Cross validation to find best param without PCA

# In[ ]:


def find_best_params(model,params,cv=10,n_jobs=-1):
  """
  Funcion which implements random Seacrh CV and returns best model
  """
  random_cv = RandomizedSearchCV(model,param_distributions=params,scoring='roc_auc',n_jobs=n_jobs,cv=cv,verbose=2)
  random_cv.fit(X_train_std,y_train)
  print("The best auc score was %.3f"%(random_cv.best_score_))
  print("The best params were: %s"%(random_cv.best_params_))
  return random_cv.best_estimator_


# ###Logistic regression(Base model)
# ####As Data is not linerly seprable so we can assume that Logistic regression will not work well .still try with Logistic regression.

# In[ ]:


from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression(class_weight='balanced')
params = {'penalty':['l2','l1','elasticnet'], 'C':[1.0,1.5,2.0,2.5,3.0], 
          'fit_intercept':[True,False] }


# In[ ]:


find_best_params_pca(model_lr,params)# Best param PCA model


# In[ ]:


find_best_params(model_lr,params)# Best param without PCA model


# Logistic regression
# 

# In[ ]:


clf_lg_pca = LogisticRegression(C=3.0, class_weight='balanced',
                   fit_intercept=True, penalty='l2', verbose=0)

clf_lg_pca.fit(X_train_pca,y_train)
y_train_pred_pca = clf_lg_pca.predict_proba(X_train_pca)

clf_lg = LogisticRegression(C=1.0, class_weight='balanced',
                   fit_intercept=True, penalty='l2', verbose=0)

clf_lg.fit(X_train_std,y_train)
y_train_pred = clf_lg.predict_proba(X_train_std)

train_fpr_pca, train_tpr_pca, tr_thresholds_pca = roc_curve(y_train, y_train_pred_pca[:,1])

train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred[:,1])
plt.plot(train_fpr_pca, train_tpr_pca, label="train pca AUC ="+str(auc(train_fpr_pca, train_tpr_pca)))
plt.plot(train_fpr, train_tpr, label="train logistic regression AUC ="+str(auc(train_fpr, train_tpr)))

plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ERROR PLOTS")
plt.grid(True)
plt.show()


# Saving models to pickle file

# In[ ]:


import pickle
model_name = "LogisticRegression.pkl"
with open(model_name, 'wb') as file: pickle.dump(clf_lg, file)


# In[ ]:


#probabilistic perdiction on test data
y_test_pred_lg = clf_lg.predict_proba(X_test_std)[:,1] #test data


# In[ ]:


#perdiction on test
sample_submission=pd.read_csv("/content/drive/MyDrive/santander-customer-satisfaction/sample_submission.csv")
sample_submission["TARGET"]=y_test_pred_lg
sample_submission.to_csv("Logisticregression.csv",index=False)


# ![image.png](data:image/png;base64,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)

# ###Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model_random_forest = RandomForestClassifier()

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 400, num = 10)]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]


random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf
              }


# In[ ]:


find_best_params_pca(model_random_forest,random_grid)# Best param with PCA model


# In[ ]:


find_best_params(model_random_forest,random_grid) # Best param without PCA model


# In[ ]:



clf_RM_pca=RandomForestClassifier(n_estimators= 133, min_samples_split= 10, min_samples_leaf=2, max_features= 'auto', max_depth= 10, 
                              random_state=45,class_weight='balanced' )

clf_RM_pca.fit(X_train_pca,y_train)
y_train_pred_pca = clf_RM_pca.predict_proba(X_train_pca)

clf_RM=RandomForestClassifier(n_estimators= 283, min_samples_split= 2, min_samples_leaf=4, max_features= 'auto', max_depth= 80, 
                               random_state=45 ,class_weight='balanced')

clf_RM.fit(X_train_std,y_train)
y_train_pred = clf_RM.predict_proba(X_train_std)

train_fpr_pca, train_tpr_pca, tr_thresholds_pca = roc_curve(y_train, y_train_pred_pca[:,1])
train_fpr_rm, train_tpr_rm, tr_thresholds_rm = roc_curve(y_train, y_train_pred[:,1])

plt.plot(train_fpr_pca, train_tpr_pca, label="train pca AUC ="+str(auc(train_fpr_pca, train_tpr_pca)))
plt.plot(train_fpr_rm, train_tpr_rm, label="train AUC ="+str(auc(train_fpr_rm, train_tpr_rm)))

plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ERROR PLOTS")
plt.grid(True)
plt.show()


# Saving models to pickle file

# In[ ]:


import pickle
model_name = "RandomForest.pkl"
with open(model_name, 'wb') as file: pickle.dump(clf_RM, file)


# In[ ]:


#probabilistic perdiction on test data
y_test_pred_Rm = clf_RM.predict_proba(X_test_std)[:,1] #test data


# In[ ]:


#perdiction on test
sample_submission=pd.read_csv("/content/drive/MyDrive/santander-customer-satisfaction/sample_submission.csv")
sample_submission["TARGET"]=y_test_pred_Rm
sample_submission.to_csv("RandomForest.csv",index=False)


# ![image.png](data:image/png;base64,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)

# ###XGBoost

# In[ ]:


counter = Counter(train_data["TARGET"])
# estimate scale_pos_weight value
#scale_pos_weight = total_negative_examples / total_positive_examples
estimate = np.sqrt((counter[1]) / (counter[0]))
print('Estimate: %.3f' % estimate)#value to consider: sum(negative instances) / sum(positive instances)


# In[ ]:


from xgboost import XGBClassifier
model_xg = XGBClassifier(random_state=42)
# Number of trees in random forest
n_estimators_ = [int(x) for x in np.linspace(start = 10, stop = 100, num = 10)]
# Maximum number of levels in tree
max_depth_ = [int(x) for x in np.linspace(10, 110, num = 11)]
# Minimum number of samples required to split a node
min_samples_split_ = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf_ = [1, 2, 4]

learning_rate=[0.01,0.1,0.02,0.03,0.2,0.4,0.5,0.6]

num_leave=[int(x) for x in np.linspace(10, 110, num = 11)]
min_child_weight=[x for x in np.linspace(0.01, 5, num = 10)]

random_grid_= {'n_estimators': n_estimators_,
               'max_depth': max_depth_,
               'min_samples_split': min_samples_split_,
               'min_samples_leaf': min_samples_leaf_,
               'learning_rate':learning_rate,
               'num_leaves':num_leave,
               'min_child_weight':min_child_weight
              }


# In[ ]:


find_best_params_pca(model_xg,random_grid_)# Best param with PCA model
find_best_params(model_xg,random_grid_)# Best param without PCA mode


# In[ ]:


clf_Xg_PCA =XGBClassifier(scale_pos_weight=0.20,learning_rate=0.02, max_depth=30, min_child_weight=0.01,
              min_samples_leaf=4, min_samples_split=2, n_estimators=30,reg_lambda=0.1,
              num_leaves=20, random_state=100)

clf_Xg_PCA.fit(X_train_pca,y_train)


y_train_pred_pca = clf_Xg_PCA.predict_proba(X_train_pca)

train_fpr_pca, train_tpr_pca, tr_thresholds_pca = roc_curve(y_train, y_train_pred_pca[:,1])
plt.plot(train_fpr_pca, train_tpr_pca, label="train pca AUC ="+str(auc(train_fpr_pca, train_tpr_pca)))

clf_Xg=XGBClassifier(scale_pos_weight=0.20,learning_rate=0.02, max_depth=45, min_child_weight=0.01,
              min_samples_leaf=4, min_samples_split=2, n_estimators=80,reg_lambda=0.1,
              num_leaves=20, random_state=100)

clf_Xg.fit(X_train_std,y_train)
y_train_pred = clf_Xg.predict_proba(X_train_std)

train_fpr_xg, train_tpr_xg, tr_thresholds_xg = roc_curve(y_train, y_train_pred[:,1])
plt.plot(train_fpr_xg, train_tpr_xg, label="train xgboost AUC ="+str(auc(train_fpr_xg, train_tpr_xg)))


plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ERROR PLOTS")
plt.grid(True)
plt.show()


# Saving models to pickle file

# In[ ]:


import pickle
model_name = "Xgboostfile.pkl"
with open(model_name, 'wb') as file: pickle.dump(clf_Xg, file)


# In[ ]:


#probabilistic perdiction on test data
y_test_pred_xg = clf_Xg.predict_proba(X_test_std)[:,1] #test data


# In[ ]:


#perdiction on test
sample_submission=pd.read_csv("/content/drive/MyDrive/santander-customer-satisfaction/sample_submission.csv")
sample_submission["TARGET"]=y_test_pred_xg
sample_submission.to_csv("Xgboostresult.csv",index=False)


# ![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA7EAAACKCAYAAAB8ds2qAAAgAElEQVR4nOzde1xUdf748deGMaU0JoZfaSE0sFqwDCgLflqwrqaGzqoLViDlhbylqWQpZuhWSOVtTS3FSyKUQmqoecEMTBfMBLYUtgRKhIIvs+KXkdDDSuf3xwwwM8wA3kp238/Hw4fMnM/53M458HnP53PO/E5VVRUhhBBCCCGEEKIduOm3roAQQgghhBBCCNFWEsQKIYQQQgghhGg3JIgVQgghhBBCCNFuSBArhBBCCCGEEKLdkCBWCCGEEEIIIUS7IUGsEEIIIYQQQoh2Q4JYIYQQQgghhBDthgSxQgghhBBCCCHajQ5Xm8Evv/zCzz9foPbiRRSljkuXLlFf/8u1qJsQQgghhBBCiP9yDg430aFDBzQaRzrecgu/U1VVvZKM6ur+zblqAwZDDU6dOtLx1lvQ3OLIzR1uxsFBJniFEEIIIYQQQly9+vpf+Pelf6NcrKP2wsUrC2L/dfYc1YYautyupbP2NglahRBCCCGEEEL8Ki4riK2r+zcVlf9C4+jIHV27SPAqhBBCCCGEEOJX1eZ7Yi9cVCivqKRrl9vp3Pm261knIYQQQgghhBDCpjYFsXV1/6a8ohKXO5y5zanT9a6TEEIIIYQQQghhU5vWA1dU/ouuXW6XAFYIIYQQQgghxG+q1SD2X2fPoXF0lCXEQgghhBBCCCF+cy0GsXV1/6baUMMdXbv8WvURQgghhBBCCCHsajGIPVdtoMvtWnkKsRBCCCGEEEKIG4Ld6PSXX37BYKihs1aWEQshhBBCCCGEuDHYDWJ//vkCTp06yiysEEIIIYQQQogbht0ItfbiRTreesuvWRchhBBCCCGEEKJFdoNYRalDc4vjr1kXIYQQQgghhBCiRXaD2EuXLnFzh5t/zboIIYQQQgghhBAtshvE1tf/IvfDCiGEEEIIIYS4oUiUKoQQQgghhBCi3ZAgVgghhBBCCCFEuyFBrBBCCCGEEEKIdkOCWCGEEEIIIYQQ7YYEsUIIIYQQQggh2g0JYoUQQgghhBBCtBsSxAohhBBCCCGEaDckiBVCCCGEEEII0W5IECuEEEIIIYQQot2QIFYIIYQQQgghRLshQawQQgghhBBCiHZDglghhBBCCCGEEO2GBLFCCCGEEEIIIdqNDr9l4YrBgIIGrVbT9GatAcMl6/fKyM8tRXH1wqeHCxoHqzw0WhqT1ysYDAoardaYTjFgqG1K3/i+7Qq1mFY5Z0AxT+9gVc96Bf3pAgrLNbj7eePe0X7eFvkrBgyKZV4W7TK1qZmOZu22dq6Y3PxKcPXBz1NrWY/LKUujRWujHc360fp9G+WglJH/VSnVnd3x93ZrdhwMxXmcLIduPr54dTHP2uoYQ8t9Xa9gMIC2i9V5hVVbhBBCCCGEEO3ObzsTe2o1IYFz2XfO9Lo2k9i+gSzJbUigUJQ8lT69BzM2Pp5XwgLpo1tGbmMwqOeTGf7M2KVvylO/kxn+0Xxieku/K5o+/QfzzHNjeCYsmPv66Fhy3EZA2GraPJb4+9M/bIxx+3NjeCb+EI0l6zOJ0/nTN2wucfETGNjbn8k7ymznbfqX+I3Ztgd1rCyw065vkkz7hNHfrA5xB83abab04yju6z+G196J55Uwf/q8sKuxnvpd0fSZsbOp3tZl5S2jj38woxrK6+3PZLP+1e+Kpo+/PzP2GMxKVMiKD6SPdb+blaMcj2fgH3TMeCeeuImD6aNbTW5D19aXsTWqN33HLCDunbmM8jcv09YxbrmvjW2wrGPuCn/6rMiz2V9CCCGEEEKI9uM3nYnVPDSFmAGBxK3JI3iOL0Wb4tkasIjDQaYZtIL1TJ6v54V9J4m6B2OwM2kwz7zzKF/HBmBvErKZvjPZuCEUFyD/3WBCNh8i6qFBaK8g7ehFacQ8ZL2TgX1vvEDivYs4ljYMFwdQsuMZGL6UfQOWMljbPO/milny4mqC903Bx3qm2Hc8u9PGA3nE3R0GNuvQII/NL2cSmVxITABQm03c0xvYWzyMSM+Wu6lJKG+lzcEPKN0SyWM7MtEPs6z3vg92UjosAneAc+ls3mT7gwEjPZ+sXk+3uAySn3KD+mK2vjCLfdkR+AVpIS+JOQcjSC6cQ6ADKEfiGbVhP0XDIvBqlpepr31X8nVaEFoHY4AcEjaLRL8UIj0a0imkvbyAwQFLGdylWSZCCCGEEEKIduo3vidWy+C5sXitXcTWzFSWvKMlZs6wxmCp6Kv9FA2bwOh7TG84uDF6XARK6lHyr7BETSct1Cq0FHK1lFapNmA4Z/rXsEHJJXuXCy+MNwawAJqAORzIWUBwJ/OdFSob9j1nVYOgmUR7r2bGirw21c0+F9x8IXHZMtJOlKFoAohJS7iMABZAofqcAcO5YrIO5uPzqI9l4B00DJ1+PWknjC9L92wmf+ggAu3mp8XdQ0PW+tUkHilGf8mT0e+lERNkiu5d3PAjiSUrdpFfoqDpN4fdG2wFsDT19dPGABZA81AEEwLyyMw1n5keT/TUAma8vAvb89VCCCGEEEKI9ui3f7BT91CiZxuIHRdD6bS5jDYLtgzlBeDqYjljejNQe5lh3rFljNXpCAkJJOQNhaiJg+zMhraeNjHKnz7+xn+NS1zP6SnFCxdny6w0XazuG81eSIh/w/7LyLVI7cLoRSsJXD+XJccNXDk3Ijdn8N6fSkmZpeO+Xv6Ery24zMB4J3HPjeGZ52axJNeT4PuteyuAsIkurNyejVJfQNoamPBUUAsz4xoCX81g20Qt2fFj6P+H3oTEpFNab9rsEcGHGSsZXJbKjJG96ekfScIJOzW22dca0AD1lkkDn19FTNVcXvm4jIuX1X4hhBBCCCHEjeq3D2IBnxHjCSaACaG+FoGQu1cQZOdTZPaeobwU3LQW6ZRLZgHPJZoHbJ7DiZo9h7B7DSh/mUP0Qy0sRG4lbVRKIT98b/y34S+m4M7FC6+OmRR8a1ayUkbukWyKzOPRoDiOfd+wv3G5roWOQcQkBLFvbjz7z9qvYovqFQyKlsDxS0k+kMMPGTPRxIex0vx2UAWzoE6x0WGhvJWWxu60NI59EERa+GoyrNL0GjqG4E1JfLIjlZUuwwm2OW1qVqRBg8+IOby3O4tv/5FA4JGpxO7UN2xE6RJA1JJEDuQU8sUsDXFPr7YK8k1s9XW9Hn0RaDtaHVcHTyL/FgsLFrLuVMv1E0IIIYQQQrQPN0QQa6Rpdoeuy4BQdMXxxG0qwFAPSkkmS1ak4zdxOD7GFPg/6knW+iSy9AooerKS1pPl+Sj+3c0y6upJYL8AImfHEvxxPAkF2NdKWovlxA3riR180U31JPHNeDLKjPXI37KQZxYexWARbZsvJzagWM0cAmgCZrJ8QCkZJ9rab1Z+PsRr/jqWHDZGz8olABc0pnq4PPAoXtnr2XxEj1KvoD+SxLpsTwIesDE3Xa9QerqISqpRrAPdLoMY8+wh5rycRPBzw433xtpVQILOn8lbio1trjdmpjVVynBwAX1ClpFxzrhNAXDW2J7ZtdXXSctYWRVB2AAbdzm7hbJwAWRltlhBIYQQQgghRDvxmz7YqVVdBvHW9lhiZ4XRZ6ECHV0Inp7CxqfcGpN4jV/He6enMu6R9SiAxjuU5evG276fsnso0bPXE7I4FV1CKO72vmqnhbSJUf4kNqQJiuOY6UFNPs+v472fZjHjsd4YAK1vBMs3zsTPPBLLXkiI/8LGl1EphTYe0KTBb/oiog6GWcxAt5l2ENErs5kR7U/PcwBaAudsJsrbtP2e8WxYWczkSYEk1AIdvRn9t3XGB2c1Ws+ou9cbf+ziS+RKs4dTmdUzcOQU3FP1jBmkhXPW2815E7VqDpMn6rhvvgJo8HlqFe89YcxU+8Qs3suexQz/XhgAugQQ88F40wcVzfk8v5nkS9FMNuvrtz6NJdDOBLv7iFgW7s9mTktVFEIIIYQQQrQLv1NVVbW1obC4hF6eHrY23ZjqFZRLmsYZR4Hx3mHrJbZmlFoFTQvbrwtFQemgsftdvUotaOS7XIUQQgghhBB23NgzsZfDwX5g9F+rlQD1Vw9gATR2lgk3bJYAVgghhBBCCNGCG+ieWCGEEEIIIYQQomUSxAohhBBCCCGEaDckiBVCCCGEEEII0W5IECuEEEIIIYQQot2QIFYIIYQQQgghRLshQawQQgghhBBCiHZDglghhBBCCCGEEO2GBLFCCCGEEEIIIdoNCWKFEEIIIYQQQrQbEsQKIYQQQgghhGg3JIgVQgghhBBCCNFuSBArhBBCCCGEEKLdkCBWCCGEEEIIIUS7IUGsEEIIIYQQQoh2Q4JYIYQQQgghhBDthgSxQgghhBBCCCHaDQlihRBCCCGEEEK0GxLECiGEEEIIIYRoNySIFUIIIYQQQgjRbkgQK4QQQgghhBCi3ZAgVgghhBBCCCFEuyFBrBBCCCGEEEKIdqPDb12Ba0GpKafo66Nk/e9JjleWYLhUTta/awBwv7kX7h1uw6ubD73/53EC+3jg7uT4G9dYCCGEEEIIIcSV+J2qqqqtDYXFJfTy9Pi169N2Sjm5WTv44OQO0i7VXdauLh0e5dnekTwV6IOL5jrVTwghhBBCCCHENdf+gth/V5G15y3mnz5K0S9XmddNzuh6zGf+UD9cbr4mtbssyqldJHxahIIXg6cMw8c8oK4vZt+KneSjwevJ8ejuaWO0XZLOku354B1K9BNu16Xe/5nK2LcslXxcCB4fgZ/2t66PEEIIIcR/OUMxGbuSSEnJpRRwfyCUsOeGE+zZzgdqFdkkrkoi5ZsyuNOPsPApRPZzaX0/i/5wwz8slDHDgvCy0R36I0msTE4l5ydjv42ZGkFg95az12euJzHPAPbGw22td72e3B1JJO7MpKi67eVfjnYVxBr+uZXZ6avZd7XBq7Wb/Hh90OtE/sHpGmfcGgP7Xgxk8i4Fr9lpHJjs3bilaFMYAxfmQVAcXySE4u7QxiyPx9MzbD08n8IPc3yvT7WvVN56QhbshCcXsPv5G6xu5BF3dxgJBBGflcDo7sb3EnQLSGM4f00bj99vXUUhhBBCiP8SyvFljHpuNfm11ls0BL+9jw1/aZ+TNcrxeELC1lNk9b7X8ynsnuOL3WmrslTGDY4hoxY03b3xopj8CgU6BhG/L4HRjd2hkBuvY9TaYqsMPIlKSSPmITslnEtnRv+ppNUCFuPhy6x3bR5LwsawskC5vPIvUzt5sFMd+XtmMnSfnQD2Jh8iPWawZlAix8cfoGTmIYt/p8bv4OCg+bzt8SeCbbX4l1zm7xvNxD2FWHf39aVl8OxYgoGid5axtcz0dsUu4t7JA3xZOO8yAtgbXb2e/BMF5Ff91hVpO/2JAvJP6H/ragghhBBC/BcpJnHuavJrNQTHpvF1YSE/fF/I12mxBHdUyHh5EWnnfus6XgElm7jn1lOEJ1GJOfzwfSE/5CQS5QlFa8cQl2kvEjGw7x1jABv8egbfZqWxO+skX7weBLWZzHknHUNDEUfieWZtMXiOJznH1G+J4/GimITn4slo9qGAKf8Fs0ir1aDpeHX1zt80i5UFCl7PJzYdt2RT+bPWk391Pdjoxg9if6kiK2UCQ78zLiMw595pBG8P2sKpaat5feQIBvt44KJt/tAmjdYZL58/MXrkfD6YdoDjg2YQ1cl61rWGfd9N4KmUoxiu9UxvS9xCiYn1BTKJfScdAwb2LZpLRi14zV5ApKdZ2no9WWtnEa7TERK5kK2nFPR7YgjR6Zizp3mgpZzaRdxkHSE6HZPjd5FvaJYEQ8EulkRHEqLTERI5iyW7CjDUWyWqN5C/axkzIo15hUcvI62geWYWeemmEpucjd6UV+5aHSFzU40vUuea1TmPBJ2OEN16cg15JEZHGn9uyLQim8T5UxvzjPs4rzFPbKaJZMYyy7bmrtXZ6COzcpt3i3HWWDeXrcYK84pOR0hMOi2Gs1Z1NW+/rX4MmbyQxCNNOZbummXs3w0FFtkq2cuM6d/IxMYhFEIIIYT4z1KRS3YxQAQvPOuN1jSho70/gpi3p/DCNC805oOiej1ZyQuZrDONe+cnkVVhlWer49mrHJPW68lKXkZitv3RouFgKom1QPgcovuZ1up2CSB6XgSgkLjjkJ2xnkL1eYAgnhjQNAPtPmAQwQDnq00TcQYyUpNQgMh5MwnsYuq3fjOJCQdqk0jLbF6CkrmMGbsU3MfFEt33auptQNEO54Vpsbw1MaDpuAU8YaxnmeGaTRje4EFsHbmfzODpH0ss376pF9P+3w6OPD+D0T6uaC6nFTc54uIzglef/5Tj/280gVb75v74Cs9+kv+rzsh6RSwg2hOUXbN4beEy4nYp4DmFt8Y1LS+mvpjEyGDC43eRVayHqv3EjnyBhNwy8k8UUGn9qUrZZl4ZOZdPSqGyuIB9a2cRMiqeXLN0+l1T6R8yi5U7jO1V8nex8kUdfSelUtp4UepJezGYkBdXk5avAAond6xmRkgg4z4ua8xLOR7PqJBZrNxfhMbLE/faoyTOj6T/pF0tB340zHbmkjh/DLE7shtnPpW81YT8KZLY5KPG41F7lISXw+hvXr+yVMY1pLnTEy9NEfvenUXIqGXkNhzEqgKbfXQtZ1mb6ppOaS2gP2Rsv249+fUACrnvhBHy4mr2FWnw8nJDOZZEbGQw43YY6+D+gB+aEwVkfZBp9imVQtae1eSfKMDLz492fgeIEEIIIUTrXLzw6giQxMpNlhMsXkNnEj1zJoMb7npU8lipCyZ8fhL7yhSo15ORvJDwP+lIaJwXaNt49mrGpIbPlhE+fzWx4XYmSICib3YBEOznY7FsWOMXgA5gV26z5bqmDqH3w55AJvszm+pbmplOBuD1sA/GO1OLyd0FEISfxcN2NPgHDAMg7RurZca12Sx5LQnFbTzxs4JsjjXbXm8tfuEziZ4ZgV+XpnTK8f1kAPi60YY7f9vkhg5iS9NnMKLEMoB1uXU0H4Wv46W+zledv0vfKXwU/hbTbrWcvc0tmcLE9BI7e10HDt5EvT0FLxTSNiVRiobIeVPwMztLDHuWEputgOd4th3LYvfuLL799HGKNmTbznNPAd0+yOHY7jSOfZ1B/AANFK/nzR2mE1/JZuXcdAwEsTAjhwNpaRw4ZkynHIwhbo/x8xTlyGpe2WOAoFi+OLaP3Wn7+PqLuGZLOfI/M66Rj0rIYtuSpbx3IIvk5wMY7FFJYQX4PZ/G7kWhxsShi9idlkb8UPPTOJ1855UcKyzkh+/n4EcZW99YRn6tL9FpWRxIS2P3gSy2Pe9pUT/90XQyaiEwLo0D7y1leUoWu+MGMfhRKC2+io8ifMezO20Ro40V5q20NHbHDbJz4TXUVYPub1l8fSCN3YezSB7nhlIQT8J+A1DAvrXFwHg2HE5h+ZJVHDicSFS/QXidKzIG+h5PEBYElKWS0fBLV8klcwdABLoBEsIKIYQQ4r+Agy9j3g7FC4WMhTr69PJnYOQslnycTZHVRGLpx4tYUqCgGbGUY8f2sXt3FscSx+NeW0DcGuMy27aOZ42ubEyq7RNEZD9vAmc+gY/NRukpPGX8ycvdakSpdaEbAMUUWs8gm/g8v47l4b7kxAymb4iOkJBABsbk4he+ig3Pmya+KopMwaQnzYroZiyBU0VmE0wKuWsWklCmIfKvMwm0tZT4iuttmtkOCaRPWBLVA2aS/H4E7rabd9lu2CBWObGa6fmWq6Zdbp/BljFTCLzjGhZ0x6O8NGYdb99uGchm5M/n3ROX99U9V0Nzry/+ja8C8L7P8qbnk1+nA6CbOQW/hhPMI5Qx4XYyDBpPVMON0w5ujB4XAUBuZp7xxM09ZFoWEM7ohk+yzNLt++woBiAn07QkIdLs3ly3UCY8BZBORrbpN4nGWFbGjiSyivUYFA2BcxJZ/ur4Nj6JLIAJ44NwaSijLJv9ecAToQx2UzCcM2A4p+D1ZCh+wL6vTeeGKX3O9lTSTpRhqAWvp1bx3usz0Xn/St+f1FDX++fwwjDTxe2gJXDuPr7OyeGvQcbg03iPQSY7t2RTVGFA0QQQk7iUmHEBpuDYhT8ODQLK2PeV6VOyE4fYWguaZwcRKF8HJYQQQoj/Eu5D4ziQlUL884Pw6a5QdGQXK1+OZOCDg5mzp2E2soysPXmANzFThjWOI7X95nAgJ4evFzyOlssYzwJXPCbtPoiFiWkkT2vh4UxXo7yA3IJiDCgoCqAoKBgoKsgmv/wK8yxYzyvvFqMZtpTooOs00KwyoKCgnM4nt+jaPWfmxgxilXzWHt5qMRXvcusY1oaNwKvTdSivkwejw5bzksWxK2Hx4UTyf5V1xQpZSxeyFdB20QKZxC5KN1sTr6f0e+NP3bqZz8ZpcO9pftOsmXu8LGcNe3oa16LXXzTm+KMpSOrU2fJCa0h3vhrFrFyNk+WJ3eu+IAAMtcYO8pu4mWhfLUUfLyR8YCB9/tCLPgOnsrKF+wIsaaCD2cuKYrIA9scw0N+fPg3/dPHG8+L7MvSAy/BYlg9zgeOrmaELpk/vXtwXGEbsx8W/3pLwhrp21dDZ/H0HDdouWrQdAXx54YOZ+HUpZuv8SAYG+nNfL38GTl5NllkXuQwYzmAgf2c2pUB+9n4UNIwe4Hd9fiEKIYQQQtyouvsyes4qdmed5IeTGWx7PRQvitn6wiwSSwD0FGcDdENjFSNoumjRdtHAZYxnTamuaEzaOi3dfm/8SamxGqWaxufgRrcuNKfksWTsVBLzXIhKzjGu+juQw9fJ4+mWl8Tksabb6Lq4mGY6FaqtB8L/Nv3/exfjkuH6YhLnL6OoYxBvzR3Uwi1rV1pvX6LSjA+g+iEjDv/ydJaETzUdt6t3QwaxpYdWsdi8j256lPm6CfhdjwC2QScfpo14GZ15jyibeefQ9V9WrBxfRuyGMnAbz3v71vGC6f7YJY1P+rJ/8lT+ZP34bJOfqy2DuCq9xRp7TcfbjD9cuoiF8jLjhWpMRWdTssYT36S01GoZc0dfXtiWw7df7mNbQhwx4QFQnM6S8DGstHxOUdt01BovwhFL+SInh6+t/y0fbgzSHdzQ/S2Lb09msDtxKQvHDcPLkEfiyzrjshF76i/a33aldVWgpVw1D01h27GTHDuQwoa35xDZD4r2LyP8mdWm+2aBLo8zeCiQd4isigIyUsugYwS6AAlhhRBCCPHfQakoIOtINlmnzMZyHd3wC49j4TiAPDK/0gMatG4AClyyl9tljGdtaeuYtFUa3L2My34z/mk1fj9tCpTv98Td1pCvIJOEYiBgPGMCmsJNbUAEEwKA4vXGW9E07njeD5BJvlURRaeMbfXxcjdOjOSlEpsHkE/CRNNDR3UTWHIMIJslE3SErM277Hor5wwYzhlQzB965RHKCxHQdNyu3o0XxP58lE3/NF9G7Ejkw/PRuV6Hsso/I+Gw2fy765O88fCfLE7EjH/uIOvn61B2AyWPlXPXU4RpLbqLLy/Mi0CDQuJry8iqBdDQu4/xk6KtyTubHmpkyGT/Fjv57jhk2hdAIWt7qvFLok03fmt9/Izfe5pung7yj+xEaUynxaev8ftc9x3MbgqK6wvI2qEAbvh5uwAGio5ksnXtLoq0nvgNCCXq9UTeGwdQTG6B1clqHTjb4ulLcEdgfy5FGq1xRrOLFu3Puew9ks/JCuNN+aV52aQlryfD4IZPv2FEvrqU914NAJTGG9e1rsYLr+jbosY2KMcOsa/1WjS6aPH0OYXSEwVNT6RrqGt2Khlm13bRpjB63t2L8C1lcK6YrIOpJOwsRuvpS/BfxrNw40qiAIrzONnYRVqChw4DMtm/aB0pZeA+cTh+/ylfsySEEEII0QpNVSZzIiMJn7ra4qGk1Jc1zqq636EFPPEboAGySTloNggrTmLU3b3oGZ5KaZvHs3a0aUxqZCguaHbPrjmvgOF4AaUf7bQcp39k/OoZrycD8LKVl0ZjvPf06wKKzPujtoiCrwFcTHf2eRL4pCdQRsp2s7bWZpOyvgDwRBdgWsXp4ILP/d74eLYegre93gUk6Pzp469jyTGz6bT6Yk5+ZfzReNyuXofWk/y69Md2kGD+FTedpvB8oPXX4VwD5Z/xcsrrbP3FkROXElkRbIyStYETmH3yM15uCFx/2cGmY5EEBl/9g6Rsyd8wl5XFWKxF1wTNZPmwVCbvWk/smifYPdMXl6FTiFqTScLBGAb2T8LLBSqLFTq7ArYmY12zidNNJXuoF5zOJHFXGXQMInqE6cZvj1DmPZ/EqLXrGRdWSdRzAbjkphK3pQw6DiPmKWM69xFzifogjIQNUYzSj2dMfxfyUuLZWgaaYXOJ9AbQUHnwBeZsUvDKq+SFcG+0FYdYuQXAl6CHTReH6ZOs0i3xzCCIwWPMnixnTRPAhAVBbH05iclh1URPDMVLOcS6RevJOqchMjGHwHs0ULCMGfPz0KTpWTjxcdxrC9i8PhvQENnP2Aavh5/AiwKKNkTR/5tBBHYtIKNSa7wYWzw6pk/3ypKImwXBQyOIfsIN/Y4XeOzlTBiwlGMJw3BprGsmsWGRFEf4oq3KIzE5DzoGERbkBh2z2f9iDIm1nuTqpzLG+zZKM1eTCOD7uMV9w9oBw4lkF4m7dgFuRAd5266eEEIIIcR/onuHM3XAauYcXM+o/pkMHvoEXppSsnbsIvcc4DmlcZVa4PhYglNjyFgYRnhRBH7OBnKTkshFQ/BfgkyzqG0Zz9rRpjEpcDyevmHrUTpOYdvJmcbJImv3RPDW86mMWrue8P4FREb4Ql4SiUcM0DGCmAhTgGmdl3co0QNWM+OgsQ5Rzw3HjzzSPlhPWi1oBsxitKkNXhGLiEoNI2FDJH1PRRDpC7lJSWSdA034HCLvMdXFdzy708ZbVVDP1nGBzMkMIHpdAqO7X2a98Wb0zCBWRmeSEB5IzohQAt0U8lNTyaiwPG5XTbXjVNFpe5uuo5/ULWseU+9a2ptfgEUAACAASURBVPDvT+o7Ocp1KOaAOnu5ZTnTDv7UuPlizjLVf6nZ9jXb1TPXvhaqWrRZHdnTS+3Rc4K6pdRqW/lOdayPl9qj5xPqu9+Y3qvOVz9ZNEV9cvhw9cnhU9R3syrVvy/yUnv09FLHplYa03y1SO3R00vtsShLPbN9ivpAT+P2Hn5j1LXfXLQs41Kl+vcVY5rS9PRSHxi5QN1rXZfKLPXdMX6NaXr09FNHvrpfPXPJLM3PReqWV0Mt8ro3YIK6NrfaLNFF9eSapvLGbq9UVTVXfbOhD8qtO+iiWpg6V33SpynPHn5j1Dc/M6vgpUr17ysmqA9bpAlVX9tZaplP0szGNPc+OVfdW5plKneRmqOqdutx8Zt16jN+pnwn7FQrVVWt/myu+kBPL/WB17PUi+ZlWNX13ifnqlu+M0vxXYr62kjzfvRRH56wTs0x7yJTXn9f4GNM03+VetJ6sxBCCCHEf7pLperni6zGeD391JGzNjcbO138LkV95UmfpnQ+w9VXUotUi5Fvq+PZqxyTfrdOfdLHS733mZSW44ZLlerni0JbHn/bysvWmNcnQB27IkutvGRVRmWG+ubIVsbuNlWqW8ba6YO21NvkzGeL1LEBPlblp6iFP7dWftv9TlVV1VZwW1hcQi9Pe9Nk10n5p0zc8nbTMs+bJ7Bnyhh8ruWi58YZWPM3HdE92DQbyy+FJKyewBuN6+b9WPHUsuuzpPly1dP4RF6A3PhejFoLupU5LB9qY3q+XsFgwHRju708FQwGBTpq0bb04YhiwFALGq0Wjb3lrQ15aRoeaGQnjaKxv91W0ecMKA4atC1UsE1plMYHKV8WxaCgMc+3hYyUcwaUltrfln4UQgghhBCAaWxFG8ZOtQbjGLOlce9VjsNaHG9ezkCztTFzS2NNgwGFlse8gKk/rvGYsy1jfYvyWzkeV+iGWk5s+CHX4j5Fn56P/voBLMBNvQjs6QqnGu6XzeX4DzXoXK/DsuY2Uo4vI+S51ZR6T2HDO+PprQWlbCcrkwB88fOxs77cQYPW1lPOmqVpw8mlaSXIbWteDpcXwILxCXOtFd2mNFd4DWmsG95CRq3Woy39KIQQQgghgLaN8QDjhExrY8yrHIe1WJfLGWi2NmZuaaypvYb9cbnaGjdcr/JNbqgHOxVVfGbxOvB/ruFMcFsDWBOv/3nU4nVixfV/SnFLNA9N4a3x3nB8NeHBxkd799UtJANvRq9cSuSvPGkuhBBCCCGEEL+FG2gmtorSc+avH+V+d8drk/VlBrAAGvdeDIammeFzp9Hj08ZHaF8PGvxmpvHtxDLyc0upBrilG73u98RFZvWEEEIIIYQQ/yVuoCD2PBctvrvJA/drETFeQQALgIsrvTALYv9d1+J3gP5qOrrh08/tt66FEEIIIYQQQvwmbqDlxDVUXrB6y07tSjM2s6/c9jYLVxrAAtyksVxrfqGca/PVvEIIIYQQQgghrtQNFMS2TennrzDiH+uYmDK/5UD2agJYIYQQQgghhBA3pBsoiHWi261Wb/1i+bL081cY8fVR44zoL1/YD2SvRQD7i4Ji/vpW19/wflghhBBCCCGEEHBDBbG3ccvN5q9LKLVev3uzI7eYv/7lCybueJsM80D2Ws3A6sspbKlsIYQQQgghhBC/uhsoiHXG3eL7TI9yorTOIoV7/9f56MFHLWdElU95bsdysv7FNV1CrJQWWnxnLV16yEysEEIIIYQQQvzGbqAgFry6/8niddb/Nv9uVvfgt9jRLJDdwdMpU3juGt4DW/S/Ry1eR3aXL2IVQlymCzXU1f/WlRBCCCGE+M9yQwWx2p5+DDZ7nf/DUfJ/aZ7OdiCbT8a1eojTL4Vk/WC+RtmPh3o6XX4+bVFfQe7GebwYEcG4iNms2ltETVsGvfq9vLVwL1VXVXgN2QuHErGxyFSXOmqsnhBdV3Plg/CT6yLYcsLGhmtS9ybl22cz7t0vqWs9KQBVexfw1l5bpVv1hz2XWf+qrI38dVIE4yIieHXJXgqr27jjZbA+Tnb7/hqq+W4vq2YZ2/XivI3kVrRlrwK2RGzk5NUWXpTMuCcWcMjUl83OUxvncptdk/OzggOzhzIkYhLxu9vUMf8hqji0cAGH5FHuQgghroY+nTk6HSGmf+HRy9ia17Y/LrlrdSTktZBvTDp6wHAwhvsGrqeVUV/7UV9GRvxUU38lkWuwk85QwNb5kaZ068my6FaFol3xTNbpCImcRUK2VZ+XZBI3WUeILpIZyXkYrGKE0oP299Vnr2dGpHHf2I8Lmu17uW6oIBZXP/7Yyez1vxPZ+w/boYl78Fvs6POonSW+V/cUYuUfn7LG/DtrOz2G33V5oHEVh+aNY4tDOLEJG3h/1TT8/hHDpHUFre9aX8fZyraGbfY4ETBxBW+O9DK+zFvDkDXmV30F6a/EkH6lA9KaCs4qNt6/JnVvyKuAA4mF1Bw+yMm2Bi0Xqjh7wVb5Vv1ht8y2178mfR7hWxx5esEG3k94n6kP5rEwag2F13R2zsZxstf310rBGibNz8Nvxvu8n/A+seGObBk3j0OtRn4KZ3+sufrye+qIXTWJxzuDzfY3O5cvw7U4Pyu+5EB5OO9vS+I1Xfery6tdqeN8ZRXnZfZZCCHE1aivpvKEH9EfbObDDzbz3vRHKVs0mNjMNgxuqgrQ2/s7VF9NZYXxE3Bt0Ex2vx9KK6O+dsLAvlk6NrtO5cOUFOKHFvPKqNXkN+uHMhLHzqIgKI5t21NYPkJP7DNN6Qx75jL5mB8xH6WwbckENBsmsCTP1Ofn0pkxMhm3KZvZlhKHrmguo9Y2xSyGPbOYvN+H6M0pbFsyBmXxmMZ9lbxlPPNaGbrXU9j2URxB+bMs9r0SDgsWLFhga0PVuWq6Ot9+VZlfvtv4H8O3rK0oM72u59i/uvIX/z/Q2Ubqzj0HMuTCt+z63zJqG9+92q/RKWfbrjdIMwtiB/u8QnhP60cnXwvf8umbGnTLQvC42QGHWzrj7vsI99zqxO2uTjhU57H9kwrce7viCGD+uuY7DnwBD/et5sCGHRwp+pk7PHpy+y0AZzi05Xu0d5xh78YdfF3VlR53O1NzNJmkTw5RUu/BH9yNM8s1hQc5cs6DP1w8QsKH6Xx9qpT6sio093lQ+clGdmafoKRKz+mfXfH37Az1NZRkpJL66SEKf3TkzntcubXho5D6Ggr3J7N93z8od/SgY0kqZXdF0vdOq2a3UPeqrI2knXalt0fDzHcVJ5O3UNbNF1dbk+H523mn9s/M80klzRDC456OTdvs1PXCP/dypN6XHmfS2L7viO3+cDXmU3U8jbRtB8gu+pk77+7JbY4N9a/jD15nOPDRPrKLoKePW1M/mDm1axGaYYsZfI8DDjc7ctvdvgR43YpjF1duuxnQF3Hgk2Q+P1RAuYMr99xpqkdeKtk/u1K1dx27K0x9X3OG3N3b2f3ZEUpqXHH36IzjTTWc3NL8OFUeTaSk+yA6fLGR3YcKqLq9F553mPVNRR67t3zMob8XUdftbty7ODavfAuqDifyde/pjHvUGYebHbnV5QH8H3KiYyc3br/1DIfezUHzSE9uA8D8dQVfbTpDj7905evEZD7PrUBz5310Mx1bY7s9+PeXyWzfV8DvPB7Atb6IA0nJfJ5bhdazF11vAW6q4Ovk79A84kKJdftvyrE6l33odksrba74ku0bjefinV3rOPol9HvyPppf9TWUZ+3l00+M50TTNWemOo/tG7ZzpPhHavRF1HV7hB5d7JRfV8SB1V/S4aFe3H4TUPUl29ed4vaGvqs4wpa9Z+nhY/odYH4MbJ2bAPUVnPwklTSL86QN2+z1T3Ue27/4GfdLX7L9o32cbHbdV5D70UZ2HyqizrUbF/7+JTw+hHucaPn3hTk710GzY3O3A7nbv+eOht+Jbc1fCCFE+1JTQNp6A4/NGUCPWzVobncnsOc5Jm+tI3KIJ8rxJLb+1IM+d2oAMJi9Lj/yLkV3T8Q1ZxMb9+RRerM7ve/s1JTvQRio86FTzSl2f36Ong+4ogGoyGPrh1vY/0UxSrce9HTW2Kya/ngqH25NJ/N7BXevHnRueDBtvZ7c7Vv4cP8himq70dPDGY3Z31ibeRvySMys5U79Ht7/uAL3AE861xsoykhlY9pB8n+8hbvudaWTKR9DWRlKJ21Tvg0qdhKX9DAx8QNw7dCBzncH4F4YxWedp9DffByuP8T777syNm4Arjd1oNNd96LJTKYyYDi9nSA/YzXOf3qV4Ls60KGTC/d1OsHyH3wY/YAW/afxbPSbw1uDXOnQQUvPADeKJxyk88R+uN5UxtY5eQxeMQV/pw506ORKn//ng0MHLe7OGr7ZFEntMx8y9kENHW7W0rO/J2Xj0uk0sR+uV/h3+4b7c+/SdwRR5rX6eTVrs+zP3Lj/0XxG9uq/B9aQtY53fjZ746YRPNvX+Yrza1lXXHse5tPtBU1LiJ3uordvd+MA7cIZjh4+07RM1vr1j3tJSDzDPbpQBnXPJ37Su5ysATjLyS1LWLn9Z/x1A3A69BKzJ83jo+pHCBniRcnySSSalprW/XCYoz/UQOe78L+3K3Tzxt/fi66OjnS91xsPJyc8vPvif7cTUMPJdycRX+DKoFAdvas/ZOzMNNOyyyoOzQtjaZEHg3T9uTNnBSuzWmi6nbo7uzlxaONhGhdzVxwm8aATrnYms04eTMO/X1969x/CyQOHMT9TCtdNIr7Ai5DIUPxrNhL+9pHGvjudmkR25/6EDPHh7MZxLDtcZ9kfGGdRX9oJ/qFjCel+lNnTU5vq9WMq2w87E6AbQu8qy7wtjnD3u8jemUZh4wylE66+vrjeClSk8fKzizndfQijQh+Bj6fz1/SqxnokzF9CjrOp7y/kkTDuJQ7UPUhIpI4eJxaYZuxtHSejQxtT+clXR0iAE7nzp7P9TEPd03h5Zhp1D+oYNcSZ3JhJJORd3syjk0t3TuxNJffHpv2c730ED2eAs5zcns/Zxi3Wr/NIWH4Qp6BQQh6s49OZ09n9I03tXryGEy4DCPGvImHSJF5cfhjXQToC2MukN/aajnFDnjba3+xcbqXNBWuImL4XAnSE+J5n+7JkTttsdR0n3x3H7H119NaNJaRnPvG2ZtUdXbnH2wMnJw96+/elR+cWynfsjuOPazj0nXHXmuM7WbZ9A9k/GF9XZaVwqL4r1p/f2D036yvYPfMlPq17kJDQIdyWE8P01XnGc9N627GXmNSwzaJ+rpyMm8Sq46ar6cIZjq5bwKrD3Xk8dAhdv1vEpHcb8iwg8enpHHDoR4jOl7NbFrPlx6Z6tnQNNmrhOmh2bN6IY0vj70Cr30d1abw0M41ymQUWQoj/TB21dDP9qJw+RObppllZ69c5K+LZ1zGIyJF+KCkTGPdxGc3UFpF5sMj4lZplqYybkIriF0rk8M5kTgprmoE0Y9gzi7EpEPj0eMK6H2LcmCRKAerL2DppAimKH2FPD6fzkamMWpFnI283cueOIe640lSHxbN482hnAh72pDMKue+M4ZVv3NA9HYp/1TqenLTL+PWihnTiHgtmyWEbs9E/FJMR4GM2q6yhd58A8ous2u3iS4BbJhl5prXGJfvZX+pLb9PSVr/n04jybUpeVJBLrzu0ABR+m0mgj2fTRo0P3n0LKCwH9HlkdvbFXZ9J4hsLiX0jiZwOvgR6apuSa6w+FKgtpvBqbj9S7ThVdNrepuvuzP7J6l1LH2v6t/xl9ZOfWtpDUQv3v6rO/rzFRK37abc6bfljFmU/u/8698P5QjV98Utq+JAhavjEGPWjz0rU85dM28o/UWdP+UQ9q9p4Xf6JOjsoVs0635TV6U3h6uxt5aqq5qorgxarOQ35fLVC7ff6YVUxvTy7bZopneXP6lcr1H7Lcs0qV67umjJN3WXarJakqJNe2qM2FXlezZqvUzflq6r6fZI61ryu6lk1/aUB6sqvbLS5xbqXq7umjFW3lRjf/+mjseqkj0ps913tYXXpkBXqiUuqqqol6rZws7qqZ9X0aeHqR9+apb/U1P7pKeVN7x9d3Nhu8/44sWKIGr//fLP91fJP1NkTUtTGs+3SUXVl0Ar1hM1KnldP71mszhs5RB08cqK6cMNB9fT/NW1VFLOkVvWwqKN14rIUdXpjf1sdJ1VVTywboK482vT6p5SJpnYpak5cqLrpG7N8v92gjn3loNlxbWjXWfXU3wvVs5esNxidPbpBXThWpw7WhavzFn+inmgsP9eqP8xf56orgyyPi5LxpvpHm+0uV3dNCVe3fW8vn4afm7ff8lxuqc2KmvX6EMvz9Jv31VCLc9m8TxRVaeyPcnXXBKtyGzeZX7st9/n5PS+pYzeVqKqqqFmvT1M3vR9jOlbn1cxXzNvfxN65qXzxphq+Kt/s/UL1o/AYNfP/jNsGxzX9Hmg4vudVRc2JG6Iu/cLs/CpJUSeNNZ3j1uf7/x1UF+reV0+plsfOmGe+uknX0Cf2r0Frtq+DVo6NeR2NuTRvhxBCiPapPEUd23OB+nlVtVpdVa1Wl+ermyb4qS9+Wq2qqqpWpk5Qx6ZWNiY3f52zyEt9cn1RU14XM9TXfBYZx8XlKerYsSlqpWr+80X17wv81NcOX2zapzJf/ft31c2qlbPUR31ln9n7pr9rFw8vUB9YkKU25nCpUj15uEittpX36c3qyJGb1TMNdXgmxfhzw7apO9WmEi6qn88OUt9t+NP+s1k+Zqz7Q1VVVf1qkdpjUW7zxKd3qi/29VJ79PRSe/QMVd/Nbd5OVVXVi18tUp+ckKKeuaSqqlqpbhk7Qd1iNebJWeSlvvmVqR39n1BHTl+n5pyuVs98tU6dFDBB3VJq1j8TNqsnq1VVvVStnvxggnpvz+b5XY4OVxH/Xjfuj0/lpaIpLG74oOGXo7yetg73MRPw62RrD0e8Br3O21dT6M/5vJu2nDTzh0NpxjD78ev8VGInLwZGv8PAaKg7k8fudS8xqWAuG6b5Nls+2IyXDz3Mpmg8vB8h+0jDXKEjjg7m5XRqPb/WVFZwMv8w0yOSG9+qq6ohYDhQW0VhL2+a5qydcf09nLrsuvsSFNKV6ZlnGBkJ2eldGbn4LptZ1H35Bdvvc+XRL78kF6DXGT7NPEPIU3cBzjweNYDZM4ey+y5fAvoPYKTuj41LkjXmfeNgu2d6j3yJT18KQ7fRh6AB/Rio0zV+UoWjI5pW9jdywmNING8MiYYLZziZtoF5UfnM2TSN3rfWcPZwGts/P0xOSQ3UVEHQHxv3tKgjUJW/l/RPj5D+bTlcqKGk+9gWygXM9tc01rGKn8pq2BYXQWbD9voaSpzHNp8h+/EgS+cd5Mmk9wn5ffPsnR8Zy2uPjIW6KkoOJxM/bjojN6xgYKu3gPpyj9kNKI73+eCfeoYqfE11tSgFx6teyd9Sm6s4W+6Fh5tZcpfu9LCbVT7paXs5mlnAT/V1nK3qTtRVlQ/Ovv244/U8qsLvIvf7RwiZ4MTJ5QXUDHEkt2wII3s2z9HeuVmjr6A8PY5xZqsgzuqdibpg3Hb/vV5NvwccnOkV6AxU8FOZF716mZ3Hd3nR+4cjnAVcwfJ879yFrtV1KEBNVQX+d5mtfHFwxrnxXGn5Gmxi7zpo5dhUVnDygX40le5Ij3u92KKvAv6b7kMWQoj/VDuJey7X+Pensye65/exvJ+2tZ0ACPyD9YxhMoV68LOZ2kBpiR/eXmYzhS7eBNp48I5f6FJSJgXT991HGTI8CN2IUPxcwFBRhr+Pl9nY0AWffi6AvnneHj745+1HD7gDaKDxziR9GbnZSTyjW9eYXCkrIzjM9KKj7SXO2jvcyMorA7OnBelLi/Fxteqvc+nMmJDL4LRClncHzmUTNyaatHUJmD/CQzm+jFEL4K8pobg7AGjp9vtscssx+xOrp7TYG7dQ08uyAF44MB4/DeAxnrfm5tI3NY/RM33R9JvDtop4XgnuRS6+RC6YSUzQBpttaasbMohF48Pz/Ufz+WdbjcEJoL+wmedTurIlbAReNgPZq/BzCVtTZrDY4mE/HrzUPxIf2+fKtWF6gqqTk3Hw6HiXLyNjoykfeoRT03zp3dr++iqL5bN1NedxvVUDXMcn+gRFs+HlR5q/fxyoswyD6lpaoWq37uD0yB+5c/phSoIcOdD1j7xjczV3FYc+PUZA9wGczDlmfMupF+fTv6T8qbtwBRzvH8vf9oylruYMp7av4cXpZ/nbhlDafEh//0de+eiPcKGCkqxU4iPiiPokxs4vQNvqqmvAycn4gcKtd9H7qQXMOjOU7G+n0aNqMbOzH+OduSuY6uQIx9+l/xE7GX23kenrIHb+mzzV3dG4BPP1y6iIhbuIWmw7MLVMFsp7GaE2N9XV1MCtpnY5OuMxYBqvVU1i2ZdVDNS1Vn4V52ug8Ub3CzX86zbjPY7X6HFfNthrc4WxVKsnGys2P/IpYsvUjbAghtfGdsfRoYLdU+Ousnyguy+PXljDiawKTj7Sn6ndnXns/AZOfHkbOf0HMNNWdnbOzR5A7/DF/C20eRBX5QCK3Q5u3gdt/dRLsV6+a1aGvWvQ/IaPmoP2rgNHNA5VnDd/YJv1sbH6JaPU14HjVX9cJ4QQ4oYQyltpcy5r3NWg8pwBaAjgDChnb6NbRzB7iI6FWxwULl5qQ8Zug4jfPQhq9RRlr+eV4IVE58TSywEUxd74WwHzvOsv2g1GARgRy+5XA9pQmSaabi50Ky7DgG9jqyvLivDytIzEDdn7SBswxhjAAnQJYEzEe8Qe0aP7iyltWSqTFxiI3hiLX8fGEuh2pwtFZQbwNZVQr6f0O0+8XAA6o3VzoZtZs7R3uKPkm17UKriPiGXbX2JN++YRt8KXq3n25Q13T2wDzf1TWOHjY/Ge/v+W89Tm1WT96xoW9K+jLN48gZf/z3IwFOzzOtPuv86DoeqDxI9aTLbZV67UfHmEQw94GWcbHIAfizht2l6eeZBsi/33kv6lKRSsryA99Rghgd5XXh8HcKw6ZzYGNQ4iaxrq5+PLyMyDTfWtr+DAkneNX63i48vIzOTGexup/pxDB1soq6W6Ow/gyftSWfVWGvc8OaDZ/YAAVH3JocIhREVPI2qa6V90NE/dmsyBAoAzHHptDbk14Oh0F71HDsD/xwqz+zJbU8PJdXHs/gG4tTseQUN43KWCs5f1YN0qDr0RxqpMs52qvyQ7y4ceHlBXXUXXe71xNX2IUXjssP2sztdQ4uFNr+6mB07lHSOncaPVcWpRd/z71/FpetMD5euOb2RZatFlBZAliRG8mGj2UPr6M2RnVtD7buMnDo6OhZwy3YNbk/U5hyz2/pJP0xu+dqaG3NRU7nz8QdvHuU1stN/iXG6pzcZtW1LzGttfkr638cMzS+c5q/fgnnu7G4N3fR65bXouf2t9fhf+QSWsWnKQ3v7eQHfuf6SIVavzGGTzerZ/bjo/1J+6vXspaQgsa/LY8kYqhXXg7NufutRU033zwA/JTB6XSrmNPihPS2Z3kC/3tNIy54es8vxuL582NrNt16D968AZ/wHObE/+nKp6oL6C7OS0pq9nsv6dU5PH9tQ6HvW9Xs8wEEIIccNwgKJvi4xf01Jfxt49mRab0z7eT6npb6GSnco692AC7U7iuuAXdJF1SXmN00BFm8IYlWx9H62B3LUL2VoMdHTBKyiUwZ5l6A3g4vc4F9cnkdsQJBcnMWpUEqW4EDjAMu/SHRvYGvooPthw/+NE7kgno+HrcerLSItfRlYFgELp8TxKbQXi3sOZUL6KlQ332palsuSDIMIGaKFeT/4R4xObtT28cD+V39g31JeRkZ6LV4+GADadOZPSeeL9WIKtZqJ9ho6ncsXqxjaW7lhGwoBQgrWA9lEG++4k7YihMd+tG1LRPWCcETccnMvARdlNfZC6mvyIJ2z3QRvdmDOxJu6DlrOjZgIjSkoa39Nf2MrTySd56fE3mPbg1Q1W9MdWMz17K1lW30Xr57GaNYOu8zJiAOchzJpbwLynh7LK2RnHC1WUOw3gtcVDjAN6lwFE6fYy/c9DobMzIyMHYPG5jLcOj2PTGfduHdTUcJsujnfuv4r6PDCEqWumo/vzUeZsiuHxzs4EhPoyfeafOfRUHO9F9mPqW4XMfvbPJDg5NZXZHaAfU/+az4tT/8x2JyfqnAYQFe5t//tAW6y7I/6P9+XVN+CdR2x/kFCVuZdTQybRy2LpaXeCQnoxLrOASG9v7ulXzryIMHBypO6CIwGvLqY3tPH7P5245xFXEmYa28OFOno8tZg5Lhgn79rEmYEvx3DqpTCGrHbG9dY6yqucCHl5MQOdgUFj6R01iYg0JxzrHQl4pIUPIHxDeW3rJHRPO9EV6BFoPlNvfZxa/iDDNTSOp954Cd0o6HprHWfrvZm1eOxlLTfvFbmYgW+8xJA/O+HqBGeravCPXMyc+wF8CYlOY17UUBIcHPGfGM7jFp02hMcc32Xc0yVQVwMPTOfNQVdzLdtov/W53EKbXUPjeHbeJEJHOdEVR+6douOp47bK8WXkjA+ZNCqCrk6A2yP4217p3kxrfe7RfwhOG0vwM/0293joEeqS6/CzeShbODcJ5bWnFjBvlOm8r4H7pyzmKUfgrlBem7Cg8Zo4W+PKs2/FGmdFQ+OIeuMlQkdBV8caznbWsWJxv9bPibtCeS18HpMiwujqBJq7J/GkruGrje6yew2ac27hOnDWxfFm/QoWPruB807ejAwdgH/Dn4NbTb+PGn7nmNo6so3HRAghRPvl8sRMxuwYQ5++4OIaQfTIIIvtUYNuYYlOR5GDQmW9H399fxgtLUR2f2op0fOn0n8gdNPoqewxhQ/fdrNKpcUnwI0lYwLZ7OICtQpe41cR6QIQwfKZMUz+02Bw0VCpd+OFD5Yalwtb5+08ng/fD7C9MlATQMwHBYwbHMgSFxfQ6+n8lxFWdQAABDRJREFU3Do2dAeUXNY9F4UmIYeYAOu93Yj82yzmTApmoIML1frORH2QQKAGKNnPa5HphH2RyGjv8WwYupDJfQPBzQVFr8f9uRTeewhAT9rCqWwtgK2P9WJOQ9bPp/DDHF/wMGujs8GqHVoGL1pK6aTB9I13oXN5Gd0mbmbDUGOva4cuYOHXE+gbCO6d9PDwAt573ZOr8TtVVVVbGwqLS+jl+SsEcq35pYqsj2fw9I8lzTa5dxrBtP83mj//wbX5o6bt5leH/p+fsubv60j4ufm0mt/v32LTXx5F+yvPUddV11Dn6ITTldz/d8G4r6ND60mvFYvlpOaslki3yk7d6w7HocsZwt4Zvrb3a6v6OmpqwKnzlc+qWywJvlIXaqipc7RZD7t9aS8fByfa2r0tqq8z1ulq7jmtq6HmAjheSf/U1VDDNWpLW7XU5rZeR5d7jre1/CvQ0rlZV12Ho63zvqVroq6Gmvor+D3UUrvaeA3a/51C0/3dP6byYpzj/2/v7kHrKgM4Dv8JCReqXrAQP4aLQRqKJiDEoRIHh1Bpu2QVhDgKHQVxkNChU51EsJildKxLioONiKJTkopGhFskUNA0Q9QhhUMkHJMrDkGs9pZ8tvg2zzOfe857uNzD+R3u+56c+3A8dz72OJDfKAAPl06d6vek2dzF3MC6StVppnlkm81uV0mzedf6JenUqaqk+XiXY+5w3/9sXiWPdDnGtp+r09jBOde36zS6jXM7nTpV3bj3edR16t5G93F36lSbjezmK7mX/3/EJkn+yI1r7+TNxYWtZaz/q2coE62Tefn4SF5sPZ3+5r9vlupqNcvLC/lucS4zy1/kqz+77eTRnDr+fj44M7jzOZPcB7fSnr6eT6/MZPDCpa6L2gCHw+rMu3l9+qm8dXYsA1nK3NRU2qc/ynv7mUQDABSvkIjdUv34cd7+/GI+6xqh+9AzkvOvns/Ec3uflccBWf8l7e+X0nj2RAbdp8Kht/bT9Xz95TdZWT+agbGxnHzehQEADruiIjZJsrGa2WsXMvnzfG7uN2Z7jmZ8YDKTZ0bS33cgowMAAOA+Ki9i/1avZGH2ai63r+aTzd29mKO/96W8MTyR10aH0u+/wwAAAMUoN2LvUK+t5OYP85n9tZ1vf1tKtbmS2Y2tRZtafYNp9T6WY08MZfjJVzL6wjNpPdCVZAAAADgoD0XEAgAAcDg84BfJAAAAwN6JWAAAAIohYgEAACiGiAUAAKAYIhYAAIBiiFgAAACKIWIBAAAohogFAACgGCIWAACAYohYAAAAiiFiAQAAKIaIBQAAoBgiFgAAgGKIWAAAAIohYgEAACiGiAUAAKAYIhYAAIBiiFgAAACKIWIBAAAohogFAACgGCIWAACAYohYAAAAiiFiAQAAKIaIBQAAoBgiFgAAgGKIWAAAAIohYgEAACjGXzO7houIn/kXAAAAAElFTkSuQmCC)

# ###LGBM

# In[ ]:


from lightgbm import LGBMClassifier
modellgbm = LGBMClassifier()
# Number of trees in random forest
n_estimators_lg = [int(x) for x in np.linspace(start = 50, stop = 400, num = 10)]
# Maximum number of levels in tree
max_depth_lg = [int(x) for x in np.linspace(10, 110, num = 11)]
# Minimum number of samples required to split a node
min_samples_split_lg = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf_lg = [1, 2, 4]

learning_rate=[0.01,0.1,0.02,0.03,0.2,0.4,0.5,0.6]

num_leave=[int(x) for x in np.linspace(10, 110, num = 11)]
min_child_weight=[x for x in np.linspace(0.01, 5, num = 10)]

random_grid_lg = {'n_estimators': n_estimators_lg,
               'max_depth': max_depth_lg,
               'min_samples_split': min_samples_split_lg,
               'min_samples_leaf': min_samples_leaf_lg,
               'learning_rate':learning_rate,
               'num_leaves':num_leave,
               'min_child_weight':min_child_weight
              }


# In[ ]:


find_best_params_pca(modellgbm,params)# Best param with PCA model


# In[ ]:


find_best_params(modellgbm,params)# Best param without PCA model


# In[ ]:



clf_lgbm_pca = LGBMClassifier(boosting_type='gbdt',class_weight='balanced',max_features= 'auto',learning_rate=0.2, max_depth=4,
               n_estimators=80, num_leaves=40, subsample=0.5,random_state=500)

clf_lgbm_pca.fit(X_train_pca,y_train)
y_train_pred_pca = clf_lgbm_pca.predict_proba(X_train_pca)


clf_lgbm = LGBMClassifier(boosting_type='gbdt', class_weight='balanced', colsample_bytree=0.5,
               importance_type='split', learning_rate=0.02, max_depth=7,
               min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
               n_estimators=375, n_jobs=-1, nthread=-1, num_leaves=20,
               objective='binary', random_state=42, reg_alpha=0.4,
               reg_lambda=0.0, scale_pos_weight=1, silent=True, subsample=0.6,
               subsample_for_bin=200000, subsample_freq=0)
clf_lgbm.fit(X_train_std, y_train)

y_train_pred = clf_lgbm.predict_proba(X_train_std)

train_fpr_lgbm, train_tpr_lgbm, tr_thresholds_lgbm = roc_curve(y_train, y_train_pred[:,1])
plt.plot(train_fpr_lgbm, train_tpr_lgbm, label="train LGBM AUC ="+str(auc(train_fpr_lgbm, train_tpr_lgbm)))

train_fpr_pca, train_tpr_pca, tr_thresholds_pca = roc_curve(y_train, y_train_pred_pca[:,1])
plt.plot(train_fpr_pca, train_tpr_pca, label="train pca AUC ="+str(auc(train_fpr_pca, train_tpr_pca)))

plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ERROR PLOTS")
plt.grid(True)
plt.show()


# Saving models to pickle file

# In[ ]:


import pickle
model_name = "LGBM.pkl"
with open(model_name, 'wb') as file: pickle.dump(clf_lgbm, file)


# In[ ]:


#probabilistic perdiction on test data
y_test_pred_lgbm = clf_lgbm.predict_proba(X_test_std)[:,1] #test data


# In[ ]:


#perdiction on test
sample_submission=pd.read_csv("/content/drive/MyDrive/santander-customer-satisfaction/sample_submission.csv")
sample_submission["TARGET"]=y_test_pred_lgbm
sample_submission.to_csv("LGBM.csv",index=False)


# ![image.png](data:image/png;base64,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)

# ###Voting Classifier

# Voting Classifier is an estimator that combines models representing different classification algorithms associated with individual weights for confidence. The Voting classifier estimator built by combining different classification models turns out to be stronger meta-classifier that balances out the individual classifiers’ weaknesses on a particular dataset. Voting classifier takes majority voting based on weights applied to the class or class probabilities and assigns a class label to a record based on majority vote.

# In[ ]:


from sklearn.ensemble import VotingClassifier
eclf1 =estimators=[('lr', clf_lg), ('rf', clf_RM), ('lgbm', clf_lgbm),('xg',clf_Xg)]
clf_ensemble = VotingClassifier(estimators=eclf1,voting='soft')
clf_ensemble.fit(X_train_std, y_train)

y_train_pred = clf_ensemble.predict_proba(X_train_std)

train_fpr_vo, train_tpr_vo, tr_thresholds_vo = roc_curve(y_train, y_train_pred[:,1])
plt.plot(train_fpr_vo, train_tpr_vo, label="train votting AUC ="+str(auc(train_fpr_vo, train_tpr_vo)))

plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ERROR PLOTS")
plt.grid(True)
plt.show()


# Saving models to pickle file

# In[ ]:


import pickle
model_name = "votting.pkl"
with open(model_name, 'wb') as file: pickle.dump(clf_ensemble, file)


# In[ ]:


#probabilistic perdiction on test data
y_test_pred_votting = clf_ensemble.predict_proba(X_test_std)[:,1] #test data


# In[ ]:


#perdiction on test
sample_submission=pd.read_csv("/content/drive/MyDrive/santander-customer-satisfaction/sample_submission.csv")
sample_submission["TARGET"]=y_test_pred_votting
sample_submission.to_csv("VottingClassfier.csv",index=False)


# ![image.png](data:image/png;base64,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)

# ###Stacking Classifier

# In[49]:


from sklearn.ensemble import StackingClassifier
eclfstack =[('lr', clf_lg), ('rf', clf_RM), ('lgbm', clf_lgbm),('xg',clf_Xg)]
clf_Stack1 = StackingClassifier(
    estimators=eclfstack, final_estimator=XGBClassifier(),stack_method='auto',
                         n_jobs=-1,)

clf_Stack1.fit(X_train_std, y_train)
y_train_pred_stack1 = clf_Stack1.predict_proba(X_train_std)

train_fpr_stack1, train_tpr_stack1, tr_thresholds_stack1 = roc_curve(y_train, y_train_pred_stack1[:,1])
plt.plot(train_fpr_stack1, train_tpr_stack1, label="train Stacking AUC ="+str(auc(train_fpr_stack1, train_tpr_stack1)))

plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ERROR PLOTS")
plt.grid(True)
plt.show()


# In[53]:



# In[54]:


#probabilistic perdiction on test data
y_test_pred_Stacking1 = clf_Stack1.predict_proba(X_test_std)[:,1] #test data


# In[55]:


#perdiction on test
sample_submission=pd.read_csv("/content/drive/MyDrive/santander-customer-satisfaction/sample_submission.csv")
sample_submission["TARGET"]=y_test_pred_Stacking1
sample_submission.to_csv("Stackingxg.csv",index=False)


# ![image.png](data:image/png;base64,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)

# As we can see model with PCA is not performing well so will consider Model without PCA which is fairly doing well as compare to PCA.
# Now we are plotting training  roc graph without PCA.

# In[56]:


plt.figure(figsize=(12,6))
plt.plot(train_fpr, train_tpr, label="train Logistic Regression AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(train_fpr_rm, train_tpr_rm, label="train Random forest AUC ="+str(auc(train_fpr_rm, train_tpr_rm)))
plt.plot(train_fpr_xg, train_tpr_xg, label="train xgboost AUC ="+str(auc(train_fpr_xg, train_tpr_xg)))
plt.plot(train_fpr_lgbm, train_tpr_lgbm, label="train LGBM AUC ="+str(auc(train_fpr_lgbm, train_tpr_lgbm)))
plt.plot(train_fpr_vo, train_tpr_vo, label="train votting AUC ="+str(auc(train_fpr_vo, train_tpr_vo)))
plt.plot(train_fpr_stack1, train_tpr_stack1, label="train Stacking AUC ="+str(auc(train_fpr_stack1, train_tpr_stack1)))

plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ERROR PLOTS")
plt.grid(True)
plt.show()


# ###Ensemble(Weighted Averaging)
# This is an extension of the averaging method. All models are assigned different weights defining the importance of each model for prediction.
# 
# https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/

# In[ ]:


Weighte_dprediction= (y_test_pred_lg *0.1 + y_test_pred_Rm *0.1 + y_test_pred_xg*0.4 + y_test_pred_lgbm*0.4)
sample_submission=pd.read_csv("/content/drive/MyDrive/santander-customer-satisfaction/sample_submission.csv")
sample_submission["TARGET"]=Weighte_dprediction
sample_submission.to_csv("Ensemble_Result.csv",index=False)


# ![image.png](data:image/png;base64,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)

# In[57]:


from prettytable import PrettyTable
table = PrettyTable()
table.field_names = ["Sequence", "Model","AUC Tarin","Public AUC Test score"]
table.add_row(['1',"Logistic Regression",'0.80' ,'0.795'])
table.add_row(['2',"Random Forest","0.92",'0.821'])
table.add_row(['3',"XGBoost","0.894",'0.820'])
table.add_row(['4',"LGBM","0.875",'0.8378'])
table.add_row(['5',"Votting classfier","0.90",'0.830'])
table.add_row(['6',"Ensemble(Weighted Averaging)","-",'0.8371'])
table.add_row(['7',"Stacking","-",'0.8361'])
print(table)


# ###Observation

# 
# 
# 1.   PCA is not usefull in this dataset.
# 2.   We applied logistic regression,Random Forest,Xgboost,LGBM,Votting Classifier,Ensemble(Weigted Average).
# 3.   Best model LGBM and  Ensemble(Weighted Averaging) which is giving good  Performace on test data,little bit LGBM perform better than Ensemble(Weighted Averaging).
# 
# 
# 
# 
# 
