#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import json
from datetime import datetime


# In[2]:


import sklearn
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score

import pickle
import os


# In[ ]:





# In[3]:


# loading data

with open(r'C:\Users\ASUS\Desktop\CAPSTONE PROJECT\Group2 capstone\behaviour.json') as file:
    data = json.load(file)
    
data_list = []

for item in data:
    for key, value in item.items():
        
        id_num = key[3:]
         
        new_item = {
            'ID': id_num,
            'Recency': value['Recency'],
            'MntWines': value['MntWines'],
            'MntFruits': value['MntFruits'],
            'MntMeatProducts': value['MntMeatProducts'],
            'MntFishProducts': value['MntFishProducts'],
            'MntSweetProducts': value['MntSweetProducts'],
            'MntGoldProds': value['MntGoldProds'],
            'NumDealsPurchases':value['NumDealsPurchases'],
            'NumWebPurchases':value['NumWebPurchases'],
            'NumCatalogPurchases':value['NumCatalogPurchases'],
            'NumStorePurchases':value['NumStorePurchases'],
            'NumWebVisitsMonth':value['NumWebVisitsMonth']
            
              }
          
   
         # Append the new dictionary to the data list
    data_list.append(new_item)
    
    df = pd.DataFrame(data_list)
df


# In[4]:


with open(r'C:\Users\ASUS\Desktop\CAPSTONE PROJECT\Group2 capstone\campaign.json') as file:
    data1 = json.load(file)
    
    data_list1 = []

for item in data1:
    for key, value in item.items():
       
        id_num = key[3:]
        new_item1 = {
            'ID': id_num,
            'AcceptedCmp1': value['AcceptedCmp1'],
            'AcceptedCmp2': value['AcceptedCmp2'],
            'AcceptedCmp3': value['AcceptedCmp3'],
            'AcceptedCmp4': value['AcceptedCmp4'],
            'AcceptedCmp5': value['AcceptedCmp5'],
            'Response': value['Response'],
            'Complain': value['Complain']
        }
        
         # Append the new dictionary to the data list
        data_list1.append(new_item1)
        
# Create a new DataFrame from the data list 
df1 = pd.DataFrame(data_list1)
df1


# In[5]:


df2 =pd.read_csv(r'C:\Users\ASUS\Desktop\CAPSTONE PROJECT\Group2 capstone\demographics.txt', sep='\t')
df2


# In[6]:


df['ID'] = df['ID'].astype(str)
df1['ID'] = df1['ID'].astype(str)
df2['ID'] = df2['ID'].astype(str)


# In[7]:


df_data = pd.merge(df2, df1, on='ID')
df_data = pd.merge(df_data, df, on='ID')
df_data


# In[8]:


df_data.rename(columns ={" Income ":"Income"})
df_data.columns = df_data.columns.str.strip()
df_data['Income'] = df_data['Income'].str.replace('$', '').str.replace(',', '')
df_data


# In[9]:


df_data['Income'] = df_data['Income'].astype(float)

# Total Children
df_data['Total_Children'] = df_data['Kidhome'] + df_data['Teenhome']

# Remove Kidhome & Teenhome columns

df_data.drop(columns=['Kidhome', 'Teenhome'], inplace=True)

df_data.columns


# In[10]:


df_data.isna().sum()


# In[11]:


# Missing Value Imputation

df_data['Income'] =df_data['Income'].fillna(df_data['Income'].median())


# In[12]:


##  Age calculation from Birth_Year

def calculate_age(birth_year):
    current_year = datetime.now().year
    age = current_year - birth_year
    return age
df_data['Age'] = df_data['Year_Birth'].apply(calculate_age)
df_data


# In[13]:


# Calculate customer tenure
current_date = pd.Timestamp.now()
df_data['Dt_Customer'] = pd.to_datetime(df_data['Dt_Customer'], errors='coerce')
df_data['Tenure_days'] = (current_date - df_data['Dt_Customer']).dt.days
df_data['Tenure'] = df_data['Tenure_days'] / 30.44

df_data


# In[14]:


## Calculation of total amount spent by each customer in all products.

amount_columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']

df_data['Total_Spent'] = df_data[amount_columns].sum(axis=1)

df_data.drop(columns =amount_columns,inplace=True)

df_data


# In[15]:


df_data[df_data.duplicated()]


# In[16]:


# Columns to drop
# Columns to drop
columns_to_drop = ['Recency','ID', 'Year_Birth','Complain','Dt_Customer','Tenure_days','Country','Education']

# Drop the columns
df_data.drop(columns=columns_to_drop, inplace=True)

df_data.columns


# In[17]:


list_cat_var =df_data.select_dtypes(include=["object"]).columns
print(list_cat_var)


# In[18]:


from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()
for i in list_cat_var:
    df_data[i]=le.fit_transform(df_data[i])


# In[19]:


df_data


# In[20]:


y = df_data["Response"]
X = df_data.drop("Response",axis =1)


# In[21]:


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


# Convert categorical features to dummy variables
X = pd.get_dummies(X, drop_first=True)

# Split data into training and testing sets before resampling
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)


# In[22]:


y_train.value_counts()


# In[23]:


y_train_res.value_counts()


# In[24]:


X_train,x_test,y_train,y_test=train_test_split(X_train_res,y_train_res,test_size=0.2,random_state=10)
print(len(X_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))


# # MODEL

# # XG BOOST CLASSIFIER

# In[ ]:





# In[25]:


y = df_data["Response"]
X = df_data.drop("Response",axis =1)


# In[26]:


import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix


# In[27]:


# Create XGBoost classifier
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Define hyperparameters for tuning
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Perform grid search
grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, 
                           cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(x_test)

# Print classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[28]:


# Make pickel file for model
pickle.dump(best_model,open("model.pkl","wb"))


# In[29]:


# path of pkl file
os.getcwd()


# In[30]:


from sklearn.metrics import classification_report


# In[31]:


print(classification_report(y_test,y_pred))


# In[32]:


x_test


# In[ ]:




