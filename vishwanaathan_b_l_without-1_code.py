#!/usr/bin/env python
# coding: utf-8

# # Supervised Learning

# In[107]:


import csv
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn


# In[108]:


import warnings
warnings.filterwarnings('ignore')


# In[109]:


'''''
# Half Moons Data  Generation
from sklearn import datasets

np.random.seed(10**7)
data, labels = datasets.make_moons(n_samples=5000, noise=0.2)

print(data)
print(labels)

np.savetxt("data.csv", data, delimiter=',', fmt="%f")
np.savetxt("labels.csv", labels, fmt="%i")
'''''


# In[110]:


# reading data as dataframes
data = pd.read_csv('illegal_fishing_trn_data.csv', header= None)
data.rename(columns={0:'mmsi',1:'timestamp',2:'distance_from_shore',3:'distance_from_port',4:'speed',5:'course',6:'lat',7:'lon'}, inplace= True)

label = pd.read_csv('illegal_fishing_trn_class_labels.csv', header= None)
label.rename(columns ={0:"A", 1:"B"}, inplace= True)


# In[111]:


# Separate out the rows with -1 in any column
df_with_minus_111 = label[label.B==-1].dropna(how='all')

# Separate out the rows without -1 in any column
df_without_minus_111 = label[label.B!=-1].dropna(how='all')


# In[112]:


df_without_minus_111


# In[113]:


df212_clean = data.drop(index=df_with_minus_111.index)

print(df212_clean)


# In[115]:


import seaborn as sns
sns.countplot(df_without_minus_111.B)


# In[116]:


data


# In[117]:


# visualising datasets
print(df212_clean .head())
print(df_without_minus_111.head())
print( )
    
print(df212_clean .shape)
print(df_without_minus_111.shape)
print( )

print(df212_clean .info())
print(df_without_minus_111.info())
print( )

print(df212_clean .nunique())
print(df_without_minus_111.nunique())


# In[118]:


missing_rows = df212_clean.loc[df212_clean.isnull().any(axis=1)].index.tolist()
print(missing_rows)


# In[119]:


df212_clean = df212_clean.drop(missing_rows)
df_without_minus_111 = df_without_minus_111.drop(missing_rows)


# In[120]:


from sklearn.model_selection import train_test_split

# split data into training and remaining data
train_data, X_rem, train_label, y_rem = train_test_split(df212_clean, df_without_minus_111.B, test_size=0.4, random_state=42)

# split remaining data into validation and testing data
valid_data, test_data, valid_label, test_label = train_test_split(X_rem, y_rem, test_size=0.5, random_state=42)


# In[121]:


print(train_data)


# In[68]:


# Preprocessing the data
from sklearn.preprocessing import MinMaxScaler

# Scaling the Data
scaler = MinMaxScaler(feature_range=(0, 1), copy=True, clip=False)
train_data = scaler.fit_transform(train_data)


# In[122]:


train_data[train_data < 0] = 0
valid_data[valid_data < 0] = 0


# In[123]:


train_data


# In[124]:


print(train_data.shape)

print(test_data.shape)

print(train_label.shape)

print(test_label.shape)


# In[125]:


from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model


# In[126]:


from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2


# In[127]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report


# In[128]:


# models to try 

# raw_data (everything month, day)
# data_with_encodings (raw_data but one-hot encoding and standard scaled encoding)
# data_with_encodings with polynomial features (data_with_encoding and polynomial features from sklearn)

# logistic regression : data_with_encodings
# logitstic regression with polynomial feature from sklearn : data_with_encodings with polynomial features

# svm with linear kernel : data_with_encodings
# svm with polynomial kernel : data_with_encodings
# svm with rbf kernel : data_with_encodings

# decision trees : raw_data

# knn : data_with_encodings

# naive bayes classifier : data_with_encodings

# random forest : raw_data

# bagging classifier : raw_data

# gradient boosted decision trees: Adaboost : raw_data


# In[129]:


# feature selection

# chi2
# mutual_info_


# In[130]:


# cross validation

# ultimate_train ultimate_test
# ultimate_train
#   train test (random split stratify)
#   train
#      0 1 2 3 4 (KFOLD Cross validation)

# create a grid of parameters (dictionry)
# grid search


# ## Classification with hyperparameter tuning

# In[131]:


train_label


# In[132]:


# pipeline parameter grid    
parameters=[
    {
        'clf': LogisticRegression(),
        'clf__penalty': ['l2', 'none'],
        'clf__multi_class': ['ovr', 'multinomial'],
        'clf__C': [0.01,0.01,1,],

    },
   
    {
        'clf': RandomForestClassifier(),
        'clf__criterion': ['gini', 'entropy'],
        'clf__n_estimators': [10,20,50,100],
        'clf__max_depth': [10, 20,30],
    },
    
    {
        'clf': GaussianNB(),
        'clf__var_smoothing':[0.000000000001, 0.000000001, 0.0000001],
    },
    
    {   'clf': SVC(),
        'clf__max_iter': [1, 10, 50],
        'clf__C': [0.1, 1, 10],
        'clf__kernel': ['linear', 'sigmoid','rbf',],
    },
    
    {
        'clf': KNeighborsClassifier(),
        'clf__n_neighbors': list(range(1, 100, 1)),
        'clf__metric': ['euclidean', 'manhattan', 'minkowski']
    },
    
    {
    'clf': DecisionTreeClassifier(),
    'clf__criterion': ['gini', 'entropy'],
    'clf__max_depth': [10, 20, 30],
    },
    
    {
    'clf': BaggingClassifier(),
    'clf__n_estimators': [10, 20, 50],
    'clf__max_samples': [0.5, 0.7, 1.0],
    }
    
    
]


# In[ ]:





# In[ ]:





# In[133]:


# storing the best hyperparameters for all the combinations
results = []

for model in parameters: 
    clf = model.pop('clf')
    print(f"\nStarted {str(clf)}")
    print("-------------------------------------------")
    pipeline = Pipeline([
    ("select", SelectKBest(score_func = chi2, k=2)),
    ("clf", clf)])
    print("\nStarted GridSearchCV")
    print("-------------------------------------------")
    grid_model = GridSearchCV(pipeline, model, verbose=2, cv=5
                                  , scoring='f1_macro', error_score='raise')
    grid_model.fit(valid_data, valid_label)
    print("Done")
    print(f"Training Score: {grid_model.best_score_}")
    print(f"Parameters: {grid_model.best_params_}") 
    print(f"Best Classifier: {grid_model.best_estimator_}")        
    results.append({
            'Model': clf,
            'Best_Score': grid_model.best_score_,
            'Best_Params': grid_model.best_params_
    })

print(results)  

# a = [0,1]
# b = [2,3]

# x -> x1, x2, x3

# a = 0, b =2
#    sum/mean(score(x1, x2 -> train, x3 ->test), score(x1, x3 -> train, x2 ->test), score(x3, x2 -> train, x2 ->test)) = score1
# a = 1, b =2
#    sum/mean(score(x1, x2 -> train, x3 ->test), score(x1, x3 -> train, x2 ->test), score(x3, x2 -> train, x2 ->test)) = score2
# a = 0, b =3
#    sum/mean(score(x1, x2 -> train, x3 ->test), score(x1, x3 -> train, x2 ->test), score(x3, x2 -> train, x2 ->test)) = score3
# a = 1, b =3
#    sum/mean(score(x1, x2 -> train, x3 ->test), score(x1, x3 -> train, x2 ->test), score(x3, x2 -> train, x2 ->test)) = score4
# In[134]:


# Random Forest

rf = RandomForestClassifier(criterion = 'gini', max_depth = 10, n_estimators = 50)
rf.fit(train_data, train_label) 
rf_pred_tune = rf.predict(test_data)
print("--------Random Forest-------")
print('\nConfusion Matrix:\n', confusion_matrix(test_label, rf_pred_tune))
print('\nClassification Report:\n', classification_report(test_label,rf_pred_tune))
print('\nAccuracy Score:\n', accuracy_score(test_label, rf_pred_tune))


# In[135]:


# Distance Tree

dt = DecisionTreeClassifier(criterion = 'entropy', max_depth = 10)
dt.fit(train_data, train_label) 
dt_pred_tune = rf.predict(test_data)
print("--------Random Forest-------")
print('\nConfusion Matrix:\n', confusion_matrix(test_label, dt_pred_tune))
print('\nClassification Report:\n', classification_report(test_label,dt_pred_tune))
print('\nAccuracy Score:\n', accuracy_score(test_label, dt_pred_tune))


# In[136]:


# KNN

knn = KNeighborsClassifier(metric = 'euclidean', n_neighbors = 17)
knn.fit(train_data, train_label)
knn_pred_tune = knn.predict(test_data)
print("--------KNN-------")
print('\nConfusion Matrix:\n', confusion_matrix(test_label, knn_pred_tune))
print('\nClassification Report:\n', classification_report(test_label, knn_pred_tune))
print('\nAccuracy Score:\n', accuracy_score(test_label, knn_pred_tune))


# In[137]:


# saving the best model
import pickle
with open('nb', 'wb') as picklefile:
    pickle.dump(grid_model, picklefile) 


# In[138]:


data_test = pd.read_csv('illegal_fishing_tst_data.csv', header= None)
data_test.rename(columns={0:'mmsi',1:'timestamp',2:'distance_from_shore',3:'distance_from_port',4:'speed',5:'course',6:'lat',7:'lon'}, inplace= True)


# In[139]:


missing_rows = data_test.loc[data_test.isnull().any(axis=1)].index.tolist()
print(missing_rows)
data_test = data_test.drop(missing_rows)


# In[140]:


#loading the best model
with open('nb', 'rb') as training_model:
    model1 = pickle.load(training_model)
 
# predicting the target labels of the test dataset    
test_predict2 = model1.predict(data_test)
print('\nPredicted Labels:\n', test_predict2)

# storing the target labels of the test dataset 
np.savetxt("test_labels.csv", test_predict2, delimiter = ',', fmt="%s")


# In[141]:


import seaborn as sns
sns.countplot(test_predict2)


# In[ ]:




