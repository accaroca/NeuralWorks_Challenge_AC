# Basic Libraries

import numpy                  as np  # linear algebra
import pandas                 as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn                as sns # data visualization
import matplotlib.pyplot      as plt # plotting library

# Machine Learning Libraries

from sklearn.preprocessing    import StandardScaler 
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn                  import metrics
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor
# Machine Learning Algorithms
from sklearn import linear_model
from sklearn.linear_model     import LogisticRegression
from sklearn.linear_model     import SGDClassifier
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.svm              import SVC
from sklearn.tree             import DecisionTreeClassifier
from sklearn.ensemble         import RandomForestClassifier
from sklearn.manifold import TSNE

from pandas_profiling import ProfileReport

import matplotlib.pyplot as plt 
import re
import datetime
from  dateutil.parser import parse
import math
import glob
import missingno as mno
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import pickle
#
import joblib
from catboost import CatBoostClassifier

def pipeline_CatBoost(df):
    df=df.drop(['datetime'],axis=1)
    region_dict=dict(Hamburg=0,Prague=1,Turin=2)
    datasource_dict=dict(baba_car=0,funny_car=1,cheap_mobile=2,pt_search_app=3,bad_diesel_vehicles=4)
    df['region']=df['region'].map(region_dict)
    df['datasource']=df['datasource'].map(datasource_dict)
    X= df.values[:,0:-1]
    Y=df.values[:,-1]
    scaler= StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    Y=Y.astype(int)
    
    X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.20, random_state=10)
    model=CatBoostClassifier(iterations=1000,
                               task_type="CPU",
                               devices='0:1')
    
    model.fit(X_train,Y_train)
    Y_pred= model.predict(X_test)
    # pickle.dump(model, open('Trained_Model.pkl', 'wb'))
    cfm_CBC=confusion_matrix(Y_test,Y_pred)
    print("Confusion Matrix: ")
    print(cfm_CBC)

    print("Classification report: ")

    print(classification_report(Y_test,Y_pred))

    acc_CBC=accuracy_score(Y_test, Y_pred)
    print("Accuracy of the model: ",acc_CBC)
    disp = ConfusionMatrixDisplay(confusion_matrix=cfm_CBC,display_labels=['Cluster1','Cluster2','Cluster3','Cluster4','Cluster5'])
    print(disp.plot())
    print(plt.show())
    
    return model
#%%
from functions import *
df=data_input()
Trip_Clustering(df)
model=pipeline_CatBoost(df)
pickle.dump(model, open('Trained_Model.pkl', 'wb'))

#%%
# Pkl_Filename = "Pickle_CatBoost_Model.pkl"  

# with open(Pkl_Filename, 'wb') as file:  
#     pickle.dump(Pkl_Filename, file)


