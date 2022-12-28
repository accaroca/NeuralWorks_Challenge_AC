import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import json
import sqlite3
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

#Reads and construct the dataframe from the last version of trips.csv
def data_input():
    df=pd.read_csv("trips.csv")
    #Replace patterns to obtain only the coordinates
    df=df.replace("POINT ", "", regex=True)
    df=df.replace('', '_', regex=True)
    df['origin_coord']=df['origin_coord'].str.replace(r"(","")
    df['origin_coord']=df['origin_coord'].str.replace(r")","")
    df['destination_coord']=df['destination_coord'].str.replace(r"(","")
    df['destination_coord']=df['destination_coord'].str.replace(r")","")
    origins=pd.DataFrame(columns=['x','y'])
    dest=pd.DataFrame(columns=['x','y'])
    df_origins = df.origin_coord.str.split(' ', expand=True)
    df_dest = df.destination_coord.str.split(' ', expand=True)
    origins["x"]=df_origins.iloc[:,0]
    origins["y"]=df_origins.iloc[:,1]
    dest["x"]=df_dest.iloc[:,0]
    dest["y"]=df_dest.iloc[:,1]
    #df_origins=pd.to_numeric(df_origins)
    #df_dest=pd.to_numeric(df_dest)
    origins["x"]=pd.to_numeric(origins["x"])
    origins["y"]=pd.to_numeric(origins["y"])
    dest["x"]=pd.to_numeric(dest["x"])
    dest["y"]=pd.to_numeric(dest["y"])
    df['Euclidian_Distance']=np.sqrt((origins.x-dest.x)**2+(origins.y-dest.y)**2)
    df["origin_x"]=origins.x
    df["origin_y"]=origins.y
    df["destination_x"]=dest.x
    df["destination_y"]=dest.y
    df=df.drop(['origin_coord','destination_coord'],axis=1)
    df['Hour_Of_Day']=pd.to_datetime(df['datetime'])
    df['Hour_Of_Day']=df['Hour_Of_Day'].dt.day
    df['Week_Of_Year']=pd.to_datetime(df['datetime'])
    df['Week_Of_Year']=df['Week_Of_Year'].dt.isocalendar().week
    return df
#Clustering Algorithm
def Trip_Clustering(df):
    X = np.array(df[["origin_x","origin_y","destination_x","destination_y","Hour_Of_Day","Euclidian_Distance"]])
    y = np.array(df['region'])
    Nc = range(1, 20)
    kmeans = [KMeans(n_clusters=i) for i in Nc]
    score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
    kmeans = KMeans(n_clusters=5).fit(X)
    centroids = kmeans.cluster_centers_
    # Predicting the clusters
    labels = kmeans.predict(X)
    # Getting the cluster centers
    C = kmeans.cluster_centers_
    #Labeling trips
    df['label'] = labels;
    return X,y,df

def Weekly_avg_Bounding_Box(df):
    #Now, let's define the dimensions of the bounding box
    #x Coordinate of the center
    x=float(input("Enter x coordinate of center:"))
    #y Coordinate of the center
    y=float(input("Enter y coordinate of center:"))
    #width of the bounding box
    w=float(input("Enter width of bounding box:"))
    #height of the bounding box
    h=float(input("Enter width of bounding box:"))
    #Now let's set the bounding box bounds
    left_bound=x-(w/2)
    right_bound=x+(w/2)
    upper_bound=y+(h/2)
    lower_bound=y-(h/2)
    #Now let's filter de dataframe by the bounds of the bounding box
    df = df[df['origin_x'] <= right_bound ]
    df = df[df['origin_x'] >= left_bound ]
    df = df[df['origin_y'] <= upper_bound ]
    df = df[df['origin_y'] >= lower_bound ]
    print(lower_bound,upper_bound,left_bound,right_bound)
    print(df)
    WK=df.groupby(['Week_Of_Year']).datetime.count()
    #We count by datetime stamp since is like an ID    
    WK.head()
    #We now can see the amount of trips by number of week of the year
    #Now we obtain the average for the dataframe 
    Weekly_Trips_Avg=WK.mean()
    print("The average weekly amount of trips for the bounding box is: ",Weekly_Trips_Avg, "trips")

    return Weekly_Trips_Avg

def Weekly_avg_region(df):
    WK=df.groupby(['region','Week_Of_Year']).datetime.count()  
    Weekly_Trips_Avg_by_region=WK.groupby(['region']).mean()
    #print("The average weekly amount of trips for each Region is: ",Weekly_Trips_Avg_by_region, "trips")
    print("Weekly Average trips by Region")
    print(Weekly_Trips_Avg_by_region)
    return Weekly_Trips_Avg_by_region 

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
    cfm_CBC=confusion_matrix(Y_test,Y_pred)
    print("Confusion Matrix: ")
    print(cfm_CBC)

    print("Classification report: ")

    print(classification_report(Y_test,Y_pred))

    acc_CBC=accuracy_score(Y_test, Y_pred)
    print("Accuracy of the model: ",acc_CBC)
    disp = ConfusionMatrixDisplay(confusion_matrix=cfm_CBC,display_labels=['Cluster1','Cluster2','Cluster3','Cluster4','Cluster5'])
    disp.plot()
    plt.show()
    
    return X,Y,df,acc_CBC