import csv
import random
import math
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from IPython.display import display
from tqdm import tqdm, tqdm_pandas
import os.path
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def TesteManual(yTest,yPrediction):
    print(type(yTest))
    total = len(yTest)
    if (type(yTest) == type(yPrediction)):
        y_test_mat=yTest
    else:
        y_test_mat=yTest.values.ravel()

    right =0
    wrong = 0
    zeroPred =0
    onePred =0
    zeroTest =0
    oneTest =0
    for i in range(total):
        if(y_test_mat[i]==yPrediction[i]):
            right+=1
        else:
            wrong +=1
        if yPrediction[i] ==0:
            zeroPred+=1
        else:
            onePred+=1
        if y_test_mat[i] ==0:
            zeroTest+=1
        else:
            oneTest+=1
    print("right ->", right)
    print("porcent of total->", right/total)
    print("=====================")
    print("wrong ->", wrong)
    print("porcent of wrong->", wrong/total)
    print("=====================")
    print("oneTest ->", oneTest)
    print("onePred ->",onePred)
    print("=====================")
    print("zeroTest ->", zeroTest)
    print("zeroPred ->", zeroPred)


def ExecuteKNN(data):
    print("Starting KNN...")
    le = LabelEncoder()
    data = data.progress_apply(le.fit_transform)
    data['OFFENSE_WEIGH'] = data.progress_apply(lambda row: row.OFFENSE_WEIGH/100,axis=1 )

    print(data)
    x_columns=['MONTH_REPORTED','WEEKDAY_REPORTED','HOUR_REPORTED','NEIGHBORHOOD_ID','OFFENSE_WEIGH','COUNT']
    y_columns=['SAFETY']

    x_train, x_test = train_test_split(data[x_columns], test_size=0.3)
    y_train, y_test = train_test_split(data[y_columns], test_size=0.3)
    print(data[x_columns])
    
    model = KNeighborsClassifier(n_neighbors=4)
    model.fit(x_train, y_train.values.ravel())
    predictions = model.predict(x_test)
    print(len(predictions))
    print(len(y_test))
    # Get the actua__________________________________l values for the test set.
    actual = y_test
    # Compute the mean squared error of our predictions.
    print(confusion_matrix(y_test, predictions))  
    print(classification_report(y_test, predictions))



    print("__________________________________ CROSSSSSSSSSSS VALLLLLLLLLLLL ________________________________")
    cross_val = cross_val_score(model,data[x_columns],data[y_columns].values.ravel(), cv=10)
    print("{}".format(np.mean(cross_val)))

    print("__________________________________ CROSSSSSSSSSSS VALLLLLLLLLLLL ________________________________")


    accuracy_value = accuracy_score(y_test, predictions)
    print(accuracy_value)
    TesteManual(y_test,predictions)

def ExecuteDecisionTree(data):
    print("Starting Decision Tree...")
    le = LabelEncoder()
    data = data.progress_apply(le.fit_transform)
    data['OFFENSE_WEIGH'] = data.progress_apply(lambda row: row.OFFENSE_WEIGH / 100, axis=1)
    x_columns = ['MONTH_REPORTED', 'WEEKDAY_REPORTED', 'HOUR_REPORTED', 'NEIGHBORHOOD_ID', 'OFFENSE_WEIGH', 'COUNT']
    y_columns = ['SAFETY']

    x_train, x_test = train_test_split(data[x_columns], test_size=0.3)
    y_train, y_test = train_test_split(data[y_columns].values.ravel(), test_size=0.3)

    model = DecisionTreeClassifier()
    model.fit(x_train, np.ravel(y_train,order='C'))
    predictions = model.predict(x_test)
    rfc_cv_score = cross_val_score(model,x_train,y_train, cv=5)
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, predictions))
    print('\n')
    print("=== Classification Report ===")
    print(classification_report(y_test, predictions))
    print('\n')
    print("=== All AUC Scores ===")
    print(rfc_cv_score)
    print('\n')
    print("=== Mean AUC Score ===")
    print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())
    accuracy_value = accuracy_score(y_test, predictions)
    print(accuracy_value)
    print("__________________________________ CROSSSSSSSSSSS VALLLLLLLLLLLL ________________________________")
    cross_val = cross_val_score(model,data[x_columns],data[y_columns].values.ravel(), cv=10)
    print("{}".format(np.mean(cross_val)))
    print("__________________________________ CROSSSSSSSSSSS VALLLLLLLLLLLL ________________________________")

    TesteManual(y_test,predictions)


def ExecuteRandomForest(data):
    print("Starting Random Forest...")
    le = LabelEncoder()
    data = data.progress_apply(le.fit_transform)
    data['OFFENSE_WEIGH'] = data.progress_apply(lambda row: row.OFFENSE_WEIGH / 100, axis=1)
    x_columns = ['MONTH_REPORTED', 'WEEKDAY_REPORTED', 'HOUR_REPORTED', 'NEIGHBORHOOD_ID', 'OFFENSE_WEIGH', 'COUNT']
    y_columns = ['SAFETY']

    x_train, x_test = train_test_split(data[x_columns], test_size=0.3)
    y_train, y_test = train_test_split(data[y_columns].values.ravel(), test_size=0.3)

    from sklearn import model_selection
    # random forest model creation
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    # predictions
    rfc_predict = rfc.predict(x_test)

    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    rfc_cv_score = cross_val_score(rfc,x_train,y_train, cv=5)
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, rfc_predict))
    print('\n')
    print("=== Classification Report ===")
    print(classification_report(y_test, rfc_predict))
    print('\n')
    print("=== All AUC Scores ===")
    print(rfc_cv_score)
    print('\n')
    print("=== Mean AUC Score ===")
    print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())
    accuracy_value = accuracy_score(y_test, rfc_predict)
    print(accuracy_value)

    print("__________________________________ CROSSSSSSSSSSS VALLLLLLLLLLLL ________________________________")
    cross_val = cross_val_score(rfc,data[x_columns],data[y_columns].values.ravel(), cv=10)
    print("{}".format(np.mean(cross_val)))
    print("__________________________________ CROSSSSSSSSSSS VALLLLLLLLLLLL ________________________________")

    TesteManual(y_test,rfc_predict)

def treatData(data):
    print("Treating Data")

    print("Filtering For Crimes:")
    data=data[data['IS_CRIME']==1]
    print("COMPLETE Filtering For Crimes: ")

    print("Getting Hour:")
    data['HOUR_REPORTED']=pd.DatetimeIndex(data['REPORTED_DATE']).hour
    print("COMPLETE Hour: ")

    print("Getting Weekday:")
    data['WEEKDAY_REPORTED']=pd.DatetimeIndex(data['REPORTED_DATE']).weekday
    print("COMPLETE Getting Weekday: ")
    
    print("Getting Day:")
    data['DAY_REPORTED']=pd.DatetimeIndex(data['REPORTED_DATE']).day
    print("COMPLETE Getting Day: ")
    
    print("Getting Month:")
    data['MONTH_REPORTED']=pd.DatetimeIndex(data['REPORTED_DATE']).month
    print("COMPLETE Getting Month: ")
    
    print("Getting Year:")
    data['YEAR_REPORTED']=pd.DatetimeIndex(data['REPORTED_DATE']).year
    print("COMPLETE Getting YEAR: ")
    
    print("Getting OFENSE_CODE_JUNCTION:")
    data['OFENSE_CODE_JUNCTION'] = data[['OFFENSE_CODE','OFFENSE_CODE_EXTENSION']].dot([100,1])
    print("COMPLETE OFENSE_CODE_JUNCTION: ")

    print("Saving File..")
    data.to_csv('db/crime-treated.csv',index=None)
    print("arquivos salvo")
    
def main():
    print("Loading Data...")
    names = ["OFFENSE_TYPE_ID", "OFFENSE_CATEGORY_ID", "FIRST_OCCURRENCE_DATE",
            "REPORTED_DATE", "DISTRICT_ID", "PRECINCT_ID",
            "NEIGHBORHOOD_ID", "IS_CRIME", "IS_TRAFFIC"]
    filePath="db/crime.csv"
    pd.set_option('display.float_format', '{:.2f}'.format)
    fileExist=False
    if(os.path.exists('db/crime-treated.csv')):
        fileExist =True
        filePath="db/crime-treated.csv"  
        names =["HOUR_REPORTED","DAY_REPORTED","WEEKDAY_REPORTED","MONTH_REPORTED","YEAR_REPORTED",
                "OFFENSE_CATEGORY_ID","NEIGHBORHOOD_ID"]
        data=pd.read_csv(filePath, parse_dates=True,usecols=names,nrows = None)
    else:
        data=pd.read_csv(filePath, parse_dates=True)

    print("======================================================================DATA INFO======================================")
    data.info()
    print("===================================DATA INFO======================================")
    if(not fileExist):
        print(data.head(5))
        display(data.groupby([data.OFFENSE_CODE,data.OFFENSE_CODE_EXTENSION,data.OFFENSE_TYPE_ID]).size())
        temp=display(data.groupby([data.INCIDENT_ID,data.OFFENSE_CODE,data.OFFENSE_CODE_EXTENSION,data.OFFENSE_TYPE_ID]).size())
        print(temp)

        treatData(data)
    crimesDict = {
        'all-other-crimes': 1,
        'larceny' : 1,
        'theft-from-motor-vehicle' : 3,
        'drug-alcohol' : 2,
        'auto-theft' : 3,
        'white-collar-crime': 1,
        'burglary': 2,
        'public-disorder' : 2,
        'aggravated-assault': 3,
        'other-crimes-against-persons' : 2,
        'robbery' : 3,
        'sexual-assault' : 3,
        'murder': 3,
        'arson': 2
    }
    tqdm.pandas()
    print("Calculating Offense Weigh...")
    data['OFFENSE_WEIGH'] = data.progress_apply(lambda row:  crimesDict[row.OFFENSE_CATEGORY_ID], axis=1 )
    dataCount = data.groupby(['MONTH_REPORTED','WEEKDAY_REPORTED','HOUR_REPORTED','NEIGHBORHOOD_ID']).MONTH_REPORTED.agg('count').to_frame('COUNT').reset_index()
    dataClenad =data.groupby(['MONTH_REPORTED','WEEKDAY_REPORTED','HOUR_REPORTED','NEIGHBORHOOD_ID'], as_index=False).agg({'OFFENSE_WEIGH':'sum'})#['OFFENSE_WEIGH'].sum()['GEO_X'].mean()
    dataClenad['COUNT'] = dataCount['COUNT']
    print(dataClenad)
    print("Calculating Safety ...")
    medianCrime = dataClenad['OFFENSE_WEIGH'].median()
    modeCrime = dataClenad['OFFENSE_WEIGH'].mode().values
    modeQtd = dataClenad['COUNT'].mode().values

    print(modeCrime)
    dataClenad['SAFETY'] = dataClenad.progress_apply(lambda row: 1 if row.OFFENSE_WEIGH <= modeCrime and row.COUNT <= modeQtd else 0, axis=1 )

    counts = dataClenad['SAFETY'].value_counts()
    print(counts)
    ExecuteKNN(dataClenad)
    ExecuteDecisionTree(dataClenad)
    ExecuteRandomForest(dataClenad)

main()
