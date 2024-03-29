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
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score


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


def ExecuteKNN(data,cvTeste=10):
    print("Starting KNN...")
    le = LabelEncoder()
    data = data.progress_apply(le.fit_transform)
    #data['OFFENSE_WEIGH'] = data.progress_apply(lambda row: row.OFFENSE_WEIGH/100,axis=1 )

    #print(data)
    x_columns=['MONTH_REPORTED','WEEKDAY_REPORTED','HOUR_REPORTED','NEIGHBORHOOD_ID','OFFENSE_WEIGH','COUNT']
    y_columns=['SAFETY']

    x_train, x_test = train_test_split(data[x_columns], test_size=0.3)
    y_train, y_test = train_test_split(data[y_columns], test_size=0.3)
    model = KNeighborsClassifier(n_neighbors=4)
    model.fit(x_train, y_train.values.ravel())
    predictions = model.predict(x_test)
    #print(len(predictions))
    #print(len(y_test))
    # Get the actua__________________________________l values for the test set.
    actual = y_test
    # Compute the mean squared error of our predictions.
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))



    print("=================================RESULTS==================================")
    cross_val = cross_val_score(model,data[x_columns],data[y_columns].values.ravel(), cv=cvTeste)
    print("CROSS VAL -->{}".format(np.mean(cross_val)))
    accuracy_value = accuracy_score(y_test, predictions)
    print("Accuracy -->", accuracy_value)
    print("Precision -->", precision_score(y_test, predictions))
    print("F1 -->", f1_score(y_test, predictions))
    print("Recall -->", recall_score(y_test, predictions))
    print("==============================================================================""\n""\n""\n\"")

    return np.mean(cross_val)

def ExecuteDecisionTree(data,cvTeste=10):
    print("Starting Decision Tree...")
    le = LabelEncoder()
    data = data.progress_apply(le.fit_transform)
    #data['OFFENSE_WEIGH'] = data.progress_apply(lambda row: row.OFFENSE_WEIGH / 100, axis=1)
    x_columns = ['MONTH_REPORTED', 'WEEKDAY_REPORTED', 'HOUR_REPORTED', 'NEIGHBORHOOD_ID', 'OFFENSE_WEIGH', 'COUNT']
    y_columns = ['SAFETY']

    x_train, x_test = train_test_split(data[x_columns], test_size=0.3)
    y_train, y_test = train_test_split(data[y_columns].values.ravel(), test_size=0.3)

    model = DecisionTreeClassifier()
    model.fit(x_train, np.ravel(y_train,order='C'))
    predictions = model.predict(x_test)
    rfc_cv_score = cross_val_score(model,x_train,y_train, cv=cvTeste)
    accuracy_value = accuracy_score(y_test, predictions)
    cross_val = cross_val_score(model,data[x_columns],data[y_columns].values.ravel(), cv=10)
    print("=================================RESULTS==================================")
    cross_val = cross_val_score(model, data[x_columns], data[y_columns].values.ravel(), cv=cvTeste)
    print("CROSS VAL -->{}".format(np.mean(cross_val)))
    accuracy_value = accuracy_score(y_test, predictions)
    print("Accuracy -->", accuracy_value)
    print("Precision -->", precision_score(y_test, predictions))
    print("F1 -->", f1_score(y_test, predictions))
    print("Recall -->", recall_score(y_test, predictions))
    print("==============================================================================""\n""\n""\n\"")
    return np.mean(cross_val)

def equal(pre1,pre2):
    if(len(pre1)!=len(pre2)):
        return False
    for i in range(len(pre1)):
        if(pre1[i]!=pre2[i]):
            return False
    return True

def vote(predictions):
    votes=[0]*len(predictions)
    for i in range(len(predictions)):
        for j in range(len(predictions)):
            if(i==j):
                continue
            if(equal(predictions[i],predictions[j])):
                votes[i]+=1
    return predictions[votes.index(max(votes))] ,votes.index(max(votes))

def bootstrap_sample(original_dataset, m):
    x_columns = ['MONTH_REPORTED', 'WEEKDAY_REPORTED', 'HOUR_REPORTED', 'NEIGHBORHOOD_ID', 'OFFENSE_WEIGH', 'COUNT']
    y_columns = ['SAFETY']
    sub_dataset_x = []
    sub_dataset_y = []
    for i in range(m):
        nrand =random.randint(0,len(original_dataset))
        sub_dataset_x.append(
            original_dataset[x_columns].iloc[nrand,:]
        )
        sub_dataset_y.append(
            original_dataset[y_columns].iloc[nrand,:]
        )
    return sub_dataset_x,sub_dataset_y


def bagging( n, m, train_dataset, target, test_dataset):
    print("Starting BAGGING...")
    predictions = [[0 for x in range(len(target))] for y in range(n)]
    y_test_vect = [[0 for x in range(len(target))] for y in range(n)]
    for i in range(n):
        algos = [KNeighborsClassifier(), LogisticRegression()]
        sub_dataset_x,sub_dataset_y = bootstrap_sample(train_dataset, m)
        x_train,x_test,y_train, y_test = train_test_split(sub_dataset_x,sub_dataset_y)
        model = algos[random.randint(0,1)]

        model.fit(x_train,np.ravel(y_train,order='C'))
        y_test_vect[i]=y_test
        predictions[i] =model.predict(x_test)

    final_predictions,ind = vote(predictions) # for classification
    accuracy_value = accuracy_score(y_test_vect[ind], final_predictions)
    print("=================================RESULTS==================================")
    print("accuracy_value Final -->",accuracy_value)
    print("F1 -->" ,f1_score(y_test_vect[ind], final_predictions))
    print("Precision -->" ,precision_score(y_test_vect[ind], final_predictions))
    print("Recall -->", recall_score(y_test_vect[ind], final_predictions))
    print("==============================================================================""\n""\n""\n\"")
    return final_predictions


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
    data.to_csv('crime-treated.csv',index=None)
    print("arquivos salvo")

def CratGraph(data):
    print("Creating graph...")
    #file = open("graphs.csv","w")
    #lines=[]
    #lines.append("KNN")
    #lines.append("N,ACCURACY")
    N=[]
    accurs=[]
    accursEXD=[]
    
    for i in range(1,11):
        print("runnig KNN  with n = ",i*10)
        accur=ExecuteKNN(data,10*i)
        N.append(i*10)
        accurs.append(accur)
    plt.plot(N,accurs,label="KNN")
    plt.xlabel("N")
    plt.ylabel("cross Validation")
    for i in range(1,11):
        print("runnig ExecuteDecisionTree with n = ",i*10)
        accur=ExecuteDecisionTree(data,10*i)
        accursEXD.append(accur)
    plt.plot(N,accursEXD,label="ExecuteDecisionTree")
    plt.legend()
    plt.savefig("graph.png")
        #lines.append(str(i*10)+","+str(accur))
    #file.writelines(lines)
    #file.close()

def main():
    print("Loading Data...")
    names = ["OFFENSE_TYPE_ID", "OFFENSE_CATEGORY_ID", "FIRST_OCCURRENCE_DATE",
            "REPORTED_DATE", "DISTRICT_ID", "PRECINCT_ID",
            "NEIGHBORHOOD_ID", "IS_CRIME", "IS_TRAFFIC"]
    filePath="crime.csv"
    pd.set_option('display.float_format', '{:.2f}'.format)
    fileExist=False
    if(os.path.exists('crime-treated.csv')):
        fileExist =True
        filePath="crime-treated.csv"
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

    print("CRIME MODE: ",modeCrime)
    dataClenad['SAFETY'] = dataClenad.progress_apply(lambda row: 1 if row.OFFENSE_WEIGH <= modeCrime and row.COUNT <= modeQtd else 0, axis=1 )

    counts = dataClenad['SAFETY'].value_counts()

    # CratGraph(dataClenad)
    ExecuteKNN(dataClenad)
    ExecuteDecisionTree(dataClenad)

    le = LabelEncoder()
    dataClenad = dataClenad.progress_apply(le.fit_transform)
    x_columns = ['MONTH_REPORTED', 'WEEKDAY_REPORTED', 'HOUR_REPORTED', 'NEIGHBORHOOD_ID', 'OFFENSE_WEIGH', 'COUNT']
    y_columns = ['SAFETY']
    x_train, x_test = train_test_split(dataClenad[x_columns], test_size=0.3)
    y_train, y_test = train_test_split(dataClenad[y_columns].values.ravel(), test_size=0.3)

    bagging(5, 200, dataClenad, np.ravel(y_train, order='C'), x_train)
    exit(0)


main()
