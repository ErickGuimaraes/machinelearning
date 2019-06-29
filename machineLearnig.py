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
    names = ["INCIDENT_ID", "OFFENSE_ID", "OFFENSE_CODE", "OFFENSE_CODE_EXTENSION",
            "OFFENSE_TYPE_ID", "OFFENSE_CATEGORY_ID", "FIRST_OCCURRENCE_DATE",
            "REPORTED_DATE", "DISTRICT_ID", "PRECINCT_ID",
            "NEIGHBORHOOD_ID", "IS_CRIME", "IS_TRAFFIC"]
    filePath="db/crime.csv"
    fileExist=False
    if(os.path.exists('db/crime-treated.csv')):
        fileExist =True
        filePath="db/crime-treated.csv"    

    data=pd.read_csv(filePath, parse_dates=True)
    print("===================================DATA INFO======================================")
    data.info()
    print("===================================DATA INFO======================================")
    print(data.head(5))
    #display(data.groupby([data.OFFENSE_CODE,data.OFFENSE_CODE_EXTENSION,data.OFFENSE_TYPE_ID]).size())
    temp=display(data.groupby([data.INCIDENT_ID,data.OFFENSE_CODE,data.OFFENSE_CODE_EXTENSION,data.OFFENSE_TYPE_ID]).size())
    print(temp)
    if(not fileExist):
        treatData(data)
    exit(0)
    

main()
