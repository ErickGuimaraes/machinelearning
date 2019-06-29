import csv
import random
import math
import operator
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from IPython.display import display
from tqdm import tqdm, tqdm_pandas

def main():
    print("Loading Data...")
    names = ["INCIDENT_ID", "OFFENSE_ID", "OFFENSE_CODE", "OFFENSE_CODE_EXTENSION",
            "OFFENSE_TYPE_ID", "OFFENSE_CATEGORY_ID", "FIRST_OCCURRENCE_DATE",
            "REPORTED_DATE", "DISTRICT_ID", "PRECINCT_ID",
            "NEIGHBORHOOD_ID", "IS_CRIME", "IS_TRAFFIC"]
    data=pd.read_csv('db/crime.csv', parse_dates=True)
    print("===================================DATA INFO======================================")
    data.info()
    print("===================================DATA INFO======================================")
    print(data.head(5))
    #display(data.groupby([data.OFFENSE_CODE,data.OFFENSE_CODE_EXTENSION,data.OFFENSE_TYPE_ID]).size())
    temp=display(data.groupby([data.INCIDENT_ID,data.OFFENSE_CODE,data.OFFENSE_CODE_EXTENSION,data.OFFENSE_TYPE_ID]).size())
    print(temp)

    data['OFENSE_CODE_JUNCTION'] = data[['OFFENSE_CODE','OFFENSE_CODE_EXTENSION']].dot([100,1])
    print("data junction he ==========================")
    print(data['OFENSE_CODE_JUNCTION'])
    #filter out traffic accidents from the crime dataset
    data=data[data['IS_CRIME']==1]
    data['HOUR_REPORTED']=pd.DatetimeIndex(data['REPORTED_DATE']).hour
    data['WEEKDAY_REPORTED']=pd.DatetimeIndex(data['REPORTED_DATE']).weekday
    data['MONTH_REPORTED']=pd.DatetimeIndex(data['REPORTED_DATE']).month
    data['YEAR_REPORTED']=pd.DatetimeIndex(data['REPORTED_DATE']).year

    #data['OFFENSE_TYPE_CODE']= pd.Int64Index((data['OFFENSE_CODE'].astype(str)+data['OFFENSE_CODE_EXTENSION'].astype(str)).astype(64))
    #pd.set_option('display.max_rows',500)
    print(data)

main()
