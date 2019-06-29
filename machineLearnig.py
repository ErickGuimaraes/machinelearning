import csv
import random
import math
import operator
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from IPython.display import display
def main():
    print("carregando dados...")
    names = ["INCIDENT_ID", "OFFENSE_ID", "OFFENSE_CODE", "OFFENSE_CODE_EXTENSION",
            "OFFENSE_TYPE_ID", "OFFENSE_CATEGORY_ID", "FIRST_OCCURRENCE_DATE",
            "LAST_OCCURRENCE_DATE", "REPORTED_DATE", "INCIDENT_ADDRESS",
            "GEO_X", "GEO_Y", "GEO_LON", "GEO_LAT", "DISTRICT_ID", "PRECINCT_ID",
            "NEIGHBORHOOD_ID", "IS_CRIME", "IS_TRAFFIC"]

#data = pd.read_csv("db/crime.csv",sep= ",",)
    data=pd.read_csv('db/crime.csv', parse_dates=True)

    data.info()

    print(data)

    display(data.groupby([data.OFFENSE_CODE,data.OFFENSE_CODE_EXTENSION,data.OFFENSE_TYPE_ID]).size())



main()
