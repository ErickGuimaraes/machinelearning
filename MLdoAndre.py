import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
import numpy as np
import folium
from folium import plugins
from sklearn.model_selection import train_test_split


def handle_non_numeric(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
        if column == "REPORTED_DATE":
            df[column] = df[column].dt.hour

        elif df[column].dtype != np.int64 and df[column].dtype != np.float64 and column != "REPORTED_DATE":
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int,df[column]))

    return df

def main():

    names = ["OFFENSE_CODE", "REPORTED_DATE",
         "DISTRICT_ID", "IS_CRIME" ]


    data=pd.read_csv('crime.csv', parse_dates=True,usecols=names,nrows = 100)
    data = data.dropna()
    data = data[data['IS_CRIME'] == 1]
    data = data.drop(labels = "IS_CRIME", axis = 1)
    data['REPORTED_DATE'] = pd.to_datetime(data.REPORTED_DATE)
    print(data.head(10))

    data = handle_non_numeric(data)
    print(data.head(10))
    features_name = ['OFFENSE_CODE','DISTRICT_ID']
    X = data[features_name]

    y = data['REPORTED_DATE']

    # create new a knn model
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    print('Accuracy of Logistic regression classifier on training set: {:.2f}'
          .format(logreg.score(X_train, y_train)))
    print('Accuracy of Logistic regression classifier on test set: {:.2f}'
          .format(logreg.score(X_test, y_test)))





main()
