import pandas as pd
import numpy as np
import datetime
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

df = pd.read_csv('SJ.TO.csv')
df['Volatility'] = (df['High'] - df['Low']) / df['Close'] * 100
df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100

df = df[['Date', 'Close', 'Volatility', 'PCT_change', 'Volume']]

forecast_col = 'Close'
df.fillna(-99999, inplace=True)
forecast_out = 30
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label', 'Date'], axis=1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

y = np.array(df['label'])[:-forecast_out]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=10)
clf.fit(X_train, y_train)
# with open("linear_regression.pickle", 'wb') as f:
#     pickle.dump(clf,f)

pickle_in = open("linear_regression.pickle", 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan
df['Forecast'] = np.concatenate((np.array(df['Forecast'])[:-forecast_out], forecast_set), axis=0)

df['label'].plot()
df['Forecast'].plot()
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
