import pandas as pd
import numpy as np
from sklearn import preprocessing, neighbors, svm
from sklearn.model_selection import train_test_split
import pickle


df = pd.read_csv('bank-additional.csv', sep=';')
df.drop(['duration'], 1, inplace=True)

mp_job = {'admin.':0, 'blue-collar':1, 'entrepreneur':2, 'housemaid':3, 'management':4, 'retired':5, 'self-employed':6, 'services':7, 'student':8, 'technician':9, 'unemployed':10, 'unknown': 999}
mp_marital = {'divorced':0 ,'married':2, 'single':1, 'unknown':999}
mp_education = {'basic.4y':0, 'basic.6y':1, 'basic.9y':2, 'high.school':3, 'illiterate':4, 'professional.course':5, 'university.degree':6, 'unknown':999}
mp_default = {'no':0, 'yes':1, 'unknown':999}
mp_housing = {'no':0, 'yes':1, 'unknown':999}
mp_loan = {'no':0, 'yes':1, 'unknown':999}
mp_contact = {'cellular':0 ,'telephone':1}
mp_month = {'jan':1 , 'feb':2 , 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
mp_day_of_week = {'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5}
mp_poutcome = {'failure':0, 'nonexistent':999, 'success':1}
mp_y = {'no':0, 'yes':1}
# mp = [mp_job, mp_marital, mp_education, mp_default, mp_housing, mp_loan, mp_contact, mp_month, mp_day_of_week, mp_poutcome, mp_y]
mp = [mp_marital, mp_education, mp_default, mp_housing, mp_loan, mp_month, mp_day_of_week, mp_poutcome, mp_y]


for i in range(len(df)):
    df.loc[i, 'job'] = mp_job[df.loc[i, 'job']]
    df.loc[i, 'marital'] = mp_marital[df.loc[i, 'marital']]
    df.loc[i, 'education'] = mp_education[df.loc[i, 'education']]
    df.loc[i, 'default'] = mp_default[df.loc[i, 'default']]
    df.loc[i, 'housing'] = mp_housing[df.loc[i, 'housing']]
    df.loc[i, 'loan'] = mp_loan[df.loc[i, 'loan']]
    df.loc[i, 'contact'] = mp_contact[df.loc[i, 'contact']]
    df.loc[i, 'month'] = mp_month[df.loc[i, 'month']]
    df.loc[i, 'day_of_week'] = mp_day_of_week[df.loc[i, 'day_of_week']]
    df.loc[i, 'poutcome'] = mp_poutcome[df.loc[i, 'poutcome']]
    df.loc[i, 'y'] = mp_y[df.loc[i, 'y']]

df.replace('unknown', 999, inplace=True)
df.drop(["job","contact","day_of_week","campaign","pdays","previous","poutcome","emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m","nr.employed"], 1, inplace=True)
X = np.array(df.drop(['y'], 1))
y = np.array(df['y'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# clf = neighbors.KNeighborsClassifier(n_neighbors=3)
clf = svm.SVC()
clf.fit(X_train, y_train)

# with open('knn.pickle', 'wb') as f:
#     pickle.dump(clf, f)
#
# pickle_in = open('knn.pickle', 'rb')
# clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measure = [
    [51,"married","basic.4y","no","no","yes","jul"],
    [28,"single","professional.course","no","yes","no","aug"]
]


def predict(clf, arr):
    for i in range(len(arr)):
        k = 0
        for j in range(len(arr[i])):
            if type(arr[i][j]) is str:
                arr[i][j] = mp[k][arr[i][j]]
                k += 1

    arr = np.array(arr)
    return clf.predict(arr)


prediction = predict(clf, example_measure)
print(prediction)