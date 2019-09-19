
import pandas as pd
import sklearn
from matplotlib import pyplot as plt

df=pd.read_csv(r'G:\Mini Project\PythonApplication6\PythonApplication6\PatientDataEdited.csv')

from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(df[['Age']],df.Status,train_size=0.65)
X_train, X_test, y_train, y_test = train_test_split(df[['TimeSpentSec']],df.Status,train_size=0.65)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


model.fit(X_train, y_train)

y_predicted = model.predict(X_test)
#y_predicted1 = model.predict([20,21])

x=model.predict_proba(X_test)

z=model.score(X_test,y_test)
print(X_test)
print(y_predicted)
#print(y_predicted1)
print(x)
print(z)