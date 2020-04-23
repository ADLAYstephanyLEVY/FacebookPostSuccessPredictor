import pickle
import pandas as pd
import numpy as np
#importtensorflowastf 
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#from sklearn.externals import joblib
#read loaded file
data = pd.read_csv('Dataset-Facebook.csv', ',')
#matrices
arrX = data[data.columns[:-1]].values
arrY = data[data.columns[-1]].values
X_train, X_test, Y_train, Y_test = train_test_split(arrX, arrY, test_size=0.2, random_state=None)
#################### Logistic Regression ########################
model = LogisticRegression()
model.fit(X_train, Y_train)
#save the model .sav
filename = 'model.sav'
#pickle.dump(model, open(filename, 'wb'))


#pickle.dump(logReg, open('model.pkl','wb'))
#model = pickle.load(open('model.pkl','rb'))
#score2 = logReg.score(X_train,Y_train)
#logReg_predict = logReg.predict(X_test)
#print("Logistic Regression Accuracy:",metrics.accuracy_score(Y_test, logReg_predict))

