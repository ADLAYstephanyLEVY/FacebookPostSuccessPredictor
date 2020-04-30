#importing libraries
import os
import numpy as np
import pandas as pd
import flask
import pickle
import tensorflow 
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.externals import joblib
from decimal import *
from tensorflow import keras

app = Flask(__name__)

@app.route('/')
@app.route('/index')


def index():
    return flask.render_template('index.html')
 
def LogisticReg(xTrain, xTest, yTrain, yTest):
   #################### Logistic Regression ########################
   logReg = LogisticRegression()
   fited = logReg.fit(xTrain, yTrain)
   score = logReg.score(xTrain, yTrain)
   LRpredict = logReg.predict(xTest)
   return(metrics.accuracy_score(yTest, LRpredict))

def KNNeighbors(xTrain, xTest, yTrain, yTest):
   #################### KNN ########################
   knn = KNeighborsClassifier(n_neighbors = 3)
   fited = knn.fit(xTrain, yTrain)
   score = knn.score(xTrain, yTrain)
   KNNpredict = knn.predict(xTest)
   return(metrics.accuracy_score(yTest, KNNpredict))

def SuperVectorMachine(xTrain, xTest, yTrain, yTest):
   #################### Support Vector Machines ########################
   svmachine = SVC()
   fited = svmachine.fit(xTrain, yTrain)
   score = svmachine.score(xTrain, yTrain)
   SVMpredict = svmachine.predict(xTest)
   return(metrics.accuracy_score(yTest, SVMpredict))

def DecisionTree(xTrain, xTest, yTrain, yTest):
   #################### Decision Tree ########################
   dTree = tree.DecisionTreeClassifier()
   fited = dTree.fit(xTrain, yTrain)
   score = dTree.score(xTrain, yTrain)
   DTpredict = dTree.predict(xTest)
   return(metrics.accuracy_score(yTest, DTpredict))
       
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      ###LOAD FILE
      fileLoaded = request.files['file']
      fileLoaded.save(secure_filename(fileLoaded.filename))
      ###GET FILE INFO 
      fileSaved = fileLoaded.filename
      data = pd.read_csv(fileSaved)
      ###MATRIX
      arrX = data[data.columns[:-1]].values
      arrY = data[data.columns[-1]].values
      X_train, X_test, Y_train, Y_test = train_test_split(arrX, arrY, test_size=0.2, random_state=7)
      ###PREDICTION
      LRpred = LogisticReg(X_train, X_test, Y_train, Y_test)
      KNNpred = KNNeighbors(X_train, X_test, Y_train, Y_test)
      SVMpred = SuperVectorMachine(X_train, X_test, Y_train, Y_test)
      DTpred = DecisionTree(X_train, X_test, Y_train, Y_test)
      ###CHOOSE THE BEST PREDICTOR
      print('LR:',LRpred, 'KNN:',KNNpred, 'SVM:',SVMpred, 'DT:',DTpred)
      largest = [LRpred, KNNpred, SVMpred, DTpred]
  		sor = sorted(largest)       
  		bestPredictor = max(sor)
      
      print(bestPredictor)
      
      ###SAVE MODEL WITH PICKLE
      model_save = pickle.dump(bestPredictor, open('model.sav', 'wb'))      
      return flask.render_template('uploaded.html', bestPredictor=bestPredictor)

@app.route('/formulario', methods = ['GET','POST'])
def prediction():    
   if request.method == 'POST':
         to_predict_list  = request.form.to_dict()
         to_predict_list  = list(to_predict_list .values())
         to_predict_list = list(map(int, to_predict_list ))
         model = pickle.load(open("predictor.sav", "rb"))
         classes = model.predict(to_predict_list)         
         if (classes == 0):
              post_predict  = 'LOW'
         else if (classes == 2):
              post_predict  = 'HIGH'
         else if (classes == 1):
              post_predict == 'MEDIUM'
         return render_template("predict.html", model=model, post_predict=post_predict)

if __name__ == '__main__':
   app.run(debug = True)
