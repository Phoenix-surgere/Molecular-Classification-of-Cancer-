# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:27:16 2019

@author: black
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
train = pd.read_csv('data_set_ALL_AML_train.csv')
test = pd.read_csv('data_set_ALL_AML_test.csv')
labels = pd.read_csv('actual.csv', index_col='patient')
#print(train.isna().sum().sum())
seed = 42


train = train.select_dtypes(exclude='object').T
test = test.select_dtypes(exclude='object').T

#data = pd.concat([train,test], axis=0,sort=False)
#data.index = pd.to_numeric(data.index)
#data = data.sort_index()

labels = labels.cancer.map({'ALL':0, 'AML':1})

train.reset_index(drop=True,inplace=True)
test.reset_index(drop=True, inplace=True)
labels.reset_index(drop=True, inplace=True)

X_train, X_val, y_train, y_val = train_test_split(train, labels[:38],
                         test_size=0.15,random_state=seed)

from tpot import TPOTClassifier
#1. TPOT 
pipeline_optimizer = TPOTClassifier(generations=25, population_size=20,
   offspring_size=5, scoring='accuracy',cv=3,random_state=seed,
   verbosity=2)

#pipeline_optimizer.fit(X_train, y_train)
#print(pipeline_optimizer.score(X_test, y_test))

#2. Bayesian SVM, XGBOOST
from sklearn.preprocessing import StandardScaler as SS
from sklearn.decomposition import PCA
import xgboost as xgb
from hyperopt import fmin, hp, tpe,space_eval
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV as RSCV,cross_val_score

components = 30 #found by reading other peoples guides and own search
scaler = SS()
#data_scaled = scaler.fit_transform(data)
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

pca = PCA(n_components=components)  #APPLY PCA ON TRAIN DATA - SEARCH+NOTEBOOK
pca.fit(X_train_scaled)
#pca.fit(data_scaled)
pca_results = pca.explained_variance_ratio_.cumsum()*100
plt.bar(range(components), pca_results); 
plt.xlabel('Number of Components'); plt.ylabel('Explained Variance (%)')
plt.title('PCA Major Principal Components Explained Variance')
plt.show()

X_train_reduced = pca.fit_transform(X_train_scaled)
X_val_reduced = pca.transform(X_val_scaled)

#X_train_reduced = scaler.fit_transform(X_train_reduced)
#X_test_reduced = scaler.transform(X_test_reduced)


dist_svm = {
        'C':np.linspace(0.0001, 1, 100),
        'gamma': np.linspace(0.0001, 1,100) ,
        'kernel': ['linear', 'rbf', 'sigmoid']}        
svm = SVC(random_state=seed,probability=True)
grid_svm = RSCV(svm, param_distributions=dist_svm, cv=5, scoring='accuracy', 
verbose=1, n_jobs=2, n_iter=800)
#grid_svm.fit(X_train_reduced, y_train) 
#best_parameters = grid_svm.best_estimator_
#y_pred_val = grid_svm.predict(X_val_reduced)
#
#y_test = labels[38:]
#X_test = scaler.transform(test)
#X_test_reduced = pca.transform(X_test)
#y_pred = grid_svm.predict(X_test_reduced)

space_svm = {'C': hp.uniform('C', 0.00001, 10), 
'gamma': hp.uniform('gamma', 0.0001, 10), 
'kernel': hp.choice('kernel', ['linear', 'rbf', 'sigmoid'])}
def objective_PLACEHOLDER(params):
    params={'C':params['C'],'gamma':params['gamma'],'kernel':params['kernel']}
    model = SVC(random_state=seed, probability=True, **params)
    best_score = cross_val_score(model, X_train_reduced, y_train, scoring='f1',
   cv=5, n_jobs=2).mean()
    return -best_score

#best = fmin(fn=objective, space=space_svm, max_evals=100, 
#            rstate=np.random.RandomState(seed),algo=tpe.suggest)
#params = space_eval(space_svm, best)
#svm = SVC(random_state=seed, probability=True, **params)

space_xgb = {
'max_depth': hp.choice('max_depth', np.arange(200, 1000, 100, dtype=int)), #hp.quniform('max_depth', 2, 8, 1),
'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
'gamma': hp.uniform('gamma', 0.0, 0.5),
'n_estimators': hp.choice('n_estimators', np.arange(200, 1000, 100, dtype=int))
        }

def objective(params):
    params = {
        'max_depth': int(params['max_depth']),
        'gamma': "{:.3f}".format(params['gamma']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
        'n_estimators': int(params['n_estimators'])
    }
    
    clf = xgb.XGBClassifier(
        objective='reg:logistic',
        seed=seed,
        n_jobs=2,
        **params
    )
    
    best_score = cross_val_score(clf, X_train_reduced, y_train, scoring='accuracy', 
                            cv=5).mean()
    loss = -best_score
    return loss

best = fmin(fn=objective, space=space_xgb,  max_evals=100,
            rstate=np.random.RandomState(seed), algo=tpe.suggest)
params = space_eval(space_xgb, best)
XGB = xgb.XGBClassifier(random_state=seed, **params)
y_pred_val = XGB.predict(X_val_reduced)
#y_pred = XGB.predict(X_test_reduced)
    