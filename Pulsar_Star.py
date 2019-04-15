# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 11:59:20 2019

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sea
import missingno

x = pd.read_csv("star.csv")

y = x["target_class"]

x= x.drop(["target_class"],axis =1)

x.isnull().sum().plot(kind ="bar")

sea.heatmap(x.corr(),annot = True)

sea.pairplot(x)

sea.countplot(y)




from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test =train_test_split(x,y,test_size = 0.2)


sea.countplot(y_train)

sea.countplot(y_test)




from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

x_train.iloc[:,[0,1,4,5,6,7]] = ss.fit_transform(x_train.iloc[:,[0,1,4,5,6,7]])

x_test.iloc[:,[0,1,4,5,6,7]] = ss.transform(x_test.iloc[:,[0,1,4,5,6,7]])

from sklearn.decomposition import PCA

pc =PCA(n_components= 3)

x_train = pc.fit_transform(x_train)

x_test = pc.transform(x_test)

exp  =pc.explained_variance_ratio_


from sklearn.svm import SVC

sv = SVC()

sv.fit(x_train,y_train)


sv.score(x_train,y_train)

y_pred = sv.predict(x_test)


from sklearn.tree import DecisionTreeClassifier as  dt

dt = dt()

dt.fit(x_train,y_train)

dt.feature_importances_

dt.score(x_train,y_train)

y_pred = dt.predict(x_test)


from sklearn.ensemble import RandomForestClassifier as rf

rf = rf(n_estimators=110)

rf.fit(x_train,y_train)


rf.score(x_train,y_train)

y_pred = rf.predict(x_test)


from sklearn.metrics import confusion_matrix,classification_report

cr = classification_report(y_test,y_pred)

cm = confusion_matrix(y_test,y_pred)

sea.heatmap(cm,annot = True,fmt= ".1f")

