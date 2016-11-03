import pandas as pd
import numpy as np
from scipy import stats
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import time


start_time = time.time()
print("go")
fields = ['DATE', 'DAY_OFF', 'DAY_DS', 'WEEK_END', 'DAY_WE_DS','TPER_TEAM', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS' ]
data=pd.read_csv("data/train_2011_2012_2013.csv", sep=";", usecols=fields, nrows=10)

x = data[fields[0:-2]]
y = data[fields[-1]]
ass = data[fields[-2]]
#x['DATE'] = pd.to_datetime(x['DATE'])

x = pd.concat([x,x['DATE']], axis=1)
print(x)

print("--- %s seconds ---" % (time.time() - start_time))



