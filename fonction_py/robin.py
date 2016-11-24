from fonction_py.tools import *
from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import svm
from sklearn import decomposition
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import uniform as sp_randint
from sklearn import datasets
from sklearn.linear_model import Ridge


def faire(xTrain,yTrain,xTest):
    le=[]
    sc = []
    data=pd.read_csv("data/trainPure.csv", sep=";", usecols=fields) # LECTURE du fichier de train,
    x,y=preprocess(data, "CAT")
    for i in range(100):
        xTrain, xTest, yTrain, yTest = faireSplitting(x, y, 0.8)
        model = RandomForestRegressor(bootstrap=False, criterion='mse', max_depth=5,
           max_features=30, max_leaf_nodes=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)
        model.fit(xTrain, yTrain)
        sc.append(model.score(xTrain, yTrain))
        pred = model.predict(xTest)
        pred[pred>max(yTrain)*1.05]=max(yTrain)*1.05
        pred[pred<0]=0
        pred=np.round(pred).astype(int)
        le.append(LinExp(pred, yTest))
    mean(sc)
    mean(le)
    plt.hist(le)
    plt.hist(sc)
    return np.round(pred).astype(int)

    
def opt(x,y):
    xTrain, xTest, yTrain, yTest = faireSplitting(x, y, 0.8)  # rajoute les features
    
    clf = RandomForestRegressor(n_estimators=20)
    m = np.random.normal(xTrain.shape[1]/2, 5, 20).astype(int)
    m[m<4]=4
    m[m>xTrain.shape[1]]= xTrain.shape[1]

    param_dist = {"max_depth": [100,90, 60, 50, 10, None],
              "max_features":list(m) ,
             # "min_samples_split": sp_randint(1, 11),
             # "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False]
              }
    
    rsearch = RandomizedSearchCV(estimator=clf, param_distributions=param_dist, n_iter=20)
    rsearch.fit(xTrain, yTrain)
    # summarize the results of the random parameter search
    print(rsearch.best_estimator_)
    model =rsearch.best_estimator_
    model.fit(xTrain, yTrain)  
    pred = model.predict(xTest)
    pred[pred>max(yTrain)*1.05]=max(yTrain)*1.05
    pred[pred<0]=0
    pred =np.round(pred)
    return [rsearch.best_estimator_, LinExp(pred, yTest)]

     
    
def robin(x, y):
    xTrain, xTest, yTrain, yTest = faireSplitting(x, y, 0.8)  # rajoute les features
    #del x
    #del y    
    listModel = []
    nest = [10,20,30]
    mfea = [30,70,100]
    mdep = [3,5,8,10]
    for i in nest:
        for j in mfea:
            for k in mdep:
                listModel.append(RandomForestRegressor(n_estimators=i, bootstrap=False, max_depth=k, max_features=j))
    # GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=10, random_state=0),  svm.SVC()
    res =[]
    start_time = time.time()
    i=0
    for model in listModel:
        i=i+1
        #start_time = time.time()
        print(i)
        model.fit(xTrain, yTrain)
        #model.score(xTrain, yTrain)
        pred = model.predict(xTest)
        pred[pred>max(yTrain)*1.05]=max(yTrain)*1.05
        pred[pred<0]=0
        pred =np.round(pred)
        res.append(LinExp(pred, yTest))
    print("--- %s seconds ---" % str((time.time() - start_time)))   
    return res
   
    
def robinTel(x,):
    res = []
    for i in range(10):
        xTrain, xTest, yTrain, yTest = faireSplitting(x, y, 0.8)  # rajoute les features
        #del x
        #del y    
        model =RandomForestRegressor(n_estimators=40, bootstrap=False, max_depth=1, max_features=12)
        model.fit(xTrain, yTrain)
        #model.score(xTrain, yTrain)
        pred = model.predict(xTest)
        pred[pred>max(yTrain)*1.05]=max(yTrain)*1.05
        pred[pred<0]=0
        pred =np.round(pred)
        res.append(LinExp(pred, yTest))
        LinExp(pred, yTest)
    mean(res)
    bins = range(-300, 300, 600)
    plt.hist(pred-yTest)
    plt.hist(res)
    print("--- %s seconds ---" % str((time.time() - start_time)))   
    return res
    
#    print("PCA")
#    pos = [1,3,5,10,20,30,40,50, 60,62, 65, 70, 75,78, 80, 90]
#    resAcc = []
#    resLin = []
#    for i in pos:
#        pca = decomposition.PCA(n_components=i)
#        pca.fit(xTrain)
#        PCAxTrain = pca.transform(xTrain)
#        model = linear_model.LinearRegression()
#        model.fit(PCAxTrain, yTrain)
#        model.score(PCAxTrain, yTrain)
#        pred = model.predict(pca.transform(xTest))
#        pred =np.floor(np.round(pred))
#        resAcc.append(Accuracy(pred, yTest.values))
#        resLin.append(LinExp(pred, yTest.values))
#    print(resAcc)    
#    print(resLin)    
#    plt.plot(pos, resLin)
#    plt.show()
#    plt.plot(pos, resAcc)
#    plt.show()    
#######################################################################   
    #best accuracy en % :
#52.5071805097
#linEx :
#23987.9825805
## del x['TPER_TEAM']
#    x['YEAR'] = x['DATE'].str[0:4]
#    x['MONTH'] = x['DATE'].str[5:7]
#   # x['DAY'] = x['DATE'].str[8:10]
#    #x['HOUR'] = x['DATE'].str[-12:-10].astype(int)
#    x['HOUR'] = x['HOUR']+ ':'+((x['DATE'].str[-9:-8].astype(int)==3)*0.5).astype(str)
#    x['HOUR'] = x['DATE'].str[-12:-8]
#    del x['DATE']
#
#    x=pd.get_dummies(x)