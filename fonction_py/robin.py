from fonction_py.tools import *
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd

# predire que Gestion renault = 0 depuis fevrier/2011
#
#




def robin(x, y):
    xTrain, xTest, yTrain, yTest = faireSplitting(x, y, 0.8)  # rajoute les features
    del x
    del y    
    print("ok")
    model = linear_model.LinearRegression()
    model.fit(xTrain, yTrain)
    model.score(xTrain, yTrain)
    pred = model.predict(xTest)
    pred =np.floor(np.round(pred))
    check(pred, yTest) 
    
    
    
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