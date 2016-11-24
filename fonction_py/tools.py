from numpy import *
import math
import pandas

def poly_exp(X, degree):
    N,D = X.shape
    for d in range(2,degree+1):
        X = column_stack([X,X[:,0:D]**d])
    return X

def MSE(yt,yp):
    print("NE PAS UTILISER MSE !! utiliser LinExp !!!")


def normalize(df):
    return (df - df.mean()) / (df.max() - df.min())




def faireSplitting(x, y, taille): # return xTrain, xTest, yTrain, yTest
    ln = (random.rand(x.shape[0]) < taille)
    return x[ln], x[~ln], y[ln], y[~ln];




def check(yEmpirique, yTest): # A UTILISER AVEC LES DATA FRAME DE PANDAS
    alpha=0.1

    if(yTest.shape[0] != yEmpirique.shape[0]):
        print("Erreur sur la taille de la prédiction")
        return 0
    print("accuracy en % :")
    print(sum(yEmpirique==yTest)*100/yEmpirique.shape[0]) # pourcentage de bonne prediction

    linex = 0
    diff = (yTest-yEmpirique).values

    for i in range(len(diff)):
        linex = linex + math.exp(alpha * diff[i]) - alpha*diff[i]-1

    print("linEx :")
    print(linex/yTest.shape[0])

def LinExp(yEmpirique, yTest):#Retourne l'erreur moyenne #UTILISER AVEC DES VECTEURS : POUR CONVERTIR DATA FRAME TO VECTOR DataFrame.values
    alpha = 0.1
<<<<<<< HEAD
=======
    coeff=linspace(1,3,len(yEmpirique))
>>>>>>> origin/master
    linex = 0
    diff = (yTest - yEmpirique).values
    for i in range(len(diff)):
        linex = linex + coeff[i]*( math.exp(alpha * diff[i]) - alpha * diff[i] - 1)
    return linex/yTest.shape[0]

def MatLinExp(yEmpirique, yTest): #retourne la matrice d'erreur#UTILISER AVEC DES VECTEURS : POUR CONVERTIR DATA FRAME TO VECTOR DataFrame.values
    alpha = 0.1
    linex = []
    diff = (yTest - yEmpirique)
    for i in range(len(diff)):
        linex.append(math.exp(alpha * diff[i]) - alpha * diff[i] - 1)
    return linex

def Accuracy(yEmpirique, yTest):
    return sum(yEmpirique==yTest)*100/yEmpirique.shape[0]