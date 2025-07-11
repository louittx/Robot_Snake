import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import *
from sklearn.metrics import *
from tqdm import tqdm

def init(x):
    W = np.full((X.shape[1],5),1)
    b = np.random.randn(1)
    return (W,b)

def Model(X,W,b):
    Z = X.dot(W) + b
    A = 1/ (1+np.exp(-Z))
    return A

def Cost(A,y) : 
    return (-1/ len(y))*sum(y *np.log(A)+(1-y)*np.log(1-A))

def Gradiants(A,X,y) :
    dW = (1/len(y))*np.dot(X.T,A-y)
    db = (1/len(y))*sum(A-y)
    return dW, db

def Uptate(W,b,dW,db, a) :
    W = W - a*dW
    b = b - a*db
    return W,b

def predict(x,W,b):
    A = Model(x,W,b)
    print(A)
    return A>=0.5

def Algo(X,y,a=0.1,i=100):
    W,b = init(X)
    LogLoss=[]
    for i in range(i):
        A = Model(X,W,b)
        LogLoss.append(Cost(A,y))
        dW,db = Gradiants(A,X,y)
        W,b = Uptate(W,b,dW,db,a)
    yPredict = predict(X,W,b)
    print(accuracy_score(y,yPredict))
    #plt.plot(LogLoss)
    #plt.show()
    return (W,b)


X = np.full((10,5),1)
y = np.full((10,5),1)
W,b= init(X)
A = Model(X,W,b)
print(A)
print(b)