import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score


def init(x):
    W = np.random.randn(X.shape[1],1)
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
