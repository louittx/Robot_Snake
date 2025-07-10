import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from tqdm import tqdm


def init(X):
    W = np.random.randn(X.shape[1],1)
    b = np.random.randn(1)
    return (W,b)

def Model(X,W,b):
    Z = X.dot(W) + b
    A = 1/ (1+np.exp(-Z))
    return A

def Cost(A,y) : 
    #return (-1/ len(y))*sum(y *np.log(A)+(1-y)*np.log(1-A))
    return log_loss(y,A)

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
    return A>=0.5

def Algo(X,y,a=0.1,i=100):
    W,b = init(X)
    LogLoss=[]
    Acc = []
    for i in tqdm(range(i)): # tqdm permet d'afficher un bare de progression
        A = Model(X,W,b)
        if (i%10 == 0): # permte d'aller plus rapitement
            LogLoss.append(Cost(A,y))
            yPredict = predict(X,W,b)
            Acc.append(accuracy_score(y,yPredict))
        dW,db = Gradiants(A,X,y)
        W,b = Uptate(W,b,dW,db,a)

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(LogLoss)
    plt.subplot(1,2,2)
    plt.plot(Acc)
    plt.show()
    return (W,b)

def Algo2(XTrain,yTrain,XTest,yTest,a=0.1,i=100):
    W,b = init(XTrain)
    LogLossTrain=[]
    AccTrain = []
    LogLossTest=[]
    AccTest = []
    for i in tqdm(range(i)): # tqdm permet d'afficher un bare de progression
        ATrain = Model(XTrain,W,b)
        if (i%10 == 0): # permte d'aller plus rapitement
            LogLossTrain.append(Cost(ATrain,yTrain))
            yPredict = predict(XTrain,W,b)
            AccTrain.append(accuracy_score(yTrain,yPredict))

            ATest = Model(XTest,W,b)
            LogLossTest.append(Cost(ATest,yTest))
            yPredict = predict(XTest,W,b)
            AccTest.append(accuracy_score(yTest,yPredict))
        dW,db = Gradiants(ATrain,XTrain,yTrain)
        W,b = Uptate(W,b,dW,db,a)

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(LogLossTrain, label="Train")
    plt.plot(LogLossTest, label="Test")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(AccTrain, label="Train")
    plt.plot(AccTest, label="Test")
    plt.legend()
    plt.show()
    return (W,b)

