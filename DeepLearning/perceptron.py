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



X, y = make_blobs(n_samples=100,n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0],1))

W,b = Algo(X,y,0.1,200)
NewData = np.array([2, 1])

x0 = np.linspace(-1,4,100)
x1 = (-W[0]*x0-b)/W[1]
plt.scatter(X[:,0],X[:,1], c=y, cmap="summer")
plt.scatter(NewData[0],NewData[1], c='r')
plt.plot(x0,x1,c="orange",lw=3)
plt.show()
Value = predict(NewData,W,b)
print("Value = ",Value)

