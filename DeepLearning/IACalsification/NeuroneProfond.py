import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import *
from sklearn.metrics import *
from tqdm import tqdm

def init(Dimmension):
    C = len(Dimmension)
    Parametre = {}
    for i in range(1,C):
        Parametre['W'+str(i)]=torch.tensor(np.random.randn(Dimmension[i],Dimmension[i-1]),requires_grad=True)
        Parametre['b'+str(i)]=torch.tensor(np.random.randn(Dimmension[i],1),requires_grad=True)
    return Parametre

def ForwardPropagation(X, Paramtre):
    Activation = {'A0' : X}
    C = len(Paramtre) // 2
    for i in range (1,C+1):
        Z = Paramtre['W' +str(i)].dot(Activation['A'+str(i-1)])+Paramtre['b'+str(i)]
        Activation['A'+str(i)] = 1/ (1+torch.exp(-Z))
    return Activation

def BackProagation(y, Parametres, Activations):

    Gradients ={}
    m = y.shape[1]
    C = len(Parametres) // 2
    dZ = Activations['A'+str(C)]-y
    for i in reversed(range(1,C+1)) :
        Gradients['dW'+str(i)]= 1 / m * np.dot(dZ,Activations['A'+str(i-1)].T)
        Gradients['db'+str(i)]= 1 / m * np.sum(dZ,axis=1, keepdims=True)
        dZ = np.dot(Parametres['W'+str(i)].T,dZ)*(Activations['A'+str(i-1)]*(1-Activations['A'+str(i-1)]))
    return Gradients

def Update(Gradients, Parametres, a):

    C = len(Parametres) // 2
    for i in range(1,C+1):
        Parametres['W'+str(i)] = Parametres['W'+str(i)] - a * Gradients['dW'+str(i)]
        Parametres['b'+str(i)] = Parametres['b'+str(i)] - a * Gradients['db'+str(i)]
    return Parametres

def Predict(X, Parametre):
    C = len(Parametre) // 2
    Activation = ForwardPropagation(X, Parametre)
    Af = Activation['A'+str(C)]
    return (Af)>=0.5

def Algo(X,y,NbNeuroneAndCouche = (32,32,32),a=0.1,i=100):

   # np.random.seed(0) # ce qui permte d'avoir toujours la meme aléoitoir quand ont lance donc permte de repoidure l'experient aevc les meme nombre
    
    # Init
    Dimension = list(NbNeuroneAndCouche)
    Dimension.insert(0,X.shape[0])
    Dimension.append(y.shape[0])
    Parametre = init(Dimension)
    LogLoss=[]
    Acc = []
    for i in tqdm(range(i)): # tqdm permet d'afficher un bare de progression
        Activation = ForwardPropagation(X,Parametre)
        Gradient = BackProagation(y,Parametre,Activation)
        Parametre = Update(Gradient,Parametre,a)
        if (i%10 == 0): # permte d'aller plus rapitement
            C = len(Parametre)//2
            LogLoss.append(log_loss(y.flatten(),Activation['A'+str(C)].flatten()))
            yPredict = Predict(X,Parametre)
            Acc.append(accuracy_score(y.flatten(),yPredict.flatten()))

    print(Acc[-1])
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(LogLoss)
    plt.subplot(1,2,2)
    plt.plot(Acc)
    plt.show()
    return (Parametre)


X, y = make_blobs(n_samples=100,n_features=3, centers=2, random_state=5)
X = X.T
y = y.reshape((1, y.shape[0]))

print(X)
print('dimensions de y:', y.shape)
print(X[0, :])

plt.scatter(X[0, :], X[1, :], c=y, cmap='summer')
plt.show()
#Parametres = Algo(X,y,(32,32,32),0.1,1000)