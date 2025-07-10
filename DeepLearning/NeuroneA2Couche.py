import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import *
from sklearn.metrics import *
from tqdm import tqdm

def init(NbInput,NbNeuron1, NbNeuron2):
    W1 = np.random.randn(NbNeuron1,NbInput)
    b1 = np.zeros((NbNeuron1,1))
    W2 = np.random.randn(NbNeuron2,NbNeuron1)
    b2 = np.zeros((NbNeuron2,1))
    
    Parametre = {
        'W1' : W1,
        'b1' : b1,
        'W2' : W2,
        'b2' : b2
    }
    return Parametre

def ForwardPropagation(X, Paramtre):
    W1 = Paramtre['W1']
    b1 = Paramtre['b1']
    W2 = Paramtre['W2']
    b2 = Paramtre['b2']

    Z1 = W1.dot(X)+b1
    A1 = 1/ (1+np.exp(-Z1))
    Z2 = W2.dot(A1)+b2
    A2 = 1/ (1+np.exp(-Z2))

    Activation ={
        'A1' : A1,
        'A2' : A2
    }

    return Activation

def BackProagation(X, y, Parametres, Activations):

    A1 = Activations['A1']
    A2 = Activations['A2']
    W2 = Parametres['W2']

    m = y.shape[1]

    dZ2 = A2 - y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims = True)

    dZ1 = np.dot((W2.T), dZ2) * A1 * (1 - A1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims = True)

    Gradients = {
        'dW1' : dW1,
        'db1' : db1,
        'dW2' : dW2,
        'db2' : db2
    }
    
    return Gradients

def Update(Gradients, Parametres, a):

    W1 = Parametres['W1']
    b1 = Parametres['b1']
    W2 = Parametres['W2']
    b2 = Parametres['b2']

    dW1 = Gradients['dW1']
    db1 = Gradients['db1']
    dW2 = Gradients['dW2']
    db2 = Gradients['db2']

    W1 = W1 - a * dW1
    b1 = b1 - a * db1
    W2 = W2 - a * dW2
    b2 = b2 - a * db2

    Parametres = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    return Parametres

def Predict(X, Parametre):
    Activation = ForwardPropagation(X, Parametre)
    A2 = Activation['A2']
    return A2>=0.5

def Algo(X,y,NbNeuron1,a=0.1,i=100):

    NbInput = X.shape[0]
    NbNeuron2 = y.shape[0]
    Parametre = init(NbInput,NbNeuron1,NbNeuron2)
    LogLoss=[]
    Acc = []
    for i in tqdm(range(i)): # tqdm permet d'afficher un bare de progression
        Activation = ForwardPropagation(X,Parametre)
        Gradient = BackProagation(X,y,Parametre,Activation)
        Parametre = Update(Gradient,Parametre,a)
        if (i%10 == 0): # permte d'aller plus rapitement
            LogLoss.append(log_loss(y.flatten(),Activation['A2'].flatten()))
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



X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=60)
X = X.T
y = y.reshape((1, y.shape[0]))

print('dimensions de X:', X.shape)
print('dimensions de y:', y.shape)

plt.scatter(X[0, :], X[1, :], c=y, cmap='summer')

Parametres = Algo(X,y,16,0.1,1000)