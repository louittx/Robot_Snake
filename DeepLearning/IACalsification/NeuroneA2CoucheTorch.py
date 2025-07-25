import torch
import matplotlib.pyplot as plt
from sklearn.datasets import *
from sklearn.metrics import *
from tqdm import tqdm

def Init(NbInput,NbNeuron1, NbNeuron2):
    W1 = torch.randn((NbNeuron1,NbInput), dtype=torch.float32, requires_grad=True)
    b1 = torch.zeros((NbNeuron1,1), dtype=torch.float32, requires_grad=True)
    W2 = torch.randn((NbNeuron2,NbNeuron1), dtype=torch.float32, requires_grad=True)
    b2 = torch.zeros((NbNeuron2,1), dtype=torch.float32, requires_grad=True)
    
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

    Z1 = (W1 @ X)+b1
    A1 = torch.tanh(Z1)
    Z2 = (W2 @ A1)+b2
    A2 = torch.tanh(Z2)

    Activation ={
        'A1' : A1,
        'A2' : A2
    }

    return Activation



X = ((torch.rand(4, 4) * 180).float())-90 # Angle random entre -90 et 90 sur 4 moteur avec 4 action avant
y = ((torch.rand(4, 4) * 180).float())-90 # dimension de ce vecteur est de 1*4


IAX = (X-90)/180
IAY = (y-90)/180
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

Parametres = Init(X.shape[0],5,10)
Activation = ForwardPropagation(X,Parametres)
#print('Activation:', Parametres)
print('dimensions de y:', y.shape[0])

#plt.scatter(X[0, :], X[1, :], c=y, cmap='summer')

#Parametres = Algo(X,y,16,0.1,1000)