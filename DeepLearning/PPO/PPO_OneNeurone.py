import numpy as np
import torch

def Init():
    W = torch.tensor(np.random.randn(1),requires_grad=True)
    b = torch.tensor(np.random.randn(1),requires_grad=True)
    return W,b

def Politique(S,W,b):
    Mu = W*S+b
    return Mu

def Action(Mu,Sigma):
    A = torch.normal(Mu, torch.tensor(Sigma))
    return A

def Environement(A):
    r = -np.abs(A)
    return r

def Retour(r):
    return r

def Avantage(R,Vs):
    Av = R-Vs
    return Av
 
def PPOLoss(OldPi, A, Mu, Sigma):
    # Utilise torch uniquement
    NewPi = (1 / (torch.sqrt(2 * torch.pi * Sigma**2))) * torch.exp(-((A - Mu)**2) / (2 * Sigma**2))
    rt = NewPi / (OldPi + 1e-8)
    return NewPi, rt

def Perte(rt, A, Epsilon):
    clipped = torch.clamp(rt, 1 - Epsilon, 1 + Epsilon)
    L = -torch.min(rt * A, clipped * A)
    return L

def Gradiant(L,W,b) : 
    W.grad.zero_()
    b.grad.zero_()
    L.backward()
    DLW = W.grad.clone()
    DLb = b.grad.clone()
    return DLW,DLb

def Update(W, b, dLW, dLb, alpha):
    with torch.no_grad(): # pour modifer les valeur sans modifer que ce sont des gradiant
        W -= alpha * dLW
        b -= alpha * dLb
    return W, b


s = torch.tensor(1.0)
Sigma = 0.1
OldPi=0
Epsilon = 1*10**(-10)
alpha = 0.1
Wu,bu = Init()
Wv,bv = Init()
for i in range(1000):
    Mu = Politique(s,Wu,bu)
    A = Action(Mu,Sigma)
    r = Environement(A)
    Recompence = Retour(r)
    VS = Politique(s,Wv,bv)
    Av = Avantage(Recompence,VS)
    NewPi,rt = PPOLoss(OldPi,A,Mu,Sigma)
    L = Perte(rt,A,Epsilon)
    DLW,DLb = Gradiant(L,Wu,bu)
    Update(Wu,bu,DLW,DLb,alpha)
    OldPi = NewPi

