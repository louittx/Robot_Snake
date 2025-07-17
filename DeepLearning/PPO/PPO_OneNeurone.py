import numpy as np

def Politique(S,W,b):
    Mu = W*S+b
    return Mu


def Action(Mu,Sigma):
    A = np.random.normal(Mu,Sigma)
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
    NewPi = (1 / (np.sqrt(2 * np.pi) * Sigma)) * np.exp(-((A - Mu)**2) / (2 * Sigma**2))
    rt = NewPi / (OldPi + 1e-8)  # pour éviter la division par zéro
    return NewPi, rt


def Perte(rt, A, Epsilon):
    clipped = np.clip(rt, 1 - Epsilon, 1 + Epsilon)
    L = -min(rt * A, clipped * A)  # on minimise donc on met le signe -
    return L
