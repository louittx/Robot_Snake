import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

### Exo1
# Résultat attendu pour A :
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]

# Résultat attendu pour B :
# [[2 2 2]
#  [2 2 2]
#  [2 2 2]]

A = np.arange(1,10).reshape(3, 3) # crées un vecteur lineaire de 1 à 9 puis le réajuste en matrcie 3 3
B = np.full((3,3),2)
print('A = ',A,'\n')
print('B = ',B,'\n')

### Exo2
# C = A + B
# D = A * B
# E = A @ B.T produit matricell et tramposer de B

C = A + B
D = A * B 
E = A @ B.T
print('C = ',C,'\n')
print('D = ',D,'\n')
print('E = ',E,'\n')

### Exo3
# Transposer de A
# moy de cahque ligne de A
#Somme total des élement de A
Atrans = A.T
moyLA = np.mean(A,1) # 1 pour chauqe ligne 0 ppur chaque colone
SumA = np.sum(A)
print('Atrans = ',Atrans,'\n')
print('moyLA = ',moyLA,'\n')
print('SumA = ',SumA,'\n')
