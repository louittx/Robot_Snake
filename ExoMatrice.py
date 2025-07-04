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

### Exo4 
# Affiche la matrice A en image avec plt.imshow(), ajoute un titre et une barre de couleur.
plt.imshow(A, cmap='gray')
plt.colorbar()
plt.title("Matrice A")
plt.show()

### Exo5
# Génére un jeu de données de 150 points, 3 centres avec make_blobs.
# Affiche les points avec plt.scatter, colore-les selon les labels avec cmap='rainbow'.
# Ajoute un titre au graphique.
X, y = make_blobs(n_samples=150, centers=3,n_features=2,random_state=42) # cres un données de 150 point, centre = 3 groupe nrepartie au centre, n_feactures=2 donc deux coredonnées a mais point 
# random_state premte de reproduire le meme resulta
print('y = ',y)
plt.scatter(X[:,0], X[:,1], c=y, cmap='rainbow')
plt.title("Exemple de make_blobs")
plt.show()

### Partie 1 Numby
## ex1.1
# crées un Matrice A 4,4 avec des nombre aléatoir entre 0 et 100
A = np.random.randint(0,101,(4,4))
print("A = ",A,"\n")

## ex1.2
# calcule la somme de chaque ligne
# calcule la moy de chque colone
# la diagonale principale
sumA = np.sum(A,0) # 0 = ligne
moyA = np.mean(A,1) # 1 = colone
diagoA = A.diagonal()
print("SumA = ",sumA,"\n")
print("moyA = ",moyA,"\n")
print("diagoA = ",diagoA,"\n")
## ex1.3
#crées un masque boolean pour extraite tout les element supérieur à 50
A50 = A[A > 50]
print("A50 = ",A50,"\n")
## ex1.4
#crées un matrice B qui contient tout les elemnt de A qui sont superieur a 30
B = A[A > 30]
print("B = ",B,"\n")