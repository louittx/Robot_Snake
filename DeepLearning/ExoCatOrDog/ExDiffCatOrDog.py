#Le but de l'exo c'est avec un datasheet d'imge de diférencier un chient ou un chat sur une imaeg de 64px par 64px

# TO DO
# 1. Normaliser le train_set et le test_set (0-255 -> 0-1)
# 2. aplatir les matrcie pour avoir 1000,4096 les variables du train_set et du test_set (64x64 -> 4096)
# 3. Entraîner le modele sur le train_set (tracer la courbe d'apprentissage, trouver les bons hyper-params)
#    (si vous rencontrez un probleme avec le log_loss, utiliser la fonction de sklearn a la place !)
# 4. Évaluer le modele sur le test_set (tracer également la courbe de Loss pour le test_set)
# 5. Partager vos conclusions dans les commentaires !

from utilities import *
from PerceptronLib import *

def Normaliser(X):
    return((X-X.min())/(X.max()-X.min()))



XTrain, yTrain, XTest, yTest = load_data()

XTrainNorm = Normaliser(XTrain)
XTestNorm = Normaliser(XTest)

XTrainNormFlat = XTrainNorm.reshape(XTrainNorm.shape[0],-1) # redimensiont la matrix de 1000 64 64 en 1000 4096/ le -1 calculet automatiquemnt le nobre de collone restant
XTestNormFlat = XTestNorm.reshape(XTestNorm.shape[0],-1)
print(XTrain.shape)
print(XTrainNormFlat.shape)

Algo(XTrainNormFlat,yTrain,XTestNormFlat,yTest,a=0.01,i=10000)

# Sur la courbe nous voyons que notre IA est en ovefiniti donc elle aprende trop pour rien, Pour resoudre cela nous pouvons rejouter plus d'image de chient et de chat,
#cela va pas cahnger car notre IA est sur un seul neront donc lineaire