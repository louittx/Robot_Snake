#Le but de l'exo c'est avec un datasheet d'imge de diférencier un chient ou un chat sur une imaeg de 64px par 64px

from utilities import *
from PerceptronLib import *

XTrain, yTrain, Xtest, yTest = load_data()

print(Xtest.shape)
print(yTrain.shape)
