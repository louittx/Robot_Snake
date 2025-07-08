# Robot_Snake

L’objectif de ce projet est de comprendre le fonctionnement d’une intelligence artificielle capable de faire avancer un robot.  
Le robot choisi pour cette expérimentation est un robot serpent.

## Composants du projet

Ce projet nécessite plusieurs composants :  
- **Servo-moteur** (référence à préciser) : assure le déplacement du robot  
- **Arduino Nano** : microcontrôleur chargé d’envoyer les consignes aux servo-moteurs

Concernant le châssis du robot, les modèles 3D des différentes pièces sont disponibles.  
Ces pièces ont été modélisées avec **Fusion 360**, puis assemblées sous **Onshape**.

## Simulation

Une simulation a été mise en place afin d'entraîner notre intelligence artificielle.  
Pour cela, nous utilisons le moteur de simulation **MuJoCo**.

### Initialisation de MuJoCo

L'utilisation de MuJoCo se fait via **Python** avec un environnement virtuel.

- **Création de l’environnement virtuel** :  
  `python -m venv .venv` ou `python3 -m venv .venv`

- **Activation de l’environnement virtuel** :  
  `source .venv/bin/activate`

- **Installation des bibliothèques nécessaires** :
```
pip install mujoco
pip install onshape-to-robot
```

Nous utilisons la bibliothèque **onshape-to-robot** pour convertir l’assemblage 3D en fichier XML, compatible avec MuJoCo.  
Ce fichier XML décrit l’environnement du robot (liaisons, poids, contraintes, etc.).

Une fois le fichier généré, vous pouvez lancer le programme principal (**code**).  
N'oubliez pas de modifier une variable dans ce code : elle doit contenir le chemin vers le dossier `Robot_Snake`.

### Intelligence Artificielle

Le répertoire **DeepLearning** contient mes expérimentations en apprentissage automatique, réalisées à partir de tutoriels trouvés sur YouTube (liens à ajouter).

