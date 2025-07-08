# Robot_Snake
Le but de ce projet est d'aprndre le fonctionnement d'une IA qui permetra de faire anvancer un Robot,
Pour le type de Robot j'ai réaliser un robot serpent.

## Composant du Projet :
Sur ce porjte il y a besoind de plusssieur composant : 
- Servo-Moteur(mettre la refet) : permet de faire le deplacement du robot
- Arduino nano : Micro-controleur permet d'apliquer les consigne au servoMoteur

Pour le Corp de notre Robot vous pouvons trouver les modéles 3D des different piece
C'est piece ont etait modélisare sur fusion 360 puis assembler sur Onshape

## Réalisation de la simulation
Pour ce projet nous avpons réaliser une simulation pour permter l'entrenement de notre IA.
Pour cela nous avons utiliser Mujoco.

### Initilalisation de Mujoco
Pour utiliser Mujoco nousa alons passer par Python aevc un environement virtuelle
**Création de dossier virtuelle** : `phyton -m venv .ven` or `phyton3 -m venv .ven`
**Acitiver l'environemet virtuelle** : `source .ven\bin\activate`
**Instalalation des lib** : 
```
pip install mujoco
pip install onshape-to-robot
```
Pour convertire notre assemblage nous utiliser la lib **onshape-to-robot** qui permte de crées un XML( environement de notre model 3D dans mujoco) aevc les liason les poids et les contarinte.

Apres avoir fait cela nous pouvons lancer le programme **code**, SUr ce code vous devrait changer une varible cette variable devra etre le chemins vers le dossier Robot_Snake.

### IA 
Sur ce github nous avosn un dossier **DeepLearning** ce dossier contient mon aprendisage de l'IA grace des tuto sur youtube (mttre le liens des vidéo)
