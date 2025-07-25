import mujoco
import mujoco.viewer
import numpy as np
from math import *
import time
from model import SnakeModel
import torch

CdXML = r"/home/louit/Documents/git/Robot_Snake" #mettre le chemins de vatre projet

# lancer al connextiona evc les XMLS
model = mujoco.MjModel.from_xml_path(rf"{CdXML}{r"/SnakeMujoco/snake/robot.xml"}")
model = mujoco.MjModel.from_xml_path(rf"{CdXML}{r"/SnakeMujoco/snake/scene.xml"}")
data = mujoco.MjData(model)

#recuper l'id de la tete
headID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head")
IDServo1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "Servo1")
IDServo2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "Servo2")
IDServo3 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "Servo3")
IDServo4 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "Servo4")

# recuper les Index des joinds
idx1 = model.jnt_qposadr[IDServo1]
idx2 = model.jnt_qposadr[IDServo2]
idx3 = model.jnt_qposadr[IDServo3]
idx4 = model.jnt_qposadr[IDServo4]


# Initialiser un modèle
snakemodel = SnakeModel()
# positions contain 4 positions for the 4 motors, the first three position are set to 0 for the motors, the last position is random
positions = torch.zeros((4,4)).float()  # 4 positions, 4 motors
positions[-1] = torch.randint(-90, 91, (4,)).float()

print('\n\nInitial positions:')
print(positions)
print("possiton = ",positions[-1][0] * (pi / 180))


timestep = 0
i = False
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)

        #controle des servo angle en radiant attention pas de limietre pour les angle mais il faute bien un ligmtre entre -90 et 90 et pas depasser
        #pas encore regadre la vitesse des moteur surement a modigfer et le couple
        data.ctrl[0] = positions[-1][0] * (pi / 180)  # Convert degrees to radians
        data.ctrl[1] = positions[-1][1] * (pi / 180)  # Convert degrees to radians
        data.ctrl[2] = positions[-1][2] * (pi / 180)  # Convert degrees to radians
        data.ctrl[3] = positions[-1][3] * (pi / 180)  # Convert degrees to  radians

        # print("Current control values (data.ctrl):", data.ctrl)
        
        # recuper les angles des joinds
        angle1 = data.qpos[idx1]
        angle2 = data.qpos[idx2]
        angle3 = data.qpos[idx3]
        angle4 = data.qpos[idx4]
        # print(f"Servo 1 = {angle1}, Servo 2 = {angle2}, Servo 3 = {angle3}, Servo 4 = {angle4}")

        # si ont reçois bien l'ID
        if headID != -1:
            # trouve la position des l'objet
            pos = data.xpos[headID]
            # print(f"Position de 'head' : x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}") # return les 3 position x y et z
        # else:
            # print("Le body 'head' est introuvable.")
        
        if (
            abs(angle1 - data.ctrl[0]) < 1 and
            abs(angle2 - data.ctrl[1]) < 1 and
            abs(angle3 - data.ctrl[2]) < 1 and
            abs(angle4 - data.ctrl[3]) < 1
        ):
            print("All servo angles match the control values within a delta of 0.1.")
            next_position = snakemodel(positions.view(1, -1))  # Reshape to [1, num_positions * num_moteurs]
            print("Next position from model:", next_position)
            positions = torch.cat([positions[1:], next_position[-1:].detach()], dim=0)
            print("Updated positions:", positions)

        timestep = timestep+0.05
        viewer.sync()
        time.sleep(0.05)
