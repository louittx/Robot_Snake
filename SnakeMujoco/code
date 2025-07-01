import mujoco
import mujoco.viewer
import numpy as np
from math import *
import time

# lancer al connextiona evc les XMLS
model = mujoco.MjModel.from_xml_path(r"/home/louit/Documents/pyton/SnakeMujoco/snake/robot.xml")
model = mujoco.MjModel.from_xml_path(r"/home/louit/Documents/pyton/SnakeMujoco/snake/scene.xml")
data = mujoco.MjData(model)

#recuper l'id de la tete
headID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head")


timestep = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)

        #controle des servo angle en radiant attention pas de limietre pour les angle mais il faute bien un ligmtre entre -90 et 90 et pas depasser
        #pas encore regadre la vitesse des moteur surement a modigfer et le couple
        data.ctrl[0] = pi
        data.ctrl[1] = pi/4
        data.ctrl[2] = pi/4
        data.ctrl[3] = pi/4
        if headID != -1:
            pos = data.xpos[headID]
            print(f"Position de 'head' : x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}") # return les 3 position x y et z
        else:
            print("Le body 'head' est introuvable.")
        timestep = timestep+0.05
        viewer.sync()
        time.sleep(0.05)


