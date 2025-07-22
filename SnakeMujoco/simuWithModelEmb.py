import mujoco
import mujoco.viewer
import numpy as np
from math import *
import time
from modelEmb import *
import torch

CdXML = r"/Users/xblanc/devs/Robot_Snake" #mettre le chemins de vatre projet

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
snakemodel = SnakeModelEmb()
# positions contain 4 positions for the 4 motors, the first three position are set to 0 for the motors, the last position is random
positions = torch.zeros((4,4)).int()  # 4 positions, 4 motors
positions[-1] = torch.randint(0, 181, (4,)).int()

print('\n\nInitial positions:')
print(positions)


nb_tries = 100  # Maximum number of tries to avoid infinite loop

timestep = 0
i = False
is_moving = False
head_last_position = 0
# Dataset to store (positions, next_position, moved)
dataset = []

def add_to_dataset(positions, next_position, moved):
    # Store a tuple: (input_positions, next_position, moved_flag)
    # Clone tensors to avoid mutation
    dataset.append((
        positions,
        next_position,
        moved
    ))

num_try = 0
current_positions = []
current_next_position = []

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)

        if (num_try > nb_tries):
            print("Maximum number of tries reached. Exiting.")
            break

        #controle des servo angle en radiant attention pas de limietre pour les angle mais il faute bien un ligmtre entre -90 et 90 et pas depasser
        #pas encore regadre la vitesse des moteur surement a modigfer et le couple
        if not is_moving:
            data.ctrl[0] = (positions[-1][0] - 90) * (pi / 180)  # Convert degrees to radians
            data.ctrl[1] = (positions[-1][1] - 90) * (pi / 180)  # Convert degrees to radians
            data.ctrl[2] = (positions[-1][2] - 90) * (pi / 180)  # Convert degrees to radians
            data.ctrl[3] = (positions[-1][3] - 90) * (pi / 180)  # Convert degrees to  radians
            is_moving = True
            head_last_position = data.xpos[headID][0] if headID != -1 else 0


        # print("Current control values (data.ctrl):", data.ctrl)
        
        # recuper les angles des joinds
        angle1 = data.qpos[idx1]
        angle2 = data.qpos[idx2]
        angle3 = data.qpos[idx3]
        angle4 = data.qpos[idx4]
        # print(f"Servo 1 = {angle1}, Servo 2 = {angle2}, Servo 3 = {angle3}, Servo 4 = {angle4}")
        
        if (
            abs(angle1 - data.ctrl[0]) < 1 and
            abs(angle2 - data.ctrl[1]) < 1 and
            abs(angle3 - data.ctrl[2]) < 1 and
            abs(angle4 - data.ctrl[3]) < 1
        ):
            
            num_try += 1
            print(f"\nTry number: {num_try}")
            head_current_position = data.xpos[headID][0] if headID != -1 else 0

            if head_current_position > head_last_position + 0.01:
                if len(current_positions) > 0:
                    add_to_dataset(current_positions, current_next_position, True)
                print("Snake has moved forward.")
            else:
                if len(current_positions) > 0:
                    add_to_dataset(current_positions, current_next_position, False)
                print("Snake did not move forward.")

            print("All servo angles match the control values within a delta of 0.1.")
            next_logits = snakemodel(positions.view(1, -1))  # Reshape to [1, num_positions * num_moteurs]
            # print("Next position from model:", next_position)

            # print("Next logits from model:", next_logits)

            current_positions = positions.clone()

            next_position = top_p_sampling(next_logits, p=0.9)
            print('\n\nSampled vocab indices (top-p):')
            print(next_position)

            current_next_position = next_position.clone()

            positions[:-1] = positions[1:].clone()
            positions[-1] = next_position.clone()

            print("Updated positions:", positions)

            is_moving = False
            

        timestep = timestep+0.05
        viewer.sync()
        time.sleep(0.05)

print("\n\nDataset collected:")
for i, (pos, next_pos, moved) in enumerate(dataset):
    print(f"Sample {i+1}:")
    print(f"  Positions: {pos}")
    print(f"  Next Position: {next_pos}")
    print(f"  Moved: {moved}")

print("\nTotal samples collected:", len(dataset))
print("Dataset:", dataset)

# Convert dataset to a consistent format
processed_dataset = []
for pos, next_pos, moved in dataset:
    processed_dataset.append({
        "positions": np.array(pos).tolist(),
        "next_position": np.array(next_pos).tolist(),
        "moved": moved
    })

# Save the processed dataset
import json
with open('snake_dataset.json', 'w') as f:
    json.dump(processed_dataset, f)

