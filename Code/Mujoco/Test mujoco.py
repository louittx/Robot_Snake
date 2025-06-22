import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path(r"C:\Users\Utilisateur\Desktop\python\Mujoco\Snake_Test.xml")
data = mujoco.MjData(model)

#cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
cube_id2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Head")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        #data.ctrl[0] = -3.0  # fait rouler le cube
        mujoco.mj_step(model, data)

        pos = data.xpos[cube_id2][:3]
        print(f"Position du cube : x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")

        viewer.sync()
        time.sleep(0.01)