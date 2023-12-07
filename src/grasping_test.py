import numpy as np
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from robosuite.models.objects import MujocoXMLObject


class ClothObject(MujocoXMLObject):
    """
    Cloth object
    """

    def __init__(self, path, name):
        super().__init__(path, name=name, obj_type="collision", duplicate_collision_geoms=False)


if __name__ == "__main__":

    # Create dict to hold options that will be passed to env creation call
    options = {
        "robots": "Panda",
        "env_name": "Lift"
    }

    # Choose controller
    controller_name = "OSC_POSE"

    controller_config = load_controller_config(default_controller=controller_name)

    # Load the desired controller
    options["controller_configs"] = controller_config

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )
    initial_state = env.reset()
    print(f"cube_pos = {initial_state['cube_pos']}")
    print(f"Initial eef_pose = {initial_state['robot0_eef_pos']}")
    env.viewer.set_camera(camera_id=0)

    cube_pos = initial_state['cube_pos']
    initial_eef_pos = initial_state['robot0_eef_pos']

    distance = np.linalg.norm(cube_pos - initial_eef_pos)
    final_eef_pos = np.zeros_like(initial_eef_pos)
    while distance > 0.01:
        action = [(cube_pos[0] - initial_eef_pos[0]),
                  (cube_pos[1] - initial_eef_pos[1]),
                  (cube_pos[2] - initial_eef_pos[2]),
                  0,
                  0,
                  0,
                  -1]
        obs, reward, done, _ = env.step(action)
        env.render()
        distance = np.linalg.norm(cube_pos - obs['robot0_eef_pos'])
        final_eef_pos = obs['robot0_eef_pos']

    print(f"eef_pose at grasp = {final_eef_pos}")

    for i in range(100):
        obs, reward, done, _ = env.step([0, 0, 0, 0, 0, 0, 1])
        env.render()
        final_eef_pos = obs['robot0_eef_pos']

    print(f"Final eef_pose = {final_eef_pos}")

