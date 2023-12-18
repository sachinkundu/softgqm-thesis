from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from src.envs import UnfoldCloth

from pathlib import Path


def grasp_imp(env, state):
    for i in range(50):
        env.step([0, 0, 0, 0, 0, 0, state])
        env.render()


def grasp(env):
    grasp_imp(env, 1)


def ungrasp(env):
    grasp_imp(env, -1)


if __name__ == "__main__":

    # Create dict to hold options that will be passed to env creation call
    options = {
        "robots": "Panda",
        "env_name": UnfoldCloth.__name__.split(".")[-1],
        "asset_path": str((Path(__file__).parent / "assets").resolve())
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
    final_eef_pos = initial_eef_pos
    while distance > 0.01:
        action_x = (cube_pos - final_eef_pos)[0]
        action_y = (cube_pos - final_eef_pos)[1]
        action_z = (cube_pos - final_eef_pos)[2]

        action = [action_x,
                  action_y,
                  action_z,
                  0,
                  0,
                  0,
                  -1]
        obs, reward, done, _ = env.step(action)
        env.render()
        distance = np.linalg.norm(cube_pos - obs['robot0_eef_pos'])
        final_eef_pos = obs['robot0_eef_pos']

    print(f"eef_pose at grasp = {final_eef_pos}")

    grasp(env)

    # cube_height = cube_pos[2]
    # while cube_height < 1.1:
    #     obs, reward, done, _ = env.step([0, 0, 0.1, 0, 0, 0, 1])
    #     env.render()
    #     cube_height = obs['cube_pos'][2]
    #
    # for i in range(100):
    #     obs, reward, done, _ = env.step([0, 0.1, 0, 0, 0, 0, 1])
    #     env.render()
    #
    # while cube_height >= cube_pos[2]:
    #     obs, reward, done, _ = env.step([0, 0, -0.1, 0, 0, 0, 1])
    #     env.render()
    #     cube_height = obs['cube_pos'][2]
    #
    # ungrasp(env)
    #
    # for i in range(100):
    #     obs, reward, done, _ = env.step([0, 0, 0.1, 0, 0, 0, -1])
    #     env.render()
    #     final_eef_pos = obs['robot0_eef_pos']
    #
    # for i in range(100):
    #     obs, reward, done, _ = env.step([0, -0.1, 0, 0, 0, 0, -1])
    #     env.render()

    print(f"Final eef_pose = {final_eef_pos}")
