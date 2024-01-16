import sys

import numpy as np
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from src.envs import UnfoldCloth

from pathlib import Path

import modern_robotics as mr
from mujoco import mju_quat2Mat

import time

np.set_printoptions(precision=3)

N = 50
Tf = 0.1 * (N - 1)


def grasp_imp(env, state):
    for i in range(50):
        env.step([0, 0, 0, 0, 0, 0, state])
        env.render()


def grasp(env):
    grasp_imp(env, 1)


def ungrasp(env):
    grasp_imp(env, -1)


def to_deg(ang_r):
    return (180.0 * ang_r / np.pi)


def angles(u, v):
    return np.arccos(u.dot(v) / (np.linalg.norm(u) * np.linalg.norm(v))) if not np.allclose(u, v) else 0


def reached_cube(current_eef_pos, cube_pos):
    return np.linalg.norm(current_eef_pos - cube_pos) < 0.001


def main():
    # Create dict to hold options that will be passed to env creation call
    options = {
        "robots": "Panda",
        "env_name": UnfoldCloth.__name__.split(".")[-1],
        "asset_path": str((Path(__file__).parent / "assets").resolve())
    }

    # Choose controller
    controller_name = "OSC_POSE"

    controller_config = load_controller_config(default_controller=controller_name)
    # controller_config['uncouple_pos_ori'] = True

    # Load the desired controller
    options["controller_configs"] = controller_config

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20
    )
    initial_state = env.reset()
    env.viewer.set_camera(camera_id=0)

    cube_pos = initial_state['cube_pos']
    cube_rot = np.zeros(shape=(9, 1))
    mju_quat2Mat(res=cube_rot, quat=initial_state['cube_quat'])
    cube_SE3 = mr.RpToTrans(cube_rot.reshape((3, 3)), cube_pos)

    initial_eef_pos = initial_state['robot0_eef_pos']
    initial_eef_rot = np.zeros(shape=(9, 1))
    mju_quat2Mat(res=initial_eef_rot, quat=initial_state['robot0_eef_quat'])
    eef_SE3 = mr.RpToTrans(initial_eef_rot.reshape((3, 3)), initial_eef_pos)

    print(f"cube_pos: {cube_pos}")

    trajectory = mr.CartesianTrajectory(eef_SE3, cube_SE3, Tf, N, 5)

    current_eef_pos = initial_eef_pos

    for step_no, frame in enumerate(trajectory):
        print(f"Step_no: {step_no}")
        expected_eef_position = frame[:-1, 3]
        while not np.allclose(expected_eef_position, current_eef_pos, rtol=0.001, atol=0.001):
            print(f"step: {step_no} -- current: {current_eef_pos} -- expected: {expected_eef_position}")
            position_action = 30 * (expected_eef_position - current_eef_pos)
            action = [position_action[0],
                      position_action[1],
                      position_action[2],
                      0,
                      0,
                      0,
                      -1]
            obs, reward, done, _ = env.step(action)
            env.render()
            current_eef_pos = obs['robot0_eef_pos']

    print(f"final position error = {current_eef_pos - cube_pos} norm = {np.linalg.norm(current_eef_pos - cube_pos)}")

    # grasp(env)
    #
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


if __name__ == "__main__":
    main()
