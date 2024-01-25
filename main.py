import numpy as np
import logging

DEBUG = False
log_level = logging.DEBUG if DEBUG else logging.INFO
logging.basicConfig(format='%(asctime)s - %(message)s', level=log_level)

import cv2
from pathlib import Path
import modern_robotics as mr
from src.envs import UnfoldCloth
from robosuite.utils.input_utils import *
from robosuite.controllers import load_controller_config
from dm_robotics.transformations import transformations as tr

np.set_printoptions(precision=3)

N = 50
Tf = 0.1 * (N - 1)

no_of_simulations = 1

def grasp_imp(env, state):
    for i in range(50):
        env.step([0, 0, 0, 0, 0, 0, state])
        env.render()


def grasp(env):
    grasp_imp(env, 1)


def ungrasp(env):
    grasp_imp(env, -1)


def to_deg(ang_r):
    return 180.0 * ang_r / np.pi


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
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=10
    )

    for _ in range(no_of_simulations):

        initial_state = env.reset()
        env.viewer.set_camera(camera_id=0)

        cube_SE3 = tr.pos_quat_to_hmat(initial_state['cube_pos'], initial_state['cube_quat'])
        eef_SE3 = tr.pos_quat_to_hmat(initial_state['robot0_eef_pos'], initial_state['robot0_eef_quat'])
        cube_in_eef_SE3 = np.matmul(mr.TransInv(eef_SE3), cube_SE3)

        trajectory_space_frame = mr.CartesianTrajectory(eef_SE3, cube_SE3, Tf, N, 5)
        trajectory_eef_frame = mr.CartesianTrajectory(eef_SE3, cube_in_eef_SE3, Tf, N, 5)

        for frame1_s, frame2_s, frame1_e, frame2_e in zip(trajectory_space_frame, trajectory_space_frame[1:],
                                                          trajectory_eef_frame, trajectory_eef_frame[1:]):
            frame1_pos = frame1_s[:-1, -1]
            frame2_pos = frame2_s[:-1, -1]

            pos_diff = 22 * (frame2_pos - frame1_pos)

            frame1_e_ax_ang = tr.quat_to_axisangle(tr.hmat_to_pos_quat(frame1_e)[1])
            frame2_e_ax_ang = tr.quat_to_axisangle(tr.hmat_to_pos_quat(frame2_e)[1])

            # print(f"frame1_ax_ang: {frame1_ax_ang} frame2_ax_ang: {frame2_ax_ang}")

            ang_diff = 3.82 * (frame2_e_ax_ang - frame1_e_ax_ang)

            action = np.array([pos_diff[0],
                               pos_diff[1],
                               pos_diff[2],
                               ang_diff[0],
                               ang_diff[1],
                               ang_diff[2],
                               -1])
            # action = np.where(action > 0.001, action, 0)
            obs, reward, done, _ = env.step(action.tolist())
            env.render()

            logging.debug(f"eef_axis: {tr.quat_axis(obs['robot0_eef_quat'])} : eef_angle: {tr.quat_angle(obs['robot0_eef_quat'])}")
            logging.debug(f"cube_pos: {obs['cube_pos']} eef_pos: {obs['robot0_eef_pos']}")

        cv2.waitKey(1000)
    # current_eef_pos = initial_eef_pos
    #
    # for step_no, frame in enumerate(trajectory):
    #     print(f"Step_no: {step_no}")
    #     expected_eef_position = frame[:-1, 3]
    #     while not np.allclose(expected_eef_position, current_eef_pos, rtol=0.001, atol=0.001):
    #         print(f"step: {step_no} -- current: {current_eef_pos} -- expected: {expected_eef_position}")
    #         position_action = 30 * (expected_eef_position - current_eef_pos)
    #         action = [position_action[0],
    #                   position_action[1],
    #                   position_action[2],
    #                   0,
    #                   0,
    #                   0,
    #                   -1]
    #         obs, reward, done, _ = env.step(action)
    #         env.render()
    #         current_eef_pos = obs['robot0_eef_pos']
    #
    # print(f"final position error = {current_eef_pos - cube_pos} norm = {np.linalg.norm(current_eef_pos - cube_pos)}")

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
    env.close()


if __name__ == "__main__":
    main()
