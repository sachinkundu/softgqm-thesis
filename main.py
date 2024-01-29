import numpy as np
import cv2
import click
import logging
from pathlib import Path
import modern_robotics as mr
from src.envs import UnfoldCloth
from robosuite.utils.input_utils import *
from robosuite.controllers import load_controller_config
from dm_robotics.transformations import transformations as tr

np.set_printoptions(precision=3)

N = 50
Tf = 0.1 * (N - 1)


def grasp_imp(env, state):
    for i in range(10):
        env.step([0, 0, 0, 0, 0, 0, state])
        env.render()


def grasp(env):
    grasp_imp(env, 1)


def ungrasp(env):
    grasp_imp(env, -1)


@click.command()
@click.option('--cloth', is_flag=True, help="include cloth in sim")
@click.option('--n', default=1, show_default=True, help="number of simulation runs")
@click.option('--debug', is_flag=True, default=False, show_default=True, help="debug logging")
def main(cloth, n, debug):

    if debug:
        log_level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(format='%(asctime)s - %(message)s', level=log_level)

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
        control_freq=10,
        include_cloth=cloth
    )

    for _ in range(n):

        initial_state = env.reset()
        env.viewer.set_camera(camera_id=0)

        eef_SE3 = tr.pos_quat_to_hmat(initial_state['robot0_eef_pos'], initial_state['robot0_eef_quat'])

        if cloth:
            cloth_SE3 = tr.pos_quat_to_hmat(initial_state['cloth_pos'], initial_state['cloth_quat'])
            cloth_in_eef_SE3 = np.matmul(mr.TransInv(eef_SE3), cloth_SE3)
            logging.info(f"cloth angle: {np.rad2deg(tr.quat_angle(tr.hmat_to_pos_quat(cloth_in_eef_SE3)[1]))}")
            trajectory_space_frame = mr.CartesianTrajectory(eef_SE3, cloth_SE3, Tf, N, 5)
            trajectory_eef_frame = mr.CartesianTrajectory(eef_SE3, cloth_in_eef_SE3, Tf, N, 5)

            cube_SE3 = tr.pos_quat_to_hmat(initial_state['cube_pos'], initial_state['cube_quat'])
            cube_in_eef_SE3 = np.matmul(mr.TransInv(eef_SE3), cube_SE3)
            logging.info(f"cube angle: {np.rad2deg(tr.quat_angle(tr.hmat_to_pos_quat(cube_in_eef_SE3)[1]))}")

            trajectory_space_frame = mr.CartesianTrajectory(eef_SE3, cube_SE3, Tf, N, 5)
            trajectory_eef_frame = mr.CartesianTrajectory(eef_SE3, cube_in_eef_SE3, Tf, N, 5)

        else:
            cube_SE3 = tr.pos_quat_to_hmat(initial_state['cube_pos'], initial_state['cube_quat'])
            cube_in_eef_SE3 = np.matmul(mr.TransInv(eef_SE3), cube_SE3)
            logging.info(f"cube angle: {np.rad2deg(tr.quat_angle(tr.hmat_to_pos_quat(cube_in_eef_SE3)[1]))}")

            trajectory_space_frame = mr.CartesianTrajectory(eef_SE3, cube_SE3, Tf, N, 5)
            trajectory_eef_frame = mr.CartesianTrajectory(eef_SE3, cube_in_eef_SE3, Tf, N, 5)

        p_gain = 25
        ang_gain = 3

        current_eef_pos = trajectory_space_frame[0][:-1, -1]
        last_obs = initial_state
        for i, (destination_pose, frame1_e, frame2_e) in enumerate(zip(trajectory_space_frame,
                                                                       trajectory_eef_frame,
                                                                       trajectory_eef_frame[1:])):

            logging.info(f"starting step: {i}")

            if np.linalg.norm(current_eef_pos - initial_state['cube_pos']) < 0.01:
                break

            frame2_pos = destination_pose[:-1, -1]
            repeat = 0
            while not np.allclose(frame2_pos, current_eef_pos, rtol=0.001, atol=0.001):
                logging.info(f"taking step: {i}")
                pos_diff = p_gain * (frame2_pos - current_eef_pos)

                frame1_e_ax_ang = tr.quat_to_axisangle(tr.hmat_to_pos_quat(frame1_e)[1])
                frame2_e_ax_ang = tr.quat_to_axisangle(tr.hmat_to_pos_quat(frame2_e)[1])

                if repeat == 0:
                    ang_diff = ang_gain * (frame2_e_ax_ang - frame1_e_ax_ang)
                else:
                    if repeat > 50:
                        break
                    ang_diff = np.zeros(shape=(3, ))
                repeat += 1
                action = np.array([pos_diff[0],
                                   pos_diff[1],
                                   pos_diff[2],
                                   ang_diff[0],
                                   ang_diff[1],
                                   ang_diff[2],
                                   -1])
                logging.debug(f"action: {action}")
                obs, reward, done, _ = env.step(action.tolist())
                current_eef_pos = obs['robot0_eef_pos']
                env.render()

                logging.debug(
                    f"eef_axis: {tr.quat_axis(obs['robot0_eef_quat'])} : eef_angle: {np.rad2deg(tr.quat_angle(obs['robot0_eef_quat']))}")
                logging.debug(f"cube_pos: {obs['cube_pos']} eef_pos: {obs['robot0_eef_pos']}")
                last_obs = obs

        if cloth:
            logging.info(f"cloth pos error: {np.linalg.norm(last_obs['cloth_pos'] - last_obs['robot0_eef_pos'])}")
            logging.info(
                f"cloth ang error: {np.rad2deg(tr.quat_angle(tr.hmat_to_pos_quat(cloth_in_eef_SE3)[1]) - tr.quat_angle(last_obs['cloth_quat']))}")
        else:
            logging.info(f"cube pos error: {np.linalg.norm(last_obs['cube_pos'] - last_obs['robot0_eef_pos'])}")
            logging.info(
                f"cube ang error: {np.rad2deg(tr.quat_angle(tr.hmat_to_pos_quat(cube_in_eef_SE3)[1]) - tr.quat_angle(last_obs['robot0_eef_quat']))}")

        cv2.waitKey(1000)
    env.close()


if __name__ == "__main__":
    main()
