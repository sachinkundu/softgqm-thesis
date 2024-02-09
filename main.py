import cv2
import click
import mujoco
import logging
from pathlib import Path
import modern_robotics as mr
import numpy as np

from src.envs import UnfoldCloth
from robosuite.utils.input_utils import *
from robosuite.controllers import load_controller_config
from dm_robotics.transformations import transformations as tr


@click.command()
@click.option('--cloth', is_flag=True, help="include cloth in sim")
@click.option('--n', default=1, show_default=True, help="number of simulation runs")
@click.option('--debug', is_flag=True, default=False, show_default=True, help="debug logging")
@click.option('--show-sites', is_flag=True, default=False, help="include cloth in sim")
@click.option('--no-ori', is_flag=True, default=False, help="just position control")
def main(cloth, n, debug, show_sites, no_ori):

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

    # Load the desired controller
    options["controller_configs"] = controller_config


    # initialize the task
    env: UnfoldCloth = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=2,
        include_cloth=cloth,
        logger=logging.getLogger(__name__),
        no_ori=no_ori
    )

    for run_no in range(n):

        logging.info("############################")
        logging.info(f"Trial No: {run_no + 1}")
        logging.info("############################")

        initial_state = env.reset()
        env.viewer.set_camera(camera_id=2)

        if show_sites:
            env.sim._render_context_offscreen.vopt.frame = mujoco.mjtFrame.mjFRAME_SITE

        if cloth:
            pick_object_pose = tr.pos_quat_to_hmat(initial_state['cloth_pos'], initial_state['cloth_quat'])
            pick_object_pose = tr.pos_quat_to_hmat(initial_state['cube_pos'], initial_state['cube_quat'])
        else:
            pick_object_pose = tr.pos_quat_to_hmat(initial_state['cube_pos'], initial_state['cube_quat'])

        eef_initial_pose = tr.pos_quat_to_hmat(initial_state['robot0_eef_pos'], initial_state['robot0_eef_quat'])

        eef_pose_copy = eef_initial_pose.copy()

        last_obs = env.reach(pick_object_pose, eef_initial_pose)

        env.grasp()

        last_obs = env.lift(tr.pos_quat_to_hmat(last_obs['robot0_eef_pos'], last_obs['robot0_eef_quat']))

        # a, b = -30, 30
        # theta_deg = (b - a) * np.random.random_sample() + a
        # logging.info(f"random rotation of : {theta_deg}")
        new_ori = np.matmul(tr.rotation_z_axis(np.array([np.deg2rad(np.deg2rad(45))]), False), pick_object_pose[:-1, :-1])
        # new_ori = np.matmul(tr.rotation_y_axis(np.array([theta_deg]), False), pick_object_pose[:-1, :-1])
        place_hmat = mr.RpToTrans(new_ori, pick_object_pose[:-1, -1] + np.array([0.2 * np.random.random_sample() - 0.1,
                                                                                 0.2 * np.random.random_sample() - 0.1
                                                                                    , 0]))
        last_obs = env.place(place_hmat, tr.pos_quat_to_hmat(last_obs['robot0_eef_pos'], last_obs['robot0_eef_quat']))

        # place_hmat = pick_object_pose
        # last_obs = env.place(place_hmat, tr.pos_quat_to_hmat(last_obs['robot0_eef_pos'], last_obs['robot0_eef_quat']))

        env.ungrasp()

        assert (np.allclose(eef_pose_copy, eef_initial_pose))

        env.home(eef_initial_pose, place_hmat)

        cv2.waitKey(1000)

    env.close()


if __name__ == "__main__":
    main()
