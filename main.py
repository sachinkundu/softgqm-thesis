import cv2
import click
import mujoco
import logging
from pathlib import Path
import modern_robotics as mr
from src.envs import UnfoldCloth
from robosuite.utils.input_utils import *
from robosuite.controllers import load_controller_config
from dm_robotics.transformations import transformations as tr


def get_random_angle(a, b):
    theta_deg = (b - a) * np.random.sample() + a
    return np.deg2rad(theta_deg)


@click.command()
@click.option('--cloth', is_flag=True, help="include cloth in sim")
@click.option('--n', default=1, show_default=True, help="number of simulation runs")
@click.option('--debug', is_flag=True, default=False, show_default=True, help="debug logging")
@click.option('--show-sites', is_flag=True, default=False, help="include cloth in sim")
@click.option('--headless', is_flag=True, default=False, help="Run without rendering")
def main(cloth, n, debug, show_sites, headless):
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
    controller_config['kd'] = [24.495, 24.495, 24.495, 21.495, 21.495, 21.495]
    controller_config['kp'] = [150., 150., 150., 150., 150., 150.]

    # Load the desired controller
    options["controller_configs"] = controller_config

    # ('frontview', 'birdview', 'agentview', 'sideview', 'qdp', 'robot0_robotview', 'robot0_eye_in_hand')
    camera_to_use = "frontview"
    # initialize the task
    env: UnfoldCloth = suite.make(
        **options,
        has_renderer=not headless,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=2,
        include_cloth=cloth,
        logger=logging.getLogger(__name__),
        camera_depths=True,
        camera_names=camera_to_use,
        camera_segmentations="element",  # {None, instance, class, element}
        headless=headless
    )

    for run_no in range(n):

        logging.info("############################")
        logging.info(f"Trial No: {run_no + 1}")
        logging.info("############################")

        initial_state = env.reset()

        if not headless:
            agent_view_camera_id = env.sim.model.camera_name2id(camera_to_use)
            env.viewer.set_camera(camera_id=agent_view_camera_id)

        if show_sites:
            env.sim._render_context_offscreen.vopt.frame = mujoco.mjtFrame.mjFRAME_SITE

        if cloth:
            # start = time.time()
            # while time.time() < start + 60:
            #     env.sim.forward()

            pick_object_pose = tr.pos_quat_to_hmat(env.sim.data.body_xpos[env.sim.model.body_name2id("cloth_40")],
                                                   env.sim.data.body_xquat[env.sim.model.body_name2id("cloth_40")])
            logging.info(f"pick_object_pos: {pick_object_pose[:-1, -1]}")
        else:
            pick_object_pose = tr.pos_quat_to_hmat(initial_state['cube_pos'], initial_state['cube_quat'])

        angle = get_random_angle(-45, -30)
        pick_object_pose[:-1, :-1] = np.matmul(pick_object_pose[:-1, :-1],
                                               tr.rotation_y_axis(np.array([angle]), full=False))

        initial_pick_pose_angle = tr.quat_to_euler(tr.hmat_to_pos_quat(pick_object_pose)[1])[-1]
        logging.info(f"initial_pick_pose_angle at {np.rad2deg(initial_pick_pose_angle)}")

        last_obs = env.pick(pick_object_pose)
        
        current_eef_pose = tr.pos_quat_to_hmat(last_obs['robot0_eef_pos'], last_obs['robot0_eef_quat'])

        current_eef_angle = tr.quat_angle(last_obs['robot0_eef_quat'])

        logging.info(f"current_eef_angle: {np.rad2deg(current_eef_angle)}")

        new_ori = np.matmul(tr.rotation_z_axis(np.array([-initial_pick_pose_angle]), full=True), current_eef_pose)

        logging.info(f"new_ori_angle: {np.rad2deg(tr.quat_angle(tr.hmat_to_pos_quat(new_ori)[1]))}")

        new_ori = np.matmul(tr.rotation_y_axis(np.array([-angle]), full=True), new_ori)

        new_ori[:-1, -1] = pick_object_pose[:-1, -1] + np.array([0.2 * np.random.random_sample() - 0.1,
                                                                 0.2 * np.random.random_sample() - 0.1
                                                                    , 0])
        last_obs = env.place(new_ori, tr.pos_quat_to_hmat(last_obs['robot0_eef_pos'], last_obs['robot0_eef_quat']))

        if not cloth:
            logging.info(f"cube angle at {np.rad2deg(tr.quat_to_euler(last_obs['cube_quat']))}")

        env.ungrasp()

        j_traj = mr.JointTrajectory(env.robots[0].recent_qpos.current, env.robots[0].init_qpos, 5, 100, 5)

        for jnt in j_traj:
            env.robots[0].set_robot_joint_positions(jnt)
            if not headless:
                env.render()

        cv2.waitKey(1000)

    env.close()


if __name__ == "__main__":
    main()
