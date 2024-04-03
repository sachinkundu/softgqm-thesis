import itertools
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import click
import cv2
import modern_robotics as mr
import mujoco
import robosuite.utils.camera_utils as rcu
import robosuite.utils.transform_utils as rtu
from dm_robotics.transformations import transformations as tr
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *

from src.envs import UnfoldCloth


def get_random_angle(a, b):
    theta_deg = (b - a) * np.random.sample() + a
    return np.deg2rad(theta_deg)


def run(options, headless=False, n_trials=1, show_sites=False, label=False, cloth=False):
    now = datetime.now().strftime("%d%m%Y-%H%M%S")
    # Choose controller
    controller_name = "OSC_POSE"

    controller_config = load_controller_config(default_controller=controller_name)
    controller_config['kd'] = [24.495, 24.495, 24.495, 21.495, 21.495, 21.495]
    controller_config['kp'] = [150., 150., 150., 150., 150., 150.]

    # Load the desired controller
    options["controller_configs"] = controller_config

    # ('frontview', 'birdview', 'agentview', 'sideview', 'qdp', 'robot0_robotview', 'robot0_eye_in_hand')
    camera_to_use = "robot0_robotview"
    # initialize the task
    env: UnfoldCloth = suite.make(
        **options,
        has_renderer=not headless,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=2,
        logger=logging.getLogger(__name__),
        camera_depths=True,
        camera_names=[camera_to_use, 'sideview', 'robot0_eye_in_hand'],
        render_camera=camera_to_use,
        camera_segmentations="element",  # {None, instance, class, element}
        headless=headless
    )

    for run_no in range(n_trials):

        try:
            if 'data' in options:
                angles = np.arange(-np.pi / 6, np.pi / 6 + np.pi / 18, step=np.pi / 18)
            else:
                angles = [0]

            combinations = itertools.product(['x', 'y'], angles)
            combinations = [(ax, ang) for ax, ang in combinations if ax == 'x' or ang != 0]
            for (axis, grasp_orientation) in combinations:
                initial_state = env.reset()

                if not headless:
                    agent_view_camera_id = env.sim.model.camera_name2id(camera_to_use)
                    env.viewer.set_camera(camera_id=agent_view_camera_id)
                    env.render()

                if show_sites:
                    # env.sim._render_context_offscreen.vopt.frame = mujoco.mjtFrame.mjFRAME_CONTACT
                    env.sim._render_context_offscreen.vopt.flags = mujoco.mjtVisFlag.mjVIS_CONTACTPOINT
                    # env.sim._render_context_offscreen.vopt.flags = mujoco.mjtVisFlag.mjVIS_CONTACTFORCE

                if label:
                    env.sim._render_context_offscreen.vopt.label = mujoco.mjtLabel.mjLABEL_BODY

                if cloth:
                    start = time.time()
                    while time.time() < start + 10:
                        action = np.zeros(shape=(env.action_dim,))
                        action[-1] = -1
                        env.step(action)
                        if not headless:
                            env.render()

                    cloth_id = get_highest_cloth_body(env, options['n_cloth'])
                    cloth_body_pos = env.sim.data.body_xpos[env.sim.model.body_name2id(f"cloth_{cloth_id}")]
                    cloth_body_quat = env.sim.data.body_xquat[env.sim.model.body_name2id(f"cloth_{cloth_id}")]

                    logging.debug(f"cloth_body_pos after settling: {cloth_body_pos}")

                    env.set_cloth_body_id(cloth_id)

                    pick_object_pose = rtu.make_pose(cloth_body_pos, tr.quat_to_mat(cloth_body_quat)[:-1, :-1])
                    logging.info("############################")
                    logging.info(f"Cloth: {options['n_cloth']} Axis: {axis} Orientation: {np.rad2deg(grasp_orientation):.0f}")
                    logging.info("############################")

                else:
                    pick_object_pose = rtu.make_pose(pixels_to_world(fake_policy(initial_state,
                                                                                 camera_to_use, env, cloth, headless),
                                                                     initial_state, camera_to_use, env),
                                                     tr.quat_to_mat(initial_state["cube_quat"])[:-1, :-1])

                optimal_pick_object_pose = optimal_grasp(pick_object_pose, axis, grasp_orientation)
                output_folder = Path(__file__).parent / "contact_data" / now
                output_folder.mkdir(exist_ok=True, parents=True)
                env.pick(optimal_pick_object_pose, axis, grasp_orientation, output_folder=output_folder)

                env.go_home()
                cv2.waitKey(1000)
        finally:
            env.close()


@click.command()
@click.option('--data', is_flag=True, default=False, help="collect data")
@click.option('--cloth', is_flag=True, default=False, help="run cloth sim")
@click.option('--kube', is_flag=True, default=False, help="run cube sim")
@click.option('--n-cloth', default=0, help="include cloth in sim")
@click.option('--n', default=1, show_default=True, help="number of simulation runs")
@click.option('--debug', is_flag=True, default=False, show_default=True, help="debug logging")
@click.option('--show-sites', is_flag=True, default=False, help="include cloth in sim")
@click.option('--headless', is_flag=True, default=False, help="Run without rendering")
@click.option('--label', is_flag=True, default=False, help="Show body labels")
def main(data, cloth, kube, n_cloth, n, debug, show_sites, headless, label):
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format='%(asctime)s - %(message)s', level=log_level)

    if cloth and not data and n_cloth == 0:
        logging.critical("Cannot run cloth sim without also specifying what type of cloth")
        sys.exit(-1)

    if data and cloth:
        for cloth_name in [25, 50, 100, 200]:
            # Create dict to hold options that will be passed to env creation call
            options = {
                "robots": "Panda",
                "env_name": UnfoldCloth.__name__.split(".")[-1],
                "n_cloth": cloth_name,
                "data": True
            }
            run(options, cloth=True, headless=headless)
    elif data and kube:
        options = {
            "robots": "Panda",
            "env_name": UnfoldCloth.__name__.split(".")[-1],
            "data": True
        }
        run(options, headless=headless)
    elif cloth:
        options = {
            "robots": "Panda",
            "env_name": UnfoldCloth.__name__.split(".")[-1],
            "n_cloth": n_cloth
        }
        run(options, cloth=True, headless=headless)
    elif kube:
        options = {
            "robots": "Panda",
            "env_name": UnfoldCloth.__name__.split(".")[-1],
        }
        run(options, headless=headless)
    else:
        logging.critical("No options specified for the run")
        sys.exit(-1)


def dest_jnt_traj(env, start_qpos, dest_qpos, headless):
    j_traj = mr.JointTrajectory(start_qpos, dest_qpos, 5, 100, 5)
    for jnt in j_traj:
        env.robots[0].set_robot_joint_positions(jnt)
        if not headless:
            env.render()


def optimal_grasp(pick_object_pose, axis, angle):
    if axis == 'y':
        pick_object_pose[:-1, :-1] = np.matmul(pick_object_pose[:-1, :-1],
                                               tr.rotation_y_axis(np.array([angle]), full=False))
    elif axis == 'x':
        pick_object_pose[:-1, :-1] = np.matmul(pick_object_pose[:-1, :-1],
                                               tr.rotation_x_axis(np.array([angle]), full=False))

    return pick_object_pose


def pixels_to_world(pixels, state, camera_name, env):
    depth_map = state["{}_depth".format(camera_name)][::-1]
    world_to_camera = rcu.get_camera_transform_matrix(
        sim=env.sim,
        camera_name=camera_name,
        camera_height=env.camera_heights[0],
        camera_width=env.camera_widths[0],
    )
    camera_to_world = np.linalg.inv(world_to_camera)

    depth_map = rcu.get_real_depth_map(sim=env.sim, depth_map=depth_map)
    estimated_obj_pos = rcu.transform_from_pixels_to_world(
        pixels=pixels,
        depth_map=depth_map,
        camera_to_world_transform=camera_to_world,
    )
    logging.debug(f"estimated_obj_pos: {estimated_obj_pos}")
    return estimated_obj_pos


def fake_policy(state, camera_to_use, env, cloth, headless):
    obj_pos = policy(cloth, env, state, headless)
    world_to_camera = rcu.get_camera_transform_matrix(
        sim=env.sim,
        camera_name=camera_to_use,
        camera_height=env.camera_heights[0],
        camera_width=env.camera_widths[0],
    )

    # transform object position into camera pixel
    obj_pixel = rcu.project_points_from_world_to_camera(
        points=obj_pos,
        world_to_camera_transform=world_to_camera,
        camera_height=env.camera_heights[0],
        camera_width=env.camera_widths[0],
    )

    return np.array([125, 200]) if cloth else obj_pixel


def get_highest_cloth_body(env, n_bodies):
    highest_height = 0.0
    highest_body = None

    for i in range(n_bodies):
        current_cloth_body_height = env.sim.data.body_xpos[env.sim.model.body_name2id(f"cloth_{i}")][2]
        logging.debug(f"highest_height: {current_cloth_body_height}, i: {i} , highest_body: {highest_body}")
        if current_cloth_body_height > highest_height:
            highest_height = current_cloth_body_height
            highest_body = i

    return highest_body


def get_nearest_body_to_pixel(pixels, state, camera_name, env):
    coordinates = pixels_to_world(pixels, state, camera_name, env)
    nearest_body_idx = None
    closest_distance = math.inf
    for i_body in range(200):
        current_cloth_body_pos = env.sim.data.body_xpos[env.sim.model.body_name2id(f"cloth_{i_body}")]
        current_distance = np.linalg.norm(coordinates - current_cloth_body_pos)
        if current_distance < closest_distance:
            closest_distance = current_distance
            nearest_body_idx = i_body

    return nearest_body_idx


def policy(cloth, env, initial_state, headless):
    if cloth:
        start = time.time()
        while time.time() < start + 60:
            env.sim.forward()
            if not headless:
                env.render()

        pick_object_pose = tr.pos_quat_to_hmat(initial_state['cloth_pos'], initial_state['cloth_quat'])
        logging.info(f"pick_object_pos: {pick_object_pose[:-1, -1]}")
    else:
        pick_object_pose = tr.pos_quat_to_hmat(initial_state['cube_pos'], initial_state['cube_quat'])

    return pick_object_pose[:-1, -1]


def optimal_place(angle, initial_pick_pose_angle, last_obs, pick_object_pose):
    current_eef_pose = tr.pos_quat_to_hmat(last_obs['robot0_eef_pos'], last_obs['robot0_eef_quat'])
    new_ori = np.matmul(tr.rotation_z_axis(np.array([-initial_pick_pose_angle]), full=True), current_eef_pose)
    new_ori = np.matmul(tr.rotation_y_axis(np.array([-angle]), full=True), new_ori)
    new_ori[:-1, -1] = pick_object_pose[:-1, -1] + np.array([0.2 * np.random.random_sample() - 0.1,
                                                             0.2 * np.random.random_sample() - 0.1, 0])
    return new_ori


if __name__ == "__main__":
    main()
