import numpy as np
import modern_robotics as mr
from dm_robotics.transformations import transformations as tr

from matplotlib import pyplot as plt

import time
from PIL import Image

N = 50
Tf = 0.1 * (N - 1)
np.set_printoptions(precision=3)


def _calculate_trajectory(destination_hmat, eef_init_hmat):
    trajectory = mr.CartesianTrajectory(eef_init_hmat, destination_hmat, Tf, N, 3)
    return trajectory


def get_final_errors(destination_hmat, final_obs):
    destination_angle = np.rad2deg(tr.quat_angle(tr.hmat_to_pos_quat(destination_hmat)[1]))
    achieved_angle = np.rad2deg(tr.quat_angle(final_obs['robot0_eef_quat']))
    angle_error = destination_angle - achieved_angle

    destination_pos = destination_hmat[:-1, -1]
    current_pos = final_obs['robot0_eef_pos']
    pos_error = destination_pos - current_pos

    return pos_error, angle_error


def compute_camera_matrix(renderer, data):
    """Returns the 3x4 camera matrix."""
    # If the camera is a 'free' camera, we get its position and orientation
    # from the scene data structure. It is a stereo camera, so we average over
    # the left and right channels. Note: we call `self.update()` in order to
    # ensure that the contents of `scene.camera` are correct.
    renderer.update_scene(data)
    pos = np.mean([camera.pos for camera in renderer.scene.camera], axis=0)
    z = -np.mean([camera.forward for camera in renderer.scene.camera], axis=0)
    y = np.mean([camera.up for camera in renderer.scene.camera], axis=0)
    rot = np.vstack((np.cross(y, z), y, z))
    fov = model.vis.global_.fovy

    # Translation matrix (4x4).
    translation = np.eye(4)
    translation[0:3, 3] = -pos

    # Rotation matrix (4x4).
    rotation = np.eye(4)
    rotation[0:3, 0:3] = rot

    # Focal transformation matrix (3x4).
    focal_scaling = (1. / np.tan(np.deg2rad(fov) / 2)) * renderer.height / 2.0
    focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]

    # Image matrix (3x3).
    image = np.eye(3)
    image[0, 2] = (renderer.width - 1) / 2.0
    image[1, 2] = (renderer.height - 1) / 2.0
    return image @ focal @ rotation @ translation


class TrajectoryFollower:
    def __init__(self, env, logger, no_ori=False):
        self.env = env
        self.logger = logger
        self.p_gain = 12
        self.ang_gain = 2.1
        self.no_ori = no_ori

    def follow(self, destination_hmat, eef_init_pose, grasp_action, angle_rotate=0):
        trajectory = _calculate_trajectory(destination_hmat, eef_init_pose)

        last_obs = None
        # Step over the trajectory one frame at a time
        start_time = time.time()
        current_eef_position = eef_init_pose[:-1, -1]
        current_eef_quat = tr.hmat_to_pos_quat(eef_init_pose)[1]

        skip_frames = 0
        for i, (desired_pose, desired_next_pose) in enumerate(
                zip(trajectory[skip_frames:], trajectory[skip_frames + 1:(len(trajectory) - skip_frames + 1)])):
            step_time = time.time()

            self.logger.debug(f"starting step: {i}")
            self.logger.debug(
                f"desired_ang: {tr.quat_angle(tr.hmat_to_pos_quat(desired_pose)[1])} current_eef_ang: {tr.quat_angle(current_eef_quat)}")

            angle_action = self.ang_gain * (
                        tr.quat_to_axisangle(tr.hmat_to_pos_quat(desired_next_pose)[1]) - tr.quat_to_axisangle(
                    tr.hmat_to_pos_quat(desired_pose)[1]))
            position_action = self.p_gain * (desired_pose[:-1, -1] - current_eef_position)

            action = np.append(np.hstack((position_action, angle_action)), grasp_action)
            obs, reward, done, _ = self.env.step(action.tolist())
            self.env.render()

            # compute_camera_matrix(self.env.viewer, self.env.sim.data)

            step_end_time = time.time()

            self.logger.debug(f"Step {i} took: {step_end_time - step_time} s")

            last_obs = obs
            current_eef_position = last_obs['robot0_eef_pos']
            current_eef_quat = last_obs['robot0_eef_quat']

        self.logger.info(f"Trajectory took: {time.time() - start_time} s")

        # Get and print final tracking performance.
        pos_error, angle_error = get_final_errors(destination_hmat, last_obs)
        self.logger.info(f"Final angle error: {angle_error:.2f} deg")
        self.logger.info(f"Final pos   error: {np.linalg.norm(pos_error):.3f}")

        return last_obs
