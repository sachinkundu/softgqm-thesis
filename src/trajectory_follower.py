import time
import numpy as np
import modern_robotics as mr
from robosuite.utils.control_utils import orientation_error
from dm_robotics.transformations import transformations as tr


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


class TrajectoryFollower:
    def __init__(self, env, logger, headless=False):
        self.env = env
        self.logger = logger
        self.p_gain = 19
        self.ang_gain = 2.2
        self.headless = headless

    def follow(self, destination_hmat, eef_init_pose, grasp_action):
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
            angle_action = self.ang_gain * orientation_error(desired_next_pose, desired_pose)
            position_action = self.p_gain * (desired_pose[:-1, -1] - current_eef_position)

            action = np.append(np.hstack((position_action, angle_action)), grasp_action)
            obs, reward, done, _ = self.env.step(action.tolist())

            if not self.headless:
                self.env.render()

            step_end_time = time.time()

            self.logger.debug(f"Step {i} took: {step_end_time - step_time} s")

            last_obs = obs
            current_eef_position = last_obs['robot0_eef_pos']
            current_eef_quat = last_obs['robot0_eef_quat']

        self.logger.debug(f"Trajectory took: {time.time() - start_time} s")

        # Get and print final tracking performance.
        # pos_error, angle_error = get_final_errors(destination_hmat, last_obs)
        # self.logger.debug(f"Final angle error: {angle_error:.2f} deg")
        # self.logger.debug(f"Final pos   error: {np.linalg.norm(pos_error):.3f}")

        return last_obs
