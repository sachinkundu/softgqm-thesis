import numpy as np
import modern_robotics as mr
from dm_robotics.transformations import transformations as tr

import time

N = 50
Tf = 0.1 * (N - 1)
np.set_printoptions(precision=3)


def _calculate_trajectory(destination_hmat, eef_init_hmat):
    trajectory = mr.CartesianTrajectory(eef_init_hmat, destination_hmat, Tf, N, 5)
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
    def __init__(self, env, logger, no_ori=False):
        self.env = env
        self.logger = logger
        self.p_gain = 20
        self.ang_gain = 2.5
        self.no_ori = no_ori

    def follow(self, destination_hmat, eef_init_pose, grasp_action, angle_rotate=0):

        trajectory = _calculate_trajectory(destination_hmat, eef_init_pose)

        last_obs = None
        # Step over the trajectory one frame at a time
        start_time = time.time()
        for i, (desired_pose, desired_next_pose) in enumerate(zip(trajectory, trajectory[1:])):
            step_time = time.time()
            self.logger.debug(f"starting step: {i}")

            position_action = 20 * (desired_next_pose[:-1, -1] - desired_pose[:-1, -1])
            angle_action = 2 * (tr.quat_to_axisangle(tr.hmat_to_pos_quat(desired_next_pose)[1]) - tr.quat_to_axisangle(tr.hmat_to_pos_quat(desired_pose)[1]))

            # angle_action = np.zeros(shape=(3, ))

            action = np.append(np.hstack((position_action, angle_action)), grasp_action)
            obs, reward, done, _ = self.env.step(action.tolist())
            self.env.render()
            step_end_time = time.time()

            self.logger.debug(f"Step {i} took: {step_end_time - step_time} s")
            last_obs = obs

        self.logger.info(f"Trajectory took: {time.time() - start_time} s")

        # Get and print final tracking performance.
        pos_error, angle_error = get_final_errors(destination_hmat, last_obs)
        self.logger.info(f"Final angle error: {angle_error:.2f} deg")
        self.logger.info(f"Final pos   error: {np.linalg.norm(pos_error):.3f}")

        return last_obs
