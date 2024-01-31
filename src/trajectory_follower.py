import numpy as np
import modern_robotics as mr
from dm_robotics.transformations import transformations as tr

N = 50
Tf = 0.1 * (N - 1)
np.set_printoptions(precision=3)


class TrajectoryFollower:
    def __init__(self, env, logger):
        self.env = env
        self.logger = logger
        self.p_gain = 25
        self.ang_gain = 3

    def _calculate_trajectory(self, destination_hmat, eef_init_hmat):
        self.logger.info(f"dest angle: {np.rad2deg(tr.quat_angle(tr.hmat_to_pos_quat(destination_hmat)[1]))}")

        trajectory = mr.CartesianTrajectory(eef_init_hmat, destination_hmat, Tf, N, 5)

        return trajectory

    def follow(self, destination_hmat, eef_init_pose, grasp_action):

        trajectory = self._calculate_trajectory(destination_hmat, eef_init_pose)

        current_eef_position = trajectory[0][:-1, -1]
        last_obs = None
        for i, (desired_pose, desired_next_pose) in enumerate(zip(trajectory, trajectory[1:])):

            self.logger.debug(f"starting step: {i}")

            if np.linalg.norm(current_eef_position - destination_hmat[:-1, -1]) < 0.001:
                break

            desired_position = desired_pose[:-1, -1]
            while not np.allclose(desired_position, current_eef_position, rtol=0.001, atol=0.001):
                self.logger.debug(f"taking step: {i}")
                pos_diff = self.p_gain * (desired_position - current_eef_position)

                frame1_e_ax_ang = tr.quat_to_axisangle(tr.hmat_to_pos_quat(desired_pose)[1])
                frame2_e_ax_ang = tr.quat_to_axisangle(tr.hmat_to_pos_quat(desired_next_pose)[1])

                ang_diff = self.ang_gain * (frame2_e_ax_ang - frame1_e_ax_ang)

                action = np.array([pos_diff[0],
                                   pos_diff[1],
                                   pos_diff[2],
                                   ang_diff[0],
                                   ang_diff[1],
                                   ang_diff[2],
                                   grasp_action])
                self.logger.debug(f"action: {action}")
                obs, reward, done, _ = self.env.step(action.tolist())
                self.env.render()
                current_eef_position = obs['robot0_eef_pos']
                last_obs = obs

        desired_angle = np.rad2deg(tr.quat_angle(tr.hmat_to_pos_quat(destination_hmat)[1]))
        achieved_angle = np.rad2deg(tr.quat_angle(last_obs['robot0_eef_quat']))
        angle_error = desired_angle - achieved_angle

        destination_pos = destination_hmat[:-1, -1]
        current_pos = last_obs['robot0_eef_pos']
        pos_error = destination_pos - current_pos
        self.logger.info(f"Final angle error: {angle_error}")
        self.logger.info(f"Final pos   error: {np.linalg.norm(pos_error)}")
        return last_obs, pos_error, angle_error
