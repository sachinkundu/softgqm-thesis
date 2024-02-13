import numpy as np
import modern_robotics as mr
from dm_robotics.transformations import transformations as tr

from matplotlib import pyplot as plt

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
        self.ang_gain = 2.3
        self.no_ori = no_ori

    def follow(self, destination_hmat, eef_init_pose, grasp_action, angle_rotate=0):

        trajectory = _calculate_trajectory(destination_hmat, eef_init_pose)

        last_obs = None
        # Step over the trajectory one frame at a time
        start_time = time.time()
        current_eef_position = eef_init_pose[:-1, -1]
        current_eef_quat = tr.hmat_to_pos_quat(eef_init_pose)[1]
        # ang_gain = self.ang_gain * abs(tr.quat_angle(tr.hmat_to_pos_quat(destination_hmat)[1]) - tr.quat_angle(tr.hmat_to_pos_quat(eef_init_pose)[1]))

        desired_pos = []
        eef_pos = []

        desired_y = []
        eef_y = []

        for i, (desired_pose, desired_next_pose) in enumerate(zip(trajectory, trajectory[1:])):
            step_time = time.time()

            self.logger.debug(f"starting step: {i}")

            self.logger.info(f"desired_ang: {tr.quat_angle(tr.hmat_to_pos_quat(desired_pose)[1])} current_eef_ang: {tr.quat_angle(current_eef_quat)}")

            angle_action = self.ang_gain * (tr.quat_to_axisangle(tr.hmat_to_pos_quat(desired_next_pose)[1]) - tr.quat_to_axisangle(tr.hmat_to_pos_quat(desired_pose)[1]))
            # ang_diff = tr.quat_angle(tr.hmat_to_pos_quat(desired_pose)[1]) - tr.quat_angle(current_eef_quat)
            # angle_action = self.ang_gain * tr.quat_axis(tr.hmat_to_pos_quat(desired_pose)[1]) * ang_diff
            repeat = 0
            while not np.allclose(desired_pose[:-1, -1], current_eef_position, rtol=0.005, atol=0.005):

                desired_pos.append(desired_pose[:-1, -1])
                eef_pos.append(current_eef_position)

                desired_y.append(tr.quat_to_euler(tr.hmat_to_pos_quat(desired_pose)[1]))
                eef_y.append(tr.quat_to_euler(current_eef_quat))

                position_action = self.p_gain * (desired_pose[:-1, -1] - current_eef_position)

                if repeat > 0:
                    angle_action = np.zeros(shape=(3, ))

                action = np.append(np.hstack((position_action, angle_action)), grasp_action)
                obs, reward, done, _ = self.env.step(action.tolist())
                self.env.render()
                step_end_time = time.time()

                self.logger.debug(f"Step {i} took: {step_end_time - step_time} s")
                last_obs = obs
                current_eef_position = last_obs['robot0_eef_pos']

                current_eef_quat = last_obs['robot0_eef_quat']

                # self.logger.info(f"desired angle: {np.rad2deg(tr.quat_angle(tr.hmat_to_pos_quat(desired_next_pose)[1]))} current angle: {np.rad2deg(current_eef_angle)}")
                # self.logger.info(
                #     f"diff: {np.rad2deg(tr.quat_angle(tr.hmat_to_pos_quat(desired_next_pose)[1])) - np.rad2deg(current_eef_angle)}")
                # self.logger.info(f"pos error: {np.linalg.norm(desired_next_pose[:-1, -1] - last_obs['robot0_eef_pos']):.3f}")
                repeat += 1
                if repeat > 50:
                    break

        self.logger.info(f"Trajectory took: {time.time() - start_time} s")

        # self.logger.info(f"{eef_pos}")

        # plt.plot(list(range(len(desired_pos))), (np.array(desired_pos)[:, 2]), 'g^', list(range(len(eef_pos))), (np.array(eef_pos)[:, 2]), 'g-')
        # plt.show()
        #
        # plt.plot(list(range(len(desired_y))), (np.array(desired_y)[:, 2]), 'g^', list(range(len(eef_y))), (np.array(eef_y)[:, 2]))
        # plt.show()


        # Get and print final tracking performance.
        pos_error, angle_error = get_final_errors(destination_hmat, last_obs)
        self.logger.info(f"Final angle error: {angle_error:.2f} deg")
        self.logger.info(f"Final pos   error: {np.linalg.norm(pos_error):.3f}")

        return last_obs
