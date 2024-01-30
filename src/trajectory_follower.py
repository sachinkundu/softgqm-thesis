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
        destination_in_eef = np.matmul(mr.TransInv(eef_init_hmat), destination_hmat)
        self.logger.info(f"cube angle: {np.rad2deg(tr.quat_angle(tr.hmat_to_pos_quat(destination_hmat)[1]))}")

        trajectory_space_frame = mr.CartesianTrajectory(eef_init_hmat, destination_hmat, Tf, N, 5)
        trajectory_eef_frame = mr.CartesianTrajectory(eef_init_hmat, destination_in_eef, Tf, N, 5)

        return trajectory_space_frame, trajectory_eef_frame, destination_in_eef

    def follow(self, start_state, eef_init_pose):

        pos_trajectory, ori_trajectory, destination_in_eef = self._calculate_trajectory(start_state, eef_init_pose)

        current_eef_pos = pos_trajectory[0][:-1, -1]
        last_obs = None
        for i, (desired_pose, frame1_e, frame2_e) in enumerate(zip(pos_trajectory,
                                                                   ori_trajectory,
                                                                   ori_trajectory[1:])):

            self.logger.info(f"starting step: {i}")

            if np.linalg.norm(current_eef_pos - start_state[:-1, -1]) < 0.001:
                break

            frame2_pos = desired_pose[:-1, -1]
            repeat = 0
            while not np.allclose(frame2_pos, current_eef_pos, rtol=0.001, atol=0.001):
                self.logger.info(f"taking step: {i}")
                pos_diff = self.p_gain * (frame2_pos - current_eef_pos)

                frame1_e_ax_ang = tr.quat_to_axisangle(tr.hmat_to_pos_quat(frame1_e)[1])
                frame2_e_ax_ang = tr.quat_to_axisangle(tr.hmat_to_pos_quat(frame2_e)[1])

                if repeat == 0:
                    ang_diff = self.ang_gain * (frame2_e_ax_ang - frame1_e_ax_ang)
                else:
                    if repeat > 50:
                        break
                    ang_diff = np.zeros(shape=(3,))
                repeat += 1
                action = np.array([pos_diff[0],
                                   pos_diff[1],
                                   pos_diff[2],
                                   ang_diff[0],
                                   ang_diff[1],
                                   ang_diff[2],
                                   -1])
                self.logger.debug(f"action: {action}")
                obs, reward, done, _ = self.env.step(action.tolist())
                self.env.render()
                current_eef_pos = obs['robot0_eef_pos']
                last_obs = obs

        destination_angle = tr.quat_angle(tr.hmat_to_pos_quat(destination_in_eef)[1])
        start_angle = tr.quat_angle(tr.hmat_to_pos_quat(start_state)[1])
        angle_error = np.rad2deg(destination_angle - start_angle)
        self.logger.info(f"Final angle error: {angle_error}")

        return last_obs
