import numpy as np
import modern_robotics as mr
from dm_robotics.transformations import transformations as tr

N = 50
Tf = 0.1 * (N - 1)
np.set_printoptions(precision=3)


def position_match(pos1, pos2):
    return np.allclose(pos1, pos2, rtol=0.001, atol=0.001)


def orientation_match(rmat1, rmat2):
    return np.allclose(np.matmul(rmat1, rmat2.T), np.eye(3), rtol=0.01, atol=0.01)


class TrajectoryFollower:
    def __init__(self, env, logger):
        self.env = env
        self.logger = logger
        self.p_gain = 25
        self.ang_gain = 1

    def _calculate_trajectory(self, destination_hmat, eef_init_hmat):
        self.logger.info(f"dest angle: {np.rad2deg(tr.quat_angle(tr.hmat_to_pos_quat(destination_hmat)[1]))}")

        trajectory = mr.CartesianTrajectory(eef_init_hmat, destination_hmat, Tf, N, 5)

        return trajectory

    def follow(self, destination_hmat, eef_init_pose, grasp_action):

        trajectory = self._calculate_trajectory(destination_hmat, eef_init_pose)

        current_eef_position = trajectory[0][:-1, -1]
        current_eef_hmat = eef_init_pose
        current_eef_orientation_ax_ang = tr.quat_to_axisangle(tr.hmat_to_pos_quat(current_eef_hmat)[1])
        current_eef_orientation_ang = tr.quat_angle(tr.hmat_to_pos_quat(current_eef_hmat)[1])
        last_obs = None
        for i, (desired_pose, desired_next_pose) in enumerate(zip(trajectory, trajectory[1:])):

            self.logger.info(f"starting step: {i}")
            if np.linalg.norm(current_eef_position - destination_hmat[:-1, -1]) < 0.001:
                break

            desired_position = desired_pose[:-1, -1]
            step_repeat = False
            last_angle_command = np.zeros(shape=(3,))
            frame1_e_ang = tr.quat_angle(tr.hmat_to_pos_quat(desired_pose)[1])
            while not position_match(desired_position, current_eef_position) and not np.allclose(frame1_e_ang, current_eef_orientation_ang, rtol=0.01, atol=0.01):
                self.logger.info(f"taking step: {i}")
                pos_diff = self.p_gain * (desired_position - current_eef_position)

                frame1_e_ax_ang = tr.quat_to_axisangle(tr.hmat_to_pos_quat(desired_pose)[1])
                frame2_e_ax_ang = tr.quat_to_axisangle(tr.hmat_to_pos_quat(desired_next_pose)[1])

                frame1_e_ax = tr.quat_axis(tr.hmat_to_pos_quat(desired_pose)[1])
                frame1_e_ang = tr.quat_angle(tr.hmat_to_pos_quat(desired_pose)[1])

                if not step_repeat:
                    ang_diff = self.ang_gain * (frame2_e_ax_ang - frame1_e_ax_ang)
                    new_ax_ang = frame1_e_ax * current_eef_orientation_ang
                    ang_diff = self.ang_gain * (frame2_e_ax_ang - new_ax_ang)

                    # step_repeat = True
                    last_angle_command = ang_diff
                else:
                    ang_diff = last_angle_command

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
                current_eef_orientation = obs['robot0_eef_quat']
                current_eef_orientation_ax_ang = tr.quat_to_axisangle(current_eef_orientation)
                current_eef_hmat = tr.pos_quat_to_hmat(current_eef_position, current_eef_orientation)
                current_eef_orientation_ang = tr.quat_angle(current_eef_orientation)

                self.logger.info(f"frame1_e_ax_ang : {frame1_e_ax_ang}")
                self.logger.info(f"new    axis:      {new_ax_ang}")

                last_obs = obs

        desired_angle = np.rad2deg(tr.quat_angle(tr.hmat_to_pos_quat(destination_hmat)[1]))
        achieved_angle = np.rad2deg(tr.quat_angle(last_obs['robot0_eef_quat']))
        angle_error = desired_angle - achieved_angle

        destination_pos = destination_hmat[:-1, -1]
        current_pos = last_obs['robot0_eef_pos']
        pos_error = destination_pos - current_pos
        self.logger.info(f"Final angle error: {angle_error}")
        self.logger.info(f"Final pos   error: {np.linalg.norm(pos_error)}")
        return last_obs
