import numpy as np
import modern_robotics as mr
from dm_robotics.transformations import transformations as tr

N = 50
Tf = 0.1 * (N - 1)
np.set_printoptions(precision=3)


def position_match(pos1, pos2):
    return np.allclose(pos1, pos2, rtol=0.001, atol=0.001)


def angle_match(desired_angle, current_eef_angle, logger, no_ori=False):
    desired_angle_deg = np.rad2deg(desired_angle)
    current_eef_angle_deg = np.rad2deg(current_eef_angle)
    logger.debug(f"desired_angle_deg: {desired_angle_deg:.2f} current_eef_angle_deg: {current_eef_angle_deg:.2f} close: {np.allclose(desired_angle_deg, current_eef_angle_deg, rtol=1.0, atol=1.0)}")
    return no_ori or abs(desired_angle_deg - current_eef_angle_deg) < 3.6/2


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
        self.ang_gain = 2
        self.no_ori = no_ori

    def position_action(self, desired_position, current_eef_position):
        position_action = np.zeros(shape=(3,))
        if not position_match(desired_position, current_eef_position):
            position_action = self.p_gain * (desired_position - current_eef_position)

        return position_action

    def angle_action(self, desired_pose, current_eef_angle, logger, no_ori):
        desired_axis = tr.quat_axis(tr.hmat_to_pos_quat(desired_pose)[1])
        desired_angle = tr.quat_angle(tr.hmat_to_pos_quat(desired_pose)[1])

        angle_action = np.zeros(shape=(3,))
        if not angle_match(desired_angle, current_eef_angle, logger, no_ori):
            angle_action = self.ang_gain * desired_axis * (desired_angle - current_eef_angle)
        logger.debug(f"desired_axis: {desired_axis} desired_angle: {np.rad2deg(desired_angle):.2f} current_eef_angle: {np.rad2deg(current_eef_angle):.2f}")
        return angle_action

    def follow(self, destination_hmat, eef_init_pose, grasp_action):

        trajectory = _calculate_trajectory(destination_hmat, eef_init_pose)

        # Initial state is the eef initial pose
        current_eef_position = eef_init_pose[:-1, -1]
        current_eef_angle = tr.quat_angle(tr.hmat_to_pos_quat(eef_init_pose)[1])

        last_obs = None
        # Step over the trajectory one frame at a time
        for i, desired_pose in enumerate(trajectory[1:]):

            self.logger.debug(f"starting step: {i}")

            desired_position = desired_pose[:-1, -1]
            destination_angle = tr.quat_angle(tr.hmat_to_pos_quat(desired_pose)[1])

            # Keep to this frame till both position and orientation are close enough
            # Keep track of how many times the reconciliation loop runs
            repeat = 1
            while not position_match(desired_position, current_eef_position) or not angle_match(destination_angle,
                                                                                                current_eef_angle,
                                                                                                self.logger,
                                                                                                self.no_ori):
                self.logger.debug(f"""taking step: {i} - {repeat} time{"" if repeat == 1 else "s"}""")

                position_action = self.position_action(desired_position, current_eef_position)
                angle_action = self.angle_action(desired_pose, current_eef_angle, self.logger, self.no_ori)

                action = np.append(np.hstack((position_action, angle_action)), grasp_action)

                self.logger.debug(f"action: {action}")
                # Take action in the environment and render the scene
                obs, reward, done, _ = self.env.step(action.tolist())
                self.env.render()

                # Sense the environment and feedback the readings to trajectory control
                current_eef_position = obs['robot0_eef_pos']
                current_eef_angle = tr.quat_angle(obs['robot0_eef_quat'])

                last_obs = obs
                repeat += 1  # Increment repeat counter

                if repeat > 50:
                    break

        # Get and print final tracking performance.
        pos_error, angle_error = get_final_errors(destination_hmat, last_obs)
        self.logger.info(f"Final angle error: {angle_error:.2f} deg")
        self.logger.info(f"Final pos   error: {np.linalg.norm(pos_error):.3f}")

        return last_obs
