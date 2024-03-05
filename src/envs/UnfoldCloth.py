import numpy as np
from pathlib import Path
from collections import OrderedDict

from typing import List

import modern_robotics as mr
from robosuite.models.arenas import TableArena
import robosuite.utils.transform_utils as rtu
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.models.objects import BoxObject, MujocoObject
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from src.ClothObject import ClothObject
from src.trajectory_follower import TrajectoryFollower

from dm_robotics.transformations import transformations as tr


class UnfoldCloth(SingleArmEnv):
    """
    Copied and modified from Lift environment, so check that for all the documentation of parameters.
    """

    def __init__(
            self,
            robots,
            env_configuration="default",
            controller_configs=None,
            gripper_types="default",
            initialization_noise="default",
            table_full_size=(0.8, 0.8, 0.05),
            table_friction=(1.0, 5e-3, 1e-4),
            use_camera_obs=False,
            use_object_obs=True,
            reward_scale=1.0,
            reward_shaping=False,
            placement_initializer=None,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera="frontview",
            render_collision_mesh=False,
            render_visual_mesh=True,
            render_gpu_device_id=0,
            control_freq=10,
            horizon=1000,
            ignore_done=False,
            hard_reset=True,
            camera_names="agentview",
            camera_heights=256,
            camera_widths=256,
            camera_depths=False,
            camera_segmentations=None,  # {None, instance, class, element}
            renderer="mujoco",
            renderer_config=None,
            asset_path=None,
            include_cloth=False,
            logger=None,
            headless=False
    ):
        # settings for table-top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        #
        self.asset_path = asset_path
        self.include_cloth = include_cloth
        self.logger = logger
        self.grasp_state = -1  # Not grasping to start with
        self.headless = headless
        self.last_obs = None
        self.cloth_body_id = None
        self.done = False
        self.cloth_body_pos = np.zeros(shape=(3, ))

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

        self.trajectory_follower = TrajectoryFollower(self, self.logger, self.headless)

    def set_cloth_body_id(self, body_id):
        self.cloth_body_pos = self.sim.data.body_xpos[self.sim.model.body_name2id(f"cloth_{body_id}")]
        self.cloth_body_id = body_id

    def reward(self, action=None):
        if self.include_cloth:
            if self.done:
                bottom_left = self.sim.data.body_xpos[self.sim.model.body_name2id(f"cloth_{self.cloth_corner_vertices[0]}")]
                top_left = self.sim.data.body_xpos[self.sim.model.body_name2id(f"cloth_{self.cloth_corner_vertices[1]}")]
                bottom_right = self.sim.data.body_xpos[
                    self.sim.model.body_name2id(f"cloth_{self.cloth_corner_vertices[2]}")]
                l1 = np.linalg.norm(bottom_left - top_left)
                l2 = np.linalg.norm(bottom_left - bottom_right)
                r = np.round(l1 * l2 * 1000, decimals=2)
                return r    # Area of a rectangular cloth is the reward - in mm
            else:
                return 0
        else:
            return 1.0

    def _create_cube(self) -> MujocoObject:
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cube = BoxObject(
            name="cube",
            size_min=[0.020, 0.020, 0.020],  # [0.015, 0.015, 0.015],
            size_max=[0.022, 0.022, 0.022],  # [0.018, 0.018, 0.018]),
            solimp=[0.99, 0.99, 0.01],
            solref=[0.01, 1],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )

    def _create_cloth(self) -> MujocoObject:
        self.cloth_corner_vertices = np.array([0, 4, 20, 24])
        return ClothObject(str((Path(self.asset_path) / "cloth_simple_long_expanded_baked.xml").resolve()),
                           "cloth")

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table-top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest

        if self.include_cloth:
            self.cloth = self._create_cloth()
            mujoco_objects: List[MujocoObject] = np.array([self.cloth])
        else:
            self._create_cube()
            mujoco_objects: List[MujocoObject] = np.array([self.cube])

            # Create placement initializer
            if self.placement_initializer is not None:
                self.placement_initializer.reset()
                self.placement_initializer.add_objects(mujoco_objects)
            else:
                self.placement_initializer = UniformRandomSampler(
                    name="ObjectSampler",
                    mujoco_objects=mujoco_objects,
                    x_range=[0, 0.15],
                    y_range=[-0.2, 0.2],
                    rotation=(-np.pi / 4, np.pi / 4),
                    rotation_axis='z',
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    z_offset=0.01,
                )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=mujoco_objects
        )

    def reset(self):
        self.last_obs = super().reset()
        return self.last_obs

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        if not self.include_cloth:
            self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            sensors = []
            if self.include_cloth:
                @sensor(modality=modality)
                def cloth_pos(obs_cache):
                    return np.array(self.sim.data.body_xpos[self.cloth_body_id])

                @sensor(modality=modality)
                def cloth_quat(obs_cache):
                    return np.array(self.sim.data.body_xquat[self.cloth_body_id])

                sensors.extend([cloth_pos, cloth_quat])
            else:
                # cube-related observables
                @sensor(modality=modality)
                def cube_pos(obs_cache):
                    return np.array(self.sim.data.body_xpos[self.cube_body_id])

                @sensor(modality=modality)
                def cube_quat(obs_cache):
                    return np.array(self.sim.data.body_xquat[self.cube_body_id])

                @sensor(modality=modality)
                def gripper_to_cube_pos(obs_cache):
                    return (
                        obs_cache[f"{pf}eef_pos"] - obs_cache["cube_pos"]
                        if f"{pf}eef_pos" in obs_cache and "cube_pos" in obs_cache
                        else np.zeros(3)
                    )

                sensors = [cube_pos, cube_quat, gripper_to_cube_pos]

            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """

        self.grasp_state = -1  # Not grasping to start with
        self.done = False
        self.cloth_body_pos = np.zeros(shape=(3, ))

        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            if self.placement_initializer:
                # Sample from the placement initializer for all objects
                object_placements = self.placement_initializer.sample()

                # Loop through all objects and reset their positions
                for obj_pos, obj_quat, obj in object_placements.values():
                    if obj.joints:
                        self.sim.data.set_joint_qpos(obj.joints[0],
                                                     np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        if not self.include_cloth:
            vis_settings["grippers"] = True

            super().visualize(vis_settings=vis_settings)

            # Color the gripper visualization site according to its distance to the cube
            if vis_settings["grippers"]:
                self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube)

    def pick(self, pick_pose):
        """
        Pick the object at pick pose, can fail to grasp. Let the caller know so they can decide what to do.
        """
        self.done = False
        self._hover(pick_pose)
        if self.include_cloth:
            cloth_id = f"cloth_{self.cloth_body_id}"
            self.logger.info(f"cloth_body_pos after hover: {self.sim.data.body_xpos[self.sim.model.body_name2id(cloth_id)]}")
        self._lift(height=-0.07)
        if self.include_cloth:
            cloth_id = f"cloth_{self.cloth_body_id}"
            self.logger.info(f"cloth_body_pos after lift down: {self.sim.data.body_xpos[self.sim.model.body_name2id(cloth_id)]}")
        self._grasp()
        last_obs = self._lift()
        if self._check_success():
            return last_obs, True
        else:
            return None, False

    def place(self, place_hmat):
        eef_init_pose = self._get_current_eef_pose()
        self.logger.info("place")
        last_obs = self.trajectory_follower.follow(place_hmat, eef_init_pose, self.grasp_state)
        self._ungrasp()
        self.done = True
        return last_obs, self.reward()

    def go_home(self):
        j_traj = mr.JointTrajectory(self.robots[0].recent_qpos.current,
                                    self.robots[0].init_qpos, 5, 100, 5)
        for jnt in j_traj:
            self.robots[0].set_robot_joint_positions(jnt)
            if not self.headless:
                self.render()

    def _hover(self, pick_object_pose):
        hover_pose = pick_object_pose.copy()
        hover_pose[:-1, -1] = hover_pose[:-1, -1] + np.array([0, 0, 0.05])
        return self.trajectory_follower.follow(hover_pose,
                                               self._get_current_eef_pose(), self.grasp_state)

    def _lift(self, height=0.2):
        eef_pose = self._get_current_eef_pose()
        self.logger.info("lift " + ("up" if height > 0 else "down"))
        lift_pose = rtu.make_pose(eef_pose[:-1, -1] + [0, 0, height], eef_pose[:-1, :-1])
        return self.trajectory_follower.follow(lift_pose, eef_pose, self.grasp_state)

    def _grasp(self):
        self.grasp_state = 1
        return self._grasp_imp()

    def _ungrasp(self):
        self.grasp_state = -1
        return self._grasp_imp()

    def _grasp_imp(self):
        last_obs = None
        for i in range(10):
            obs, reward, done, _ = self.step([0, 0, 0, 0, 0, 0, self.grasp_state])
            if not self.headless:
                self.render()
            last_obs = obs
        return last_obs

    def _check_success(self, lift_height=0.2):
        """
        Check if cube has been lifted.

        Returns:
            bool: True if cube has been lifted
        """
        if not self.include_cloth:
            object_height = self.sim.data.body_xpos[self.cube_body_id][2]
            table_height = self.model.mujoco_arena.table_offset[2]
            return object_height > table_height + 0.04
        else:
            cloth_original_height = self.cloth_body_pos[2]
            cloth_current_height = self.sim.data.body_xpos[self.sim.model.body_name2id(f"cloth_{self.cloth_body_id}")][2]
            return cloth_current_height > cloth_original_height + lift_height

    def _get_current_eef_pose(self):
        current_obs = self._get_observations()
        return tr.pos_quat_to_hmat(current_obs['robot0_eef_pos'],
                                   current_obs['robot0_eef_quat'])
