#!/usr/bin/env python3

"""FrankaPick task."""

import numpy as np
import os
import torch
import imageio
import random

from typing import Tuple
from torch import Tensor

from pixmc.utils.torch_jit_utils import *
from pixmc.tasks.base.base_task import BaseTask

from isaacgym import gymtorch
from isaacgym import gymapi, gymutil
import cv2
# Find the path to the parent directory of the folder containing this file.
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
# exlucde the last folder
DIR_PATH = os.path.dirname(DIR_PATH)
DIR_PATH = os.path.dirname(DIR_PATH)
DIR_PATH = os.path.dirname(DIR_PATH)

class FrankaPush(BaseTask):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        assert self.physics_engine == gymapi.SIM_PHYSX

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.action_scale = self.cfg["env"]["actionScale"]

        self.object_dist_reward_scale = self.cfg["env"]["objectDistRewardScale"]
        self.lift_bonus_reward_scale = self.cfg["env"]["liftBonusRewardScale"]
        self.goal_dist_reward_scale = self.cfg["env"]["goalDistRewardScale"]
        self.goal_bonus_reward_scale = self.cfg["env"]["goalBonusRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]

        self.up_axis = "z"
        self.up_axis_idx = 2
        self.dt = 1 / 60.

        self.obs_type = self.cfg["env"]["obs_type"]
        assert self.obs_type in ["robot", "oracle", "pixels"]

        if self.obs_type == "robot":
            num_obs = 9 * 2
            self.compute_observations = self.compute_robot_obs
        elif self.obs_type == "oracle":
            num_obs = 37
            self.compute_observations = self.compute_oracle_obs
        else:
            self.cam_w = self.cfg["env"]["cam"]["w"]
            self.cam_h = self.cfg["env"]["cam"]["h"]
            self.cam_fov = self.cfg["env"]["cam"]["fov"]
            self.cam_ss = self.cfg["env"]["cam"]["ss"]
            self.cam_loc_p = self.cfg["env"]["cam"]["loc_p"]
            self.cam_loc_r = self.cfg["env"]["cam"]["loc_r"]
            self.im_size = self.cfg["env"]["im_size"]
            num_obs = (3, self.im_size, self.im_size)
            self.compute_observations = self.compute_pixel_obs
            assert self.cam_h == self.im_size
            assert self.cam_w % 2 == 0

        self.cfg["env"]["numObservations"] = num_obs
        self.cfg["env"]["numStates"] = 9 * 2
        self.cfg["env"]["numActions"] = 9

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        # Third person cam [FOR STATE MODEL USE ONLY]
        self.enable_third_person_cam = self.cfg["env"]["enable_third_person_cam"]
        self.save_third_person_view_image = False
        if self.enable_third_person_cam:
            self.cam_w = self.cfg["env"]["cam"]["w"]
            self.cam_h = self.cfg["env"]["cam"]["h"]
            self.cam_fov = self.cfg["env"]["cam"]["fov"]
            self.cam_ss = self.cfg["env"]["cam"]["ss"]
            self.cam_loc_p = self.cfg["env"]["cam"]["loc_p"]
            self.cam_loc_r = self.cfg["env"]["cam"]["loc_r"]
            self.im_size = self.cfg["env"]["im_size"]
            assert self.cam_h == self.im_size
            assert self.cam_w % 2 == 0

        super().__init__(cfg=self.cfg, enable_camera_sensors=(self.obs_type == "pixels" or self.enable_third_person_cam))

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Default franka dof pos
        self.franka_default_dof_pos = to_torch(
            #[1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.001, 0.001], device=self.device
            #[1.2765e+00, -9.1308e-01, -3.4194e-01, -2.4059e+00, -1.6185e+00,
            #1.2260e+00,  6.0274e-01,  8.9237e-06,  4.0000e-02], device=self.device
            [1.0373e+00, -9.8524e-01, -3.2429e-01, -2.6312e+00, -1.3451e+00,
            1.3418e+00,  8.8256e-01,  0.0000e+00,  4.0000e-02], device=self.device
        )

        # Dof state slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.franka_dof_state = self.dof_state.view(self.num_envs, self.num_franka_dofs, 2)
        self.franka_dof_pos = self.franka_dof_state[..., 0]
        self.franka_dof_vel = self.franka_dof_state[..., 1]

        # (N, num_bodies, 13)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)

        # (N, 3, 13)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        # (N, num_bodies, 3)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)

        # Finger pos
        self.lfinger_pos = self.rigid_body_states[:, self.rigid_body_lfinger_ind, 0:3]
        self.rfinger_pos = self.rigid_body_states[:, self.rigid_body_rfinger_ind, 0:3]

        # Finger rot
        self.lfinger_rot = self.rigid_body_states[:, self.rigid_body_lfinger_ind, 3:7]
        self.rfinger_rot = self.rigid_body_states[:, self.rigid_body_rfinger_ind, 3:7]

        # Object pos
        self.object_pos = self.root_state_tensor[:, self.env_object_ind, :3]

        # Avoidance box pose
        self.avoidance_box_pos = self.root_state_tensor[:, self.env_avoidance_box_ind, :3]
        
        # Dof targets
        self.dof_targets = torch.zeros((self.num_envs, self.num_franka_dofs), dtype=torch.float, device=self.device)

        # Global inds
        self.global_indices = torch.arange(
            self.num_envs * (1 + 1 + 1 + 1 + 1), dtype=torch.int32, device=self.device
        ).view(self.num_envs, -1)

        # Franka dof pos and vel scaled
        self.franka_dof_pos_scaled = torch.zeros_like(self.franka_dof_pos)
        self.franka_dof_vel_scaled = torch.zeros_like(self.franka_dof_vel)

        # Finger to object vecs # TODO: rename
        self.lfinger_to_target = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.rfinger_to_target = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        # Goal height diff
        self.to_height = torch.zeros((self.num_envs, 1), dtype=torch.float, device=self.device)
        
        # Collision detection for seach step
        self.in_collision = torch.zeros((self.num_envs, 1), dtype=torch.float, device=self.device)
        
        # Distance between two objects
        self.objects_distance = torch.zeros((self.num_envs, 1), dtype=torch.float, device=self.device)
        # Image mean and std
        if self.obs_type == "pixels" or self.enable_third_person_cam:
            self.im_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float, device=self.device).view(3, 1, 1)
            self.im_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float, device=self.device).view(3, 1, 1)

        # Object pos randomization
        self.object_pos_init = torch.tensor(cfg["env"]["object_pos_init"], dtype=torch.float, device=self.device)
        self.object_pos_delta = torch.tensor(cfg["env"]["object_pos_delta"], dtype=torch.float, device=self.device)

        # Goal height
        self.goal_height = torch.tensor(cfg["env"]["goal_height"], dtype=torch.float, device=self.device)

        # Success counts
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.extras["successes"] = self.successes
        # Put privilege_rew_buf in extras so that it is saved
        self.extras["privilege_rew_buf"] = self.privilege_rew_buf

        self.reset(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # Retrieve asset paths
        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        franka_asset_file = self.cfg["env"]["asset"]["assetFileNameFranka"]

        # Load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        # Create table asset
        table_dims = gymapi.Vec3(0.6, 1.5, 0.37)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

        # Create object asset
        self.object_size = 0.05
        asset_options = gymapi.AssetOptions()
        asset_options.density = 5000
        object_asset = self.gym.create_box(self.sim, self.object_size, self.object_size, 0.06, asset_options)

        # Creat the avoidance box asset
        self.avoidance_box_size = self.object_size
        avoidance_box_dims = gymapi.Vec3(self.avoidance_box_size, self.avoidance_box_size, self.avoidance_box_size)
        asset_options = gymapi.AssetOptions()
        #asset_options.fix_base_link = True
        asset_options.density = 5000
        avoidance_box_asset = self.gym.create_box(self.sim, avoidance_box_dims.x, avoidance_box_dims.y, avoidance_box_dims.z, asset_options)

        # Create av asset
        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = True
        asset_options.fix_base_link = True
        goal_asset = self.gym.create_box(self.sim, table_dims.x / 3.0, table_dims.y, 0.0001, asset_options)
        

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)

        # Franka dof gains (should probably be in the config)
        franka_dof_stiffness = [400, 400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6]
        franka_dof_damping = [80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2]

        # Set franka dof props
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        for i in range(self.num_franka_dofs):
            franka_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            franka_dof_props["stiffness"][i] = franka_dof_stiffness[i]
            franka_dof_props["damping"][i] = franka_dof_damping[i]

        # Record franka dof limits
        self.franka_dof_lower_limits = torch.zeros(self.num_franka_dofs, device=self.device, dtype=torch.float)
        self.franka_dof_upper_limits = torch.zeros(self.num_franka_dofs, device=self.device, dtype=torch.float)
        for i in range(self.num_franka_dofs):
            self.franka_dof_lower_limits[i] = franka_dof_props["lower"][i].item()
            self.franka_dof_upper_limits[i] = franka_dof_props["upper"][i].item()

        # Set franka gripper dof props
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[7, 8]] = 0.1
        franka_dof_props['effort'][7] = 200
        franka_dof_props['effort'][8] = 200

        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)

        avoidance_box_start_pose = gymapi.Transform()
        avoidance_box_start_pose.p = gymapi.Vec3(0.55, 0.1, table_dims.z + 0.03)
        self.avoidance_box_start_position = avoidance_box_start_pose.p
        
        self.goal_x = 0.4

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0.6, 0.0, table_dims.z + 0.03) # this will be overwritten
        self.object_z_init = object_start_pose.p.z

        goal_reagion_start_pose = gymapi.Transform()
        goal_reagion_start_pose.p = gymapi.Vec3(0.3, table_start_pose.p.y, table_dims.z)

        # Compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        num_table_bodies = self.gym.get_asset_rigid_body_count(table_asset)
        num_table_shapes = self.gym.get_asset_rigid_shape_count(table_asset)
        num_object_bodies = self.gym.get_asset_rigid_body_count(object_asset)
        num_object_shapes = self.gym.get_asset_rigid_shape_count(object_asset)
        num_avoidance_box_bodies = self.gym.get_asset_rigid_body_count(avoidance_box_asset)
        num_avoidance_box_shapes = self.gym.get_asset_rigid_shape_count(avoidance_box_asset)
        num_goal_bodies = self.gym.get_asset_rigid_body_count(goal_asset)
        num_goal_shapes = self.gym.get_asset_rigid_shape_count(goal_asset)
        max_agg_bodies = num_franka_bodies + num_table_bodies + num_object_bodies + num_avoidance_box_bodies + num_goal_bodies
        max_agg_shapes = num_franka_shapes + num_table_shapes + num_object_shapes + num_avoidance_box_shapes + num_goal_shapes

        self.frankas = []
        self.tables = []
        self.objects = []
        self.envs = []
        self.avoidance_boxes = []

        if self.obs_type == "pixels":
            self.cams = []
            self.cam_tensors = []
        if self.enable_third_person_cam:
            self.third_person_cams = []
            self.third_person_cam_tensors = []
            self.save_third_person_view_image = True
            self.third_person_cam_image_id = 0
            self.visual_obs_buf = torch.zeros((self.num_envs, 3, self.im_size, self.im_size), device=self.device, dtype=torch.float)
        # self.enable_third_person_cam = False
        # self.third_person_cams = []
        # self.third_person_cam_tensors = []
        # self.save_third_person_view_image = True
        # self.third_person_cam_image_id = 0

        for i in range(self.num_envs):
            # Create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Aggregate actors
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Franka actor
            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            # Table actor
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 0, 0)


            # Object actor
            object_actor = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 0, 0)
            object_color = gymapi.Vec3(1, 0, 0) # this is the rgb value
            self.gym.set_rigid_body_color(env_ptr, object_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, object_color)

            # Avoidance box actor
            avoidance_box_actor = self.gym.create_actor(env_ptr, avoidance_box_asset, avoidance_box_start_pose, "avoidance_box", i, 0, 0)
            self.gym.set_rigid_body_color(env_ptr, avoidance_box_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 1, 0))

            # Goal reagion actor
            goal_actor = self.gym.create_actor(env_ptr, goal_asset, goal_reagion_start_pose, "goal", i, 0, 0)
            goal_color = gymapi.Vec3(100, 159, 255) / 255
            self.gym.set_rigid_body_color(env_ptr, goal_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, goal_color)


            self.gym.end_aggregate(env_ptr)

            # TODO: move up
            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)
            self.tables.append(table_actor)
            self.objects.append(object_actor)
            self.avoidance_boxes.append(avoidance_box_actor)
            #self.goals.append(goal_actor)


            # Set up a third person camera
            if self.enable_third_person_cam:
                cam_props = gymapi.CameraProperties()
                cam_props.width = self.cam_w
                cam_props.height = self.cam_h
                cam_props.horizontal_fov = self.cam_fov
                cam_props.supersampling_horizontal = self.cam_ss
                cam_props.supersampling_vertical = self.cam_ss
                cam_props.enable_tensors = True
                cam_handle = self.gym.create_camera_sensor(env_ptr, cam_props)
                # Fix the camera to the hand
                # rigid_body_hand_ind = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_hand")
                # local_t = gymapi.Transform()
                # local_t.p = gymapi.Vec3(*self.cam_loc_p)
                # xyz_angle_rad = [np.radians(a) for a in self.cam_loc_r]
                # local_t.r = gymapi.Quat.from_euler_zyx(*xyz_angle_rad)
                # self.gym.attach_camera_to_body(
                #     cam_handle, env_ptr, rigid_body_hand_ind,
                #     local_t, gymapi.FOLLOW_TRANSFORM
                # )
                # Manually set camera position
                cam_pos = gymapi.Vec3(0.4, 0.35, 0.55)
                cam_target = gymapi.Vec3(1.5, -4.0, -0.0)
                self.gym.set_camera_location(cam_handle, env_ptr, cam_pos, cam_target)
                self.third_person_cams.append(cam_handle)
                # Camera tensor
                cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_handle, gymapi.IMAGE_COLOR)
                cam_tensor_th = gymtorch.wrap_tensor(cam_tensor)
                self.third_person_cam_tensors.append(cam_tensor_th)

            # Set up camera
            if self.obs_type == "pixels":
                # Camera
                cam_props = gymapi.CameraProperties()
                cam_props.width = self.cam_w
                cam_props.height = self.cam_h
                cam_props.horizontal_fov = self.cam_fov
                cam_props.supersampling_horizontal = self.cam_ss
                cam_props.supersampling_vertical = self.cam_ss
                cam_props.enable_tensors = True
                cam_handle = self.gym.create_camera_sensor(env_ptr, cam_props)
                rigid_body_hand_ind = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_hand")
                local_t = gymapi.Transform()
                local_t.p = gymapi.Vec3(*self.cam_loc_p)
                xyz_angle_rad = [np.radians(a) for a in self.cam_loc_r]
                local_t.r = gymapi.Quat.from_euler_zyx(*xyz_angle_rad)
                self.gym.attach_camera_to_body(
                    cam_handle, env_ptr, rigid_body_hand_ind,
                    local_t, gymapi.FOLLOW_TRANSFORM
                )
                # cam_pos = gymapi.Vec3(0.8, -0.2, 0.8)
                # cam_target = gymapi.Vec3(0.5, -0.2, 0.6)
                # cam_pos = gymapi.Vec3(-0.1, 0.0, 0.8)
                # cam_target = gymapi.Vec3(0.0, 0.0, 0.8)
                cam_pos = gymapi.Vec3(-0.2, 0.2, 0.8)
                cam_target = gymapi.Vec3(0.0, 0.15, 0.75)
                self.gym.set_camera_location(cam_handle, env_ptr, cam_pos, cam_target)
                self.cams.append(cam_handle)
                # Camera tensor
                cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_handle, gymapi.IMAGE_COLOR)
                cam_tensor_th = gymtorch.wrap_tensor(cam_tensor)
                self.cam_tensors.append(cam_tensor_th)

        self.rigid_body_lfinger_ind = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_leftfinger")
        self.rigid_body_rfinger_ind = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_rightfinger")

        self.env_franka_ind = self.gym.get_actor_index(env_ptr, franka_actor, gymapi.DOMAIN_ENV)
        self.env_table_ind = self.gym.get_actor_index(env_ptr, table_actor, gymapi.DOMAIN_ENV)
        self.env_object_ind = self.gym.get_actor_index(env_ptr, object_actor, gymapi.DOMAIN_ENV)
        self.env_avoidance_box_ind = self.gym.get_actor_index(env_ptr, avoidance_box_actor, gymapi.DOMAIN_ENV)
        

        franka_rigid_body_names = self.gym.get_actor_rigid_body_names( env_ptr, franka_actor)
        franka_arm_body_names = [name for name in franka_rigid_body_names if "link" in name]

        self.rigid_body_arm_inds = torch.zeros(len(franka_arm_body_names), dtype=torch.long, device=self.device)
        for i, n in enumerate(franka_arm_body_names):
            self.rigid_body_arm_inds[i] = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, n)

        self.init_grasp_pose()

    def init_grasp_pose(self):
        self.local_finger_grasp_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.local_finger_grasp_pos[:, 2] = 0.045
        self.local_finger_grasp_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.local_finger_grasp_rot[:, 3] = 1.0

        self.lfinger_grasp_pos = torch.zeros_like(self.local_finger_grasp_pos)
        self.lfinger_grasp_rot = torch.zeros_like(self.local_finger_grasp_rot)
        self.lfinger_grasp_rot[..., 3] = 1.0

        self.rfinger_grasp_pos = torch.zeros_like(self.local_finger_grasp_pos)
        self.rfinger_grasp_rot = torch.zeros_like(self.local_finger_grasp_rot)
        self.rfinger_grasp_rot[..., 3] = 1.0

    def compute_reward(self, actions):
      self.rew_buf[:], self.reset_buf[:], self.successes[:], self.privilege_rew_buf[:] = compute_franka_reward(
            self.reset_buf, self.progress_buf, self.successes, self.actions,
            self.lfinger_grasp_pos, self.rfinger_grasp_pos, self.object_pos, self.to_height,
            self.object_z_init, self.object_dist_reward_scale, self.lift_bonus_reward_scale,
            self.goal_dist_reward_scale, self.goal_bonus_reward_scale, self.action_penalty_scale,
            self.contact_forces, self.rigid_body_arm_inds, self.max_episode_length, self.in_collision,
            self.objects_distance, self.avoidance_box_pos
        )
    
    def reset_all(self):
        for i in range(self.num_envs):
            self.reset([i])

    def reset(self, env_ids):

        # Franka multi env ids
        franka_multi_env_ids_int32 = self.global_indices[env_ids, self.env_franka_ind].flatten()

        # Reset franka dofs
        dof_pos_noise = torch.rand((len(env_ids), self.num_franka_dofs), device=self.device)
        dof_pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) + 0.25 * (dof_pos_noise - 0.5) * 0,
            self.franka_dof_lower_limits, self.franka_dof_upper_limits
        )
        self.franka_dof_pos[env_ids, :] = dof_pos
        self.franka_dof_vel[env_ids, :] = 0.0
        self.dof_targets[env_ids, :] = dof_pos

        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_targets),
            gymtorch.unwrap_tensor(franka_multi_env_ids_int32),
            len(franka_multi_env_ids_int32)
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(franka_multi_env_ids_int32),
            len(franka_multi_env_ids_int32)
        )

        # Object multi env ids
        object_multi_env_ids_int32 = self.global_indices[env_ids, self.env_object_ind].flatten()

        avoidance_box_multi_env_ids_int32 = self.global_indices[env_ids, self.env_avoidance_box_ind].flatten()

        # Reset object pos
        delta_x = torch_rand_float(
            -self.object_pos_delta[0], self.object_pos_delta[0],
            (len(env_ids), 1), device=self.device
        ).squeeze(dim=1)
        delta_y = torch_rand_float(
            -self.object_pos_delta[1], self.object_pos_delta[1],
            (len(env_ids), 1), device=self.device
        ).squeeze(dim=1)

        self.root_state_tensor[env_ids, self.env_object_ind, 0] = self.object_pos_init[0] + delta_x
        self.root_state_tensor[env_ids, self.env_object_ind, 1] = self.object_pos_init[1] + delta_y
        self.root_state_tensor[env_ids, self.env_object_ind, 2] = self.object_z_init
        self.root_state_tensor[env_ids, self.env_object_ind, 3:6] = 0.0
        self.root_state_tensor[env_ids, self.env_object_ind, 6] = 1.0
        self.root_state_tensor[env_ids, self.env_object_ind, 7:10] = 0.0
        self.root_state_tensor[env_ids, self.env_object_ind, 10:13] = 0.0
        
        delta_x = torch_rand_float(
            -self.object_pos_delta[0], self.object_pos_delta[0],
            (len(env_ids), 1), device=self.device
        ).squeeze(dim=1)
        delta_y = torch_rand_float(
            -0.12, 0.12,
            (len(env_ids), 1), device=self.device
        ).squeeze(dim=1)

        # delta_y_1 = torch_rand_float(
        #     -0.12, -0.05,
        #     (len(env_ids), 1), device=self.device
        # ).squeeze(dim=1)
        # delta_y_2 = torch_rand_float(
        #     0.05, 0.12,
        #     (len(env_ids), 1), device=self.device
        # ).squeeze(dim=1)
        # random_n = torch.rand(1)
        # if random_n < 0.5:
        #     delta_y = delta_y_1
        # else:
        #     delta_y = delta_y_2

        self.root_state_tensor[env_ids, self.env_avoidance_box_ind, 0] = self.object_pos_init[0] + delta_x - 0.12
        self.root_state_tensor[env_ids, self.env_avoidance_box_ind, 1] = self.object_pos_init[1] + delta_y
        self.root_state_tensor[env_ids, self.env_avoidance_box_ind, 2] = self.object_z_init
        self.root_state_tensor[env_ids, self.env_avoidance_box_ind, 3:6] = 0.0
        self.root_state_tensor[env_ids, self.env_avoidance_box_ind, 6] = 1.0
        self.root_state_tensor[env_ids, self.env_avoidance_box_ind, 7:10] = 0.0
        self.root_state_tensor[env_ids, self.env_avoidance_box_ind, 10:13] = 0.0

        # connect object_multi_env_ids_int32 and avoidance_box_multi_env_ids_int32
        object_multi_env_ids_int32 = torch.cat((object_multi_env_ids_int32, avoidance_box_multi_env_ids_int32), dim=0)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(object_multi_env_ids_int32),
            len(object_multi_env_ids_int32)
        )

        # # Reset object pos
        # delta_x = torch_rand_float(
        #     -self.object_pos_delta[0], self.object_pos_delta[0],
        #     (len(env_ids), 1), device=self.device
        # ).squeeze(dim=1)
        # delta_y = torch_rand_float(
        #     -self.object_pos_delta[1], self.object_pos_delta[1],
        #     (len(env_ids), 1), device=self.device
        # ).squeeze(dim=1)

        # self.root_state_tensor[env_ids, self.env_avoidance_box_ind, 0] = self.object_pos_init[0] + delta_x + 0.1
        # self.root_state_tensor[env_ids, self.env_avoidance_box_ind, 1] = self.object_pos_init[1] + delta_y + 0.1
        # self.root_state_tensor[env_ids, self.env_avoidance_box_ind, 2] = self.object_z_init
        # self.root_state_tensor[env_ids, self.env_avoidance_box_ind, 3:6] = 0.0
        # self.root_state_tensor[env_ids, self.env_avoidance_box_ind, 6] = 1.0
        # self.root_state_tensor[env_ids, self.env_avoidance_box_ind, 7:10] = 0.0
        # self.root_state_tensor[env_ids, self.env_avoidance_box_ind, 10:13] = 0.0



        # self.gym.set_actor_root_state_tensor_indexed(
        #     self.sim,
        #     gymtorch.unwrap_tensor(self.root_state_tensor),
        #     gymtorch.unwrap_tensor(avoidance_box_multi_env_ids_int32),
        #     len(avoidance_box_multi_env_ids_int32)
        # )

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        targets = self.dof_targets \
            + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.dof_targets[:, :] = tensor_clamp(
            targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits
        )
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_targets))

    def compute_task_state(self):
        self.lfinger_grasp_rot[:], self.lfinger_grasp_pos[:] = tf_combine(
            self.lfinger_rot, self.lfinger_pos,
            self.local_finger_grasp_rot, self.local_finger_grasp_pos
        )
        self.rfinger_grasp_rot[:], self.rfinger_grasp_pos[:] = tf_combine(
            self.rfinger_rot, self.rfinger_pos,
            self.local_finger_grasp_rot, self.local_finger_grasp_pos
        )
        self.lfinger_to_target[:] = self.object_pos - self.lfinger_grasp_pos
        self.rfinger_to_target[:] = self.object_pos - self.rfinger_grasp_pos

        # distance to the goal region (max between 2 objects)
        self.to_height[:] = torch.maximum(torch.abs(self.object_pos[:, 0].unsqueeze(1) - self.goal_x), torch.abs(self.avoidance_box_pos[:, 0].unsqueeze(1) - self.goal_x))
        #goal_position = torch.tensor([self.goal_x, 0.0, 0.43], device=self.device)
        #goal_position = goal_position.expand(self.object_pos.shape[0], 3)
        #self.to_height[:] = torch.maximum(torch.norm(self.object_pos - goal_position, dim=1).unsqueeze(1), torch.norm(self.avoidance_box_pos - goal_position, dim=1).unsqueeze(1))
        # distance between 2 objects
        self.objects_distance[:] = torch.norm(self.object_pos - self.avoidance_box_pos, dim=1).unsqueeze(1)

    def compute_robot_state(self):
        self.franka_dof_pos_scaled[:] = \
            (2.0 * (self.franka_dof_pos - self.franka_dof_lower_limits) /
                (self.franka_dof_upper_limits - self.franka_dof_lower_limits) - 1.0)
        self.franka_dof_vel_scaled[:] = self.franka_dof_vel * self.dof_vel_scale

        self.states_buf[:, :self.num_franka_dofs] = self.franka_dof_pos_scaled
        self.states_buf[:, self.num_franka_dofs:] = self.franka_dof_vel_scaled

    def compute_robot_obs(self):
        self.obs_buf[:, :self.num_franka_dofs] = self.franka_dof_pos_scaled
        self.obs_buf[:, self.num_franka_dofs:] = self.franka_dof_vel_scaled

    def compute_oracle_obs(self):
        self.obs_buf[:] = torch.cat((
            self.franka_dof_pos_scaled, self.franka_dof_vel_scaled,
            self.lfinger_grasp_pos, self.rfinger_grasp_pos, self.object_pos,
            self.lfinger_to_target, self.rfinger_to_target, self.to_height, self.avoidance_box_pos,
        ), dim=-1)

    def compute_pixel_obs(self):
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        for i in range(self.num_envs):
            crop_l = (self.cam_w - self.im_size) // 2
            crop_r = crop_l + self.im_size
            self.obs_buf[i] = self.cam_tensors[i][:, crop_l:crop_r, :3].permute(2, 0, 1).float() / 255.
            self.obs_buf[i] = (self.obs_buf[i] - self.im_mean) / self.im_std
        self.gym.end_access_image_tensors(self.sim)
    
    def compute_visual_observations(self):
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        for i in range(self.num_envs):
            crop_l = (self.cam_w - self.im_size) // 2
            crop_r = crop_l + self.im_size
            self.visual_obs_buf[i] = self.third_person_cam_tensors[i][:, crop_l:crop_r, :3].permute(2, 0, 1).float()
            #self.visual_obs_buf[i] = (self.visual_obs_buf[i] - self.im_mean) / self.im_std
        self.gym.end_access_image_tensors(self.sim)

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset(env_ids)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.compute_task_state()
        self.compute_robot_state()
        self.compute_observations()
        self.compute_reward(self.actions)
        # Compute the third person view camera observations
        self.compute_visual_observations()
        
        
        # for i in range(self.num_envs):
        #     # define the vertices of the box
        #     center_pos = self.avoidance_box_start_position.x
        #     p1 = gymapi.Vec3(center_pos + self.avoidance_box_size/2, -self.avoidance_box_size/2, 0.401)
        #     p2 = gymapi.Vec3(center_pos - self.avoidance_box_size/2, -self.avoidance_box_size/2, 0.401)
        #     p3 = gymapi.Vec3(center_pos - self.avoidance_box_size/2, self.avoidance_box_size/2, 0.401)
        #     p4 = gymapi.Vec3(center_pos + self.avoidance_box_size/2, self.avoidance_box_size/2, 0.401)
            
        #     gymutil.draw_line(p1, p2, gymapi.Vec3(0, 0, 1), self.gym, self.viewer, self.envs[i])
        #     gymutil.draw_line(p2, p3, gymapi.Vec3(0, 0, 1), self.gym, self.viewer, self.envs[i])
        #     gymutil.draw_line(p3, p4, gymapi.Vec3(0, 0, 1), self.gym, self.viewer, self.envs[i])
        #     gymutil.draw_line(p4, p1, gymapi.Vec3(0, 0, 1), self.gym, self.viewer, self.envs[i])
        
        # # Same third view camera image in env id 0
        # if self.save_third_person_view_image:
            
        #     self.gym.render_all_camera_sensors(self.sim)
        #     self.gym.start_access_image_tensors(self.sim)
        #     crop_l = (self.cam_w - self.im_size) // 2
        #     crop_r = crop_l + self.im_size
        #     image_tensor = self.third_person_cam_tensors[0][:, crop_l:crop_r, :3].permute(2, 0, 1).float() / 255
        #     image_tensor = (image_tensor - self.im_mean) / self.im_std
        #     self.gym.end_access_image_tensors(self.sim)
        #     #print(image_tensor.shape)
        #     out_f = DIR_PATH + '/mvp_exp_data/rollout_images_save/' + "%d.png" % self.third_person_cam_image_id
        #     # check if the folder exists
        #     if not os.path.exists(DIR_PATH + '/mvp_exp_data/rollout_images_save/'):
        #         os.makedirs(DIR_PATH + '/mvp_exp_data/rollout_images_save/')
        #     #self.gym.write_camera_image_to_file(self.sim, self.envs[0], self.third_person_cams[0], gymapi.IMAGE_COLOR, out_f)
        #     ttt = image_tensor.cpu().numpy()
        #     ttt = np.moveaxis(ttt, 0, -1) * 255
        #     if self.third_person_cam_image_id > 0:
        #         cv2.imwrite(out_f, ttt)
        #     self.third_person_cam_image_id += 1


@torch.jit.script
def compute_franka_reward(
    reset_buf: Tensor, progress_buf: Tensor, successes: Tensor, actions: Tensor,
    lfinger_grasp_pos: Tensor, rfinger_grasp_pos: Tensor, object_pos: Tensor, to_height: Tensor,
    object_z_init: float, object_dist_reward_scale: float, lift_bonus_reward_scale: float,
    goal_dist_reward_scale: float, goal_bonus_reward_scale: float, action_penalty_scale: float,
    contact_forces: Tensor, arm_inds: Tensor, max_episode_length: int, collision_penalty: Tensor,
    objects_distance: Tensor, avoidance_box_pos: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    # Left finger to object distance
    lfo_d = torch.norm(object_pos - lfinger_grasp_pos, p=2, dim=-1)
    lfo_d = torch.clamp(lfo_d, min=0.02)
    lfo_dist_reward = 1.0 / (0.04 + lfo_d)

    #print(lfinger_grasp_pos)
    # Right finger to object distance
    rfo_d = torch.norm(object_pos - rfinger_grasp_pos, p=2, dim=-1)
    rfo_d = torch.clamp(rfo_d, min=0.02)
    rfo_dist_reward = 1.0 / (0.04 + rfo_d)

    # Object above table
    object_above = (object_pos[:, 2] - object_z_init) > 0.005

    # Above the table bonus
    lift_bonus_reward = torch.zeros_like(lfo_dist_reward)
    lift_bonus_reward = torch.where(object_above, lift_bonus_reward + 0.5, lift_bonus_reward)

    # Object to goal height distance
    og_d_x = torch.maximum(torch.abs(object_pos[:, 0].unsqueeze(1) - 0.4), torch.abs(avoidance_box_pos[:, 0].unsqueeze(1) - 0.4)).squeeze()
    og_d = torch.norm(to_height, p=2, dim=-1)
    og_dist_reward = torch.zeros_like(lfo_dist_reward)
    og_dist_reward =1.0 / (0.04 + og_d)


    # Regularization on the actions
    action_penalty = torch.sum(actions ** 2, dim=-1)
    
    rewards = -2.0 * og_d - 3.0 * objects_distance.squeeze() - 0.5 * lfo_d - 0.5 * rfo_d - action_penalty_scale * action_penalty
    rewards = rewards - lift_bonus_reward
    # Goal reached
    goal_height = 0.8 - 0.4  # absolute goal height - table height
    #s = torch.where(successes < 1.0, torch.zeros_like(successes), successes)
    successes = torch.where(og_d_x <= 0.05, torch.ones_like(successes), torch.zeros_like(successes))

    # Object below table height
    object_below = (object_z_init - object_pos[:, 2]) > 0.04
    #reset_buf = torch.where(object_below, torch.ones_like(reset_buf), reset_buf)

    # Arm collision
    arm_collision = torch.any(torch.norm(contact_forces[:, arm_inds, :], dim=2) > 1.0, dim=1)
    previlege_rewards = -1. * arm_collision.int()
    #print(arm_collision, collision_cost)
    rewards = rewards + 5.0 * previlege_rewards
    # if torch.any(arm_collision):
    #     print('arm collision')
    
    # check which env has collision
    #print(arm_collision)
    # if torch.any(arm_collision):
    #    print('arm collision')
    #print(arm_collision)
    #reset_buf = torch.where(arm_collision, torch.ones_like(reset_buf), reset_buf)

    # Max episode length exceeded
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    binary_s = torch.where(successes >= 1, torch.ones_like(successes), torch.zeros_like(successes))
    successes = torch.where(reset_buf > 0, binary_s, successes)
    return rewards, reset_buf, successes, previlege_rewards