"""
Environment file for "fixed arm" (FA) humanoid environment
with potential-based rewards implemented
"""

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from gpugym.utils.math import *
# from gpugym.envs import LeggedRobot, LeggedRobotCfg
from gpugym.envs.base.legged_robot import LeggedRobot, LeggedRobotCfg

from isaacgym import gymtorch, gymapi, gymutil
import os
from gpugym import LEGGED_GYM_ROOT_DIR, envs
from collections import deque
import random


class PBRS_OGHR_V4(LeggedRobot):

    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        if hasattr(self, "_custom_init"):
            self._custom_init(cfg)

    def _custom_init(self, cfg):
        self.dt_step = self.cfg.sim.dt * self.cfg.control.decimation
        self.pbrs_gamma = 0.99
        self.phase = torch.zeros(
            self.num_envs, 1, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.p5 = 0.5*torch.ones(
            self.num_envs, 1, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.stop = torch.randint(0,2,(self.num_envs,),device=self.device, requires_grad=False)
        self.eps = 0.2
        self.phase_freq = 1.
        if self.cfg.env.num_privileged_obs:
            self.num_privileged_obs = self.cfg.env.num_privileged_obs
        self.num_history_short = self.cfg.env.num_history_short
        self.num_history_long = self.cfg.env.num_history_long
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.size_history_long = self.cfg.env.size_history_long
        self.static_delay = self.cfg.commands.static_delay
        self.resampling_time = self.cfg.commands.resampling_time * torch.ones((self.num_envs, ), device=self.device)
        self.min_sample_time = self.cfg.commands.resampling_range[0]
        self.max_sample_time = self.cfg.commands.resampling_range[1]
        # for random pd
        self.low_p_ratio = self.cfg.domain_rand.p_range[0]
        self.high_p_ratio = self.cfg.domain_rand.p_range[1]

        
        self.low_d_ratio = self.cfg.domain_rand.d_range[0]
        self.high_d_ratio = self.cfg.domain_rand.d_range[1]
        
        if self.cfg.domain_rand.random_pd:
            self.random_p = torch_rand_float(self.low_p_ratio, self.high_p_ratio, 
                                    shape=(self.num_envs, self.num_dof), device=self.device)
            self.random_d = torch_rand_float(self.low_d_ratio, self.high_d_ratio, 
                                    shape=(self.num_envs, self.num_dof), device=self.device)
        else:
            self.random_p = torch.ones((self.num_envs, self.num_dof), device=self.device, requires_grad=False)
            self.random_d = torch.ones((self.num_envs, self.num_dof), device=self.device, requires_grad=False)
            self.low_p_ratio = 1.
            self.high_p_ratio = 1.
            self.low_d_ratio = 1.
            self.high_d_ratio = 1.

        self.average_p_ratio = (self.low_p_ratio + self.high_p_ratio) / 2.
        self.p_ratio_diff = self.high_p_ratio - self.average_p_ratio + 1e-8
        self.average_d_ratio = (self.low_d_ratio + self.high_d_ratio) / 2.
        self.d_ratio_diff = self.high_d_ratio - self.average_d_ratio + 1e-8


        #----------------------
        self.lin_vel1 = torch.zeros_like(self.base_lin_vel, device=self.device)
        self.lin_vel2 = torch.zeros_like(self.base_lin_vel, device=self.device)
        self.ang_vel1 = torch.zeros_like(self.base_ang_vel, device=self.device)
        self.ang_vel2 = torch.zeros_like(self.base_ang_vel, device=self.device)
        self.contact_force0 = torch.zeros_like(self.contact_forces[:, self.feet_indices, 2])
        self.contact_force1 = torch.zeros_like(self.contact_forces[:, self.feet_indices, 2])
        self.contact_force2 = torch.zeros_like(self.contact_forces[:, self.feet_indices, 2])
        self._rigid_body_vel0 = torch.zeros_like(self._rigid_body_vel)
       
        self.max_vel =torch.full([6], float("-inf"),device=self.device)

        # short history deques
        self.ctrl_hist_deque_short = deque(maxlen=self.num_history_short)
        self.dof_pos_hist_deque_short = deque(maxlen=self.num_history_short)
        self.dof_vel_hist_deque_short = deque(maxlen=self.num_history_short)
        self.base_ang_vel_hist_deque_short = deque(maxlen=self.num_history_short)
        self.proj_gravity_hist_deque_short = deque(maxlen=self.num_history_short)
        for _ in range(self.num_history_short):
            self.ctrl_hist_deque_short.append(torch.zeros(self.num_envs, self.num_actions,
                                     dtype=torch.float, device=self.device,
                                     requires_grad=False))
            self.dof_pos_hist_deque_short.append(torch.zeros(self.num_envs, self.num_dof,
                                     dtype=torch.float, device=self.device,
                                     requires_grad=False))
            self.dof_vel_hist_deque_short.append(torch.zeros(self.num_envs, self.num_dof,
                                     dtype=torch.float, device=self.device,
                                     requires_grad=False))
            self.base_ang_vel_hist_deque_short.append(torch.zeros(self.num_envs, 3,
                                     dtype=torch.float, device=self.device,
                                     requires_grad=False))
            self.proj_gravity_hist_deque_short.append(torch.zeros(self.num_envs, 3,
                                     dtype=torch.float, device=self.device,
                                     requires_grad=False))
        # long history deque
        if self.num_history_long:
            self.long_history_deque = deque(maxlen=self.num_history_long)
            for _ in range(self.num_history_long):
                self.long_history_deque.append(torch.zeros(self.num_envs, self.size_history_long,
                                        dtype=torch.float, device=self.device,
                                        requires_grad=False))
            
            self.long_history = torch.zeros(self.num_envs, self.num_history_long * self.size_history_long,
                                        dtype=torch.float, device=self.device,
                                        requires_grad=False)
            
        self.rand_push_force = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        self.rand_push_torque = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device, requires_grad=False)

        self.comm_delay = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False)
        max_delay = self.cfg.domain_rand.comm_delay_range[1]
        # for i in range(self.num_envs):
        #     if self.cfg.domain_rand.comm_delay:
        #         rng = self.cfg.domain_rand.comm_delay_range
        #         rd_num = np.random.randint(rng[0], rng[1])
        #         self.delay_deque_lst.append(deque(maxlen=rd_num+1))
        #         self.comm_delay[i] = rd_num
        #     else:
        #         self.delay_deque_lst.append(deque(maxlen=1))
        self.actions_record = torch.zeros((self.num_envs, max_delay, self.num_actions), device=self.device, requires_grad=False)
        if self.cfg.domain_rand.comm_delay:
            rng = self.cfg.domain_rand.comm_delay_range
            self.comm_delay = torch.randint(rng[0], rng[1], (self.num_envs, 1), device=self.device)

        self.pre_loc = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=False)
        self.pre_quat = torch.zeros((self.num_envs, 4), device=self.device, requires_grad=False)
        self.command_error_accumulate = torch.zeros((self.num_envs, 2), device=self.device, requires_grad=False)
        self.command_ang_error_accumulate = torch.zeros((self.num_envs, 1), device=self.device, requires_grad=False)
        self.pre_commands = torch.zeros((self.num_envs, 4), device=self.device, requires_grad=False)
        self.pre_loc[:] = self.root_states[:, :3].detach()
        _, _, self.yaw = get_euler_xyz(self.root_states[:, 3:7])
        self.pre_quat[:] = self.root_states[:, 3:7].detach()
        self.pre_commands[:] = self.commands.detach()
        _, _, self.pre_yaw = get_euler_xyz(self.root_states[:, 3:7])

        self.avg_footforce = torch.ones([self.num_envs,2], device=self.device, requires_grad=False)*400.

        self.dof_friction = torch.zeros((self.num_envs, self.num_dof), device=self.device, requires_grad=False)

        self.smooth_w = 1.

        self.foot_speed = self._rigid_body_vel[:, self.feet_indices, :3]
        self.foot_height = self._rigid_body_pos[:, self.feet_indices, 2] - 0.07 - self._get_foot_terrain_heights()
        self.fatigue = torch.zeros((self.num_envs, self.num_dof), device=self.device, requires_grad=False)
        self.fatigue_torque = torch.zeros((self.num_envs, self.num_dof), device=self.device, requires_grad=False)
        self.single_contact_last = torch.zeros((self.num_envs,), device=self.device, requires_grad=False)
        self.double_contact_last = torch.zeros((self.num_envs,), device=self.device, requires_grad=False)



    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        if self.cfg.asset.disable_actions:
            self.actions[:] = 0.
        else:
            ############### changed ########################################
            # delay = 0.14*torch.rand((self.num_envs, 1), device=self.device)
            # actions = (1 - delay) * actions + delay * self.actions
            # actions += self.cfg.domain_rand.dynamic_randomization * torch.randn_like(actions)
            ############### changed ########################################
            
            clip_actions = self.cfg.normalization.clip_actions

            self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
            # delayed_action = torch.clip(delayed_action, -clip_actions, clip_actions).to(self.device)
        self.pre_physics_step()
        # step physics and render each frame
        self.render()

        #这里是对decimation进行随机化，让其随机的在-2 ---2之间加减
        for _ in range(self.cfg.control.decimation + torch.randint(-2,3,[1])):
            if self.cfg.domain_rand.comm_delay:
                self.actions_record = torch.cat((self.actions_record[:, 1:, :], self.actions.unsqueeze(1)), dim=1)
                # print(self.actions.unsqueeze(1).size())

                # self.comm_delay可能是一个张量，表示每个环境的延迟量。torch.arange(self.num_envs)生成环境索引，
                # -1 - self.comm_delay计算要选取的动作的索引（考虑到延迟），.squeeze(1)去除可能的单维度。
                # action_records 按照 num_envs, delay, actions
                delayed_action = self.actions_record[torch.arange(self.num_envs), (-1 - self.comm_delay).squeeze(1)]
               

                if self.cfg.control.exp_avg_decay:
                    # 这里是设置一个衰减指数，这里的系数是0.05，即delayed action只占0.05
                    self.action_avg = exp_avg_filter(delayed_action, self.action_avg,
                                                    self.cfg.control.exp_avg_decay)
                    self.torques = self._compute_torques(self.action_avg).view(self.torques.shape)
                else:
                    self.torques = self._compute_torques(delayed_action).view(self.torques.shape)
            else:
                if self.cfg.control.exp_avg_decay:
                    self.action_avg = exp_avg_filter(self.actions, self.action_avg,
                                                    self.cfg.control.exp_avg_decay)
                    self.torques = self._compute_torques(self.action_avg).view(self.torques.shape)
                else:
                    self.torques = self._compute_torques(self.actions).view(self.torques.shape)

            if self.cfg.asset.disable_motors:
                self.torques[:] = 0.

            # print(self.torques[0,])
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        reset_env_ids, terminal_amp_states = self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf,
                                                 -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, \
            self.reset_buf, self.extras, reset_env_ids, terminal_amp_states        


    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        self.num_rigid_shape = len(rigid_shape_props_asset)

        self.friction_buf = torch.ones((self.num_envs, 1), device=self.device, requires_grad=False, dtype=torch.float32)
        self.damping_buf = torch.ones((self.num_envs, self.num_dof), device=self.device, requires_grad=False)
        self.mass_mask = torch.ones((self.num_envs, self.num_bodies), device=self.device, requires_grad=False)

        self.inertia_mask_xx = torch.ones((self.num_envs, self.num_bodies), device=self.device, requires_grad=False)
        self.inertia_mask_xy = torch.ones((self.num_envs, self.num_bodies), device=self.device, requires_grad=False)
        self.inertia_mask_xz = torch.ones((self.num_envs, self.num_bodies), device=self.device, requires_grad=False)
        self.inertia_mask_yy = torch.ones((self.num_envs, self.num_bodies), device=self.device, requires_grad=False)
        self.inertia_mask_yz = torch.ones((self.num_envs, self.num_bodies), device=self.device, requires_grad=False)
        self.inertia_mask_zz = torch.ones((self.num_envs, self.num_bodies), device=self.device, requires_grad=False)

        self.com_diff_x = torch.zeros((self.num_envs, self.num_bodies), device=self.device, requires_grad=False)
        self.com_diff_y = torch.zeros((self.num_envs, self.num_bodies), device=self.device, requires_grad=False)
        self.com_diff_z = torch.zeros((self.num_envs, self.num_bodies), device=self.device, requires_grad=False)

        self.joint_armatures = torch.tensor([
            # 0.000832,           #J_head_yaw
            # 0.000832,           #J_head_pitch
            # 0.02035,            #J_arm_r_01
            # 0.02035,            #J_arm_r_02
            # 0.01631,            #J_arm_r_03
            # 0.01631,            #J_arm_r_04
            # 0.00303,            #J_arm_r_05
            # 0.00303,            #J_arm_r_06
            # 0.00303,            #J_arm_r_07
            # 0.02035,            #J_arm_l_01
            # 0.02035,            #J_arm_l_02
            # 0.01631,            #J_arm_l_03
            # 0.01631,            #J_arm_l_04
            # 0.00303,            #J_arm_l_05
            # 0.00303,            #J_arm_l_06
            # 0.00303,            #J_arm_l_07
            0.52488,            #J_waist_pitch
            0.52488,            #J_waist_roll
            # 0.20808,          #J_waist_yaw
            0.091107173,        #Joint-hip-r-roll
            0.028943293,        #Joint-hip-r-yaw
            0.11460904,         #Joint-hip-r-pitch
            0.11460904,         #Joint-knee-r-pitch
            0.004378347,        #Joint-ankel-r-pitch
            0.004378347,        #Joint-ankel-r-roll
            0.091107173,        #Joint-hip-l-roll
            0.028943293,        #Joint-hip-l-yaw
            0.11460904,         #Joint-hip-l-pitch
            0.11460904,         #Joint-knee-l-pitch
            0.004378347,        #Joint-ankel-l-pitch
            0.004378347,        #Joint-ankel-l-roll

        ], device=self.device, requires_grad=False)



        if self.cfg.domain_rand.randomize_joint_armature:
            self.low_armature_range = self.cfg.domain_rand.armature_range[0]
            self.high_armature_range = self.cfg.domain_rand.armature_range[1]
            self.random_armature = torch_rand_float(self.low_armature_range, self.high_armature_range, 
                                    shape=(self.num_envs, self.num_dof), device=self.device)
        else:
            self.random_armature = torch.ones((self.num_envs, self.num_dof), device=self.device, requires_grad=False)

            self.low_armature_range = 1.
            self.high_armature_range = 1.

        # print("+++++++++++++++++++++++++++++++++++")
        # print("rigid_shape:", self.num_rigid_shape)
        # print("num body:", self.num_bodies)
        # print("num dof:", self.num_dof)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        # print(body_names, "+++++++")
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        # print(self.dof_names, "+++++++")
        self.num_bodies = len(body_names)
        # self.num_dofs = len(self.dof_names)  # ! replaced with num_dof
        feet_names = self.cfg.asset.foot_name
        knee_names = self.cfg.asset.knee_name
        # waist_names = self.cfg.asset.waist_name
		
		# 对于contact惩罚项和termination 项的定义
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            legged_robot_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "legged_robot", i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, legged_robot_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, legged_robot_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, legged_robot_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(legged_robot_handle)


        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        # self.waist_indices = torch.zeros(len(waist_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])
        # for i in range(len(waist_names)):
        #     self.waist_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], waist_names[i])


        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

        #foot sensors
        sensor_pose = gymapi.Transform()
        for name in feet_names:
            sensor_options = gymapi.ForceSensorProperties()
            sensor_options.enable_forward_dynamics_forces = False # for example gravity
            sensor_options.enable_constraint_solver_forces = True # for example contacts
            sensor_options.use_world_frame = True # report forces in world frame (easier to get vertical components)
            index = self.gym.find_asset_rigid_body_index(robot_asset, name)
            self.gym.create_asset_force_sensor(robot_asset, index, sensor_pose, sensor_options)
    
    def compute_observations(self):


        base_z = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1).unsqueeze(1)*self.obs_scales.base_z
        ######################################################################

        in_contact = torch.gt(
            self.contact_forces[:, self.end_eff_ids, 2], 0).int()
        
        in_contact = torch.cat(
            (in_contact[:, 0].unsqueeze(1), in_contact[:, 1].unsqueeze(1)),
            dim=1)
        # self.commands[:, 0:2] = torch.where(
        #     torch.norm(self.commands[:, 0:2], dim=-1, keepdim=True) < 0.3,
        #     0., self.commands[:, 0:2].double()).float()
        # self.commands[:, 2:3] = torch.where(
        #     torch.abs(self.commands[:, 2:3]) < 0.3,
        #     0., self.commands[:, 2:3].double()).float()


        square_wave = 0.* self.smooth_sqr_wave(self.phase)
        # torch.where((self.terrain_types<4).unsqueeze(1),
        #                                 0.* self.smooth_sqr_wave(self.phase),
        #                                 -self.p5)
        square_wave = torch.where(self.time_to_stand_still.unsqueeze(1) > self.static_delay, self.p5, square_wave)
        self.obs_buf = torch.cat((
            # base_z,                                 # [1] Base height *
            # self.base_lin_vel,                      # [3] Base linear velocity *
            self.commands[:, 0:4],                  # [4] Velocity commands
            square_wave,                            # [1] Contact schedule [;5]
            ####################################################################
            self.base_ang_vel,                      # [3] Base angular velocity [5:47]
            self.projected_gravity,                 # [3] Projected gravity
            # torch.sin(2*torch.pi*self.phase),       # [1] Phase variable
            # torch.cos(2*torch.pi*self.phase),       # [1] Phase variable
            self.actions*self.cfg.control.action_scale, # [12] Joint actions
            self.dof_pos,                           # [12] Joint states
            self.dof_vel,                           # [12] Joint velocities
            # in_contact,                             # [2] Contact states
            ####################################################################
            self.base_ang_vel_hist,                 # [9] Base angular velocity history
            self.proj_gravity_hist,                 # [9] Projected gravity
            self.ctrl_hist,                         # [36] action history
            self.dof_pos_hist,                      # [36] dof position history
            self.dof_vel_hist,                      # [36] dof velocity history history
        ), dim=-1)
        if self.num_privileged_obs:
            self.privileged_obs_buf = torch.cat((
                base_z,                                   # [1] Base height *
                self.base_lin_vel,                      # [3] Base linear velocity *
                in_contact,                             # [2] Contact states
                # self.contact_forces[:, self.feet_indices[0], 2].unsqueeze(1),
                # self.contact_forces[:, self.feet_indices[1], 2].unsqueeze(1),
                self.contact_forces[:, self.feet_indices[0], :],
                self.contact_forces[:, self.feet_indices[1], :],
                (self.random_p - self.average_p_ratio) / self.p_ratio_diff,  # [12]
                (self.random_d - self.average_d_ratio) / self.d_ratio_diff,  # [12]
                self.friction_buf,                      # [1]
                self.damping_buf,                       # [12]
                self.mass_mask,                         # [32]
                self.com_diff_x,                        # [32]
                self.com_diff_y,                        # [32]
                self.com_diff_z,                        # [32]
                self.rand_push_force,                   # [3]
                self.rand_push_torque,                  # [3]
                self.comm_delay,                        # [1]
                self.foot_height,                        # [2]
                # self.foot1_projected_gravity,            # [3]
                # self.foot2_projected_gravity,            # [3]
                # quat_rotate_inverse(self.base_quat, (self.root_states[:, :3]- self.pre_loc))[:, :2], # [2]
                self.command_error_accumulate,
                # (self.yaw - self.pre_yaw).unsqueeze(1),                # [1]
                self.command_ang_error_accumulate,
                self.foot_speed[:,0,:],
                self.foot_speed[:,1,:],
                self._rigid_body_vel0[:, self.feet_indices[0], :],
                self._rigid_body_vel0[:, self.feet_indices[1], :],
                self.feet_air_time,
                self.feet_stand_time,
                self.double_contact_last.unsqueeze(1),
                # damping(added)
                # base_z need to change
                # foot_height??
                # mass      
                # COM
                # friction(added)
                # inertia
                # delay
                # PD
                # external_force(torque)
                # pos_e
                # angle_e
                self.obs_buf.clone(),

            ), dim=-1)
            if self.cfg.terrain.measure_heights:
                heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 1.- self.measured_heights, -1, 1.)
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)

        # self.obs_buf[:,4] *= 0.
        if self.num_history_long:
            self.obs_buf = torch.cat([self.obs_buf, self.long_history], dim=-1)
        if self.add_noise:
            self.obs_buf += (torch.randn_like(self.obs_buf)) \
                * self.noise_scale_vec
            self.obs_buf[:,8:11] /= torch.norm(self.obs_buf[:,8:11],dim=-1).unsqueeze(1)

    def get_amp_observations(self):
        joint_pos = self.dof_pos   
        # foot_pos = self.foot_positions_in_base_frame(self.dof_pos).to(self.device)  
        base_lin_vel = self.base_lin_vel  #3
        base_ang_vel = self.base_ang_vel  #3
        # print("vel", self.dof_vel)
        # joint_vel = self.dof_vel  
        z_pos = self.root_states[:, 2:3] 
        # print("height",)
        # return torch.cat((joint_pos, foot_pos, base_lin_vel, base_ang_vel, joint_vel, z_pos), dim=-1)
        return torch.cat((joint_pos, base_lin_vel, base_ang_vel, z_pos), dim=-1) 








    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
    
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        # move_up = distance > self.terrain.env_length / 2
        move_up = torch.logical_or(distance > self.terrain.env_length / 2,
                                    torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.5 * self.reward_scales["tracking_lin_vel"])
        # robots that walked less than half of their required distance go to simpler terrains
        #
        move_down = distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5
        # torch.logical_or(distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5, 
        #                              torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length < 0.3 * self.reward_scales["tracking_lin_vel"]) * ~move_up

        # move_down = torch.logical_or(distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5, (torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length < 0.3 * self.reward_scales["tracking_lin_vel"])) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(torch.logical_or(self.terrain_levels[env_ids]>=self.max_terrain_level, torch.rand_like(self.terrain_levels[env_ids], dtype=float) < 0.05),
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        # print("###########################################")
        # print(torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length)
        # print((torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.5 * self.reward_scales["tracking_lin_vel"]))
        # print("move_up: ",move_up)
        # print("move_down: ",move_down)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]


    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        #######################  change later  ############################
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        # noise_vec[0] = noise_scales.base_z * self.obs_scales.base_z
        noise_vec[5:8] = noise_scales.ang_vel
        noise_vec[8:11] = noise_scales.gravity
        noise_vec[11:23] = 0. # actions
        noise_vec[23:35] = noise_scales.dof_pos
        noise_vec[35:47] = noise_scales.dof_vel
        # noise_vec[47:83] = 0 # ctrl hist
        # noise_vec[83:119] = noise_scales.dof_pos
        # noise_vec[119:155] = noise_scales.dof_vel
        # noise_vec[155:164] = noise_scales.ang_vel
        # noise_vec[164:173] = noise_scales.gravity
        # if self.cfg.terrain.measure_heights:
        #     noise_vec[48:235] = noise_scales.height_measurements \
        #         * noise_level \
        #         * self.obs_scales.height_measurements
        
        # if self.num_history_long:
        #     for i in range(self.num_history_long):
        #         noise_vec[self.cfg.env.num_observations+self.cfg.env.size_history_long*i:self.cfg.env.num_observations+self.cfg.env.size_history_long*i+12] = 0.
        #         noise_vec[self.cfg.env.num_observations+self.cfg.env.size_history_long*i+12:self.cfg.env.num_observations+self.cfg.env.size_history_long*i+24] = noise_scales.dof_pos
        #         noise_vec[self.cfg.env.num_observations+self.cfg.env.size_history_long*i+24:self.cfg.env.num_observations+self.cfg.env.size_history_long*i+36] = noise_scales.dof_vel
        #         noise_vec[self.cfg.env.num_observations+self.cfg.env.size_history_long*i+36:self.cfg.env.num_observations+self.cfg.env.size_history_long*i+39] = noise_scales.ang_vel
        #         noise_vec[self.cfg.env.num_observations+self.cfg.env.size_history_long*i+39:self.cfg.env.num_observations+self.cfg.env.size_history_long*i+42] = noise_scales.ang_vel
        
        noise_vec = noise_vec * noise_level
        return noise_vec

    def _custom_reset(self, env_ids):
        if self.cfg.commands.resampling_time == -1:
            self.commands[env_ids, :] = 0.
        self.phase[env_ids, 0] = torch.rand(
            (torch.numel(env_ids),), requires_grad=False, device=self.device)
        self.max_feet_air_time[env_ids, :] =torch.zeros((len(env_ids), 2), device=self.device)
        #下面这个是与feet stand time进行比较的 
        self.max_feet_stand_time[env_ids, :] =torch.zeros((len(env_ids), 2), device=self.device)
        self.max_feet_height[env_ids, :] = torch.zeros((len(env_ids), 2), device=self.device)

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.foot1_projected_gravity[:] = quat_rotate_inverse(self._rigid_body_ori[:, self.feet_indices[0], :], self.gravity_vec)
        self.foot2_projected_gravity[:] = quat_rotate_inverse(self._rigid_body_ori[:, self.feet_indices[1], :], self.gravity_vec)

        self._post_physics_step_callback()

        self.lin_vel2[:] = self.base_lin_vel
        self.ang_vel2[:] = self.base_ang_vel
        self.contact_force2[:] = self.contact_forces[:, self.feet_indices, 2]
    
        self.avg_footforce[:] = 0.99 * self.avg_footforce \
            + 0.01 * self.avg_footforce * (self.time_to_stand_still <= self.static_delay).unsqueeze(1) \
            + 0.01 * self.contact_forces[:, self.feet_indices, 2] * (self.time_to_stand_still > self.static_delay).unsqueeze(1)
        
        self.fatigue[:] = 0.995 * self.fatigue \
            + 0.005 * torch.abs(self.torques)* torch.abs(self.dof_vel) 
        
        self.fatigue_torque = 0.995 * self.fatigue_torque \
            + 0.005 * torch.abs(self.torques) 

        self.foot_speed = self._rigid_body_vel[:, self.feet_indices, :3]


        self.foot_height = self._rigid_body_pos[:, self.feet_indices, 2] - 0.07 - self._get_foot_terrain_heights()



        # not_contact = self.contact_forces[:, self.feet_indices, 2] < 5.
        # foot_height_rew = torch.sum(torch.sqrt(torch.norm(self.foot_speed[:,:,:2],dim=-1)) * torch.square(0.3 - self.foot_height) * not_contact,dim=-1)
        # print(torch.exp(-10.*foot_height_rew))
        # compute observations, rewards, resets, ...
        self.check_termination()
        
     
        self.commands_body = quat_rotate(self.pre_quat,torch.cat((self.pre_commands[:, :2], torch.zeros((self.num_envs,1), device=self.device, requires_grad=False)), dim=1))
        # print("position error1:", self.pre_loc)
        ori_loc = self.pre_loc[:]
        desired_moving = self.commands_body * self.dt_step
        self.pre_loc += desired_moving
        _, _, self.yaw = get_euler_xyz(self.root_states[:, 3:7])
        self.pre_yaw += self.pre_commands[:, 2] * self.dt_step

        # r_s = 0.
        # nact = self.num_actions
        # r_s -= 4000. * torch.square(self.actions*self.cfg.control.action_scale \
        #                     - self.ctrl_hist[:, :nact] \
        #                 )
        # r_s -= 2000. * torch.square(self.actions*self.cfg.control.action_scale \
        #                 - 2.*self.ctrl_hist[:, :nact]  \
        #                 + self.ctrl_hist[:, nact:2*nact]  \
        #                 )
        # r_s -= torch.square(self.dof_vel)
        # r_s -= 4. * torch.square((self.dof_vel_hist[:, :self.num_dof] - self.dof_vel))

        r_s = 0.
        r_s -= 1.*torch.square(self.actions*self.cfg.control.action_scale \
                            - self.ctrl_hist[:, -self.num_actions:] \
                        )
        r_s -= 0.5 * torch.square(self.actions*self.cfg.control.action_scale \
                        - 2.*self.ctrl_hist[:, -self.num_actions:]  \
                        + self.ctrl_hist[:, -2*self.num_actions:-self.num_actions]  \
                        ) * (self.time_to_stand_still > self.static_delay).unsqueeze(1)
        # r_s -= self.cfg.rewards.smooth_w * 2. * torch.abs(self.dof_vel/self.dof_vel_limits*self.torques/self.torque_limits)/self.num_dof
        r_s -= 1./4000. * torch.square(self.dof_vel) * (self.time_to_stand_still > self.static_delay).unsqueeze(1)
        r_s -= 1./1000. * torch.square((self.dof_vel_hist[:, -self.num_dof:] - self.dof_vel))
        # print(r_s[0,:])
        r_s *= torch.square(self.rs_w)
        

        self.smooth_w = torch.exp(
                                    self._reward_foot_pos()  #这个越大
                                  - 0.1 * self._reward_dof_pos_limits() #这个越小
                                #   - (self.cfg.rewards.smooth_w > 0) * 0.1 * self._reward_torque_limits()
                                #   - self.cfg.rewards.smooth_w * 2. * self._reward_termination()
                                  - self.cfg.rewards.smooth_w * 0.02 * self._reward_delta_torques()  #这个越小
                                  + self.cfg.rewards.smooth_w * 0.48 * torch.sum(r_s, dim=-1)
                                #   - self.cfg.rewards.smooth_w * 0.02* self._reward_foot_smooth()
                                #   - (self.cfg.rewards.smooth_w > 0) * 0.4 * self._reward_feet_air_time_diff()
                                #   - 0.004 * self._reward_feet_contact_forces()
                                #   torch.where(self.time_to_stand_still > self.static_delay, 
                                #                 - 0.000006 * self._reward_foot_impulse(),
                                #             #    - 0.02 * self._reward_feet_air_time_diff()
                                               
                                #                )
                                  )
        # print(- 0.8 * self._reward_lin_vel_z()
        # - 0.4 * self._reward_ang_vel_xy())
        # self.smooth_w = 1.0
        # print(self.smooth_w)
        # print(torch.exp(self.cfg.rewards.smooth_w * 0.48 * (torch.sum(r_s, dim=-1)))[0],"\n")
        if not self.cfg.rewards.accumulate_target:
            tar = (((self.root_states[:, :3] - ori_loc) * desired_moving).sum(axis = 1)/torch.square(torch.norm(desired_moving,dim = 1) + 1e-8)).unsqueeze(1) * desired_moving + ori_loc
            self.pre_loc[:] = torch.where((torch.logical_and(torch.norm(self.commands[:,:2],dim=1) == 0, torch.norm(self.base_lin_vel[:,:2],dim=1) <= 0.2)).unsqueeze(1), tar, self.root_states[:, :3])
            # self.pre_loc[:] = self.root_states[:, :3].detach()
            # print("before sim:", self.pre_loc[0])
            self.pre_yaw[:] = torch.where(self.commands[:,2]==0, self.pre_yaw[:], get_euler_xyz(self.root_states[:, 3:7])[2])
            # _, _, self.pre_yaw[:] = get_euler_xyz(self.root_states[:, 3:7])

        self.compute_reward()
        self.pre_commands[:] = self.commands.detach()
        self.pre_quat[:] = self.root_states[:, 3:7].detach()
        self.pre_quat[:,:2] = 0.
        self.pre_quat /= torch.norm(self.pre_quat,dim=-1).unsqueeze(1) 
        # print("after sim:",self.pre_loc[0])
        # print("root state after sim:",self.root_states[0, :3])


        if len(self.recommand_ids) != 0 and self.cfg.commands.resampling_time != -1 :
            self.pre_loc[self.recommand_ids, :] = self.root_states[self.recommand_ids, :3]
            _, _, cur_yaw = get_euler_xyz(self.root_states[:, 3:7])
            self.pre_yaw[self.recommand_ids] = cur_yaw[self.recommand_ids]

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        terminal_amp_states = self.get_amp_observations()[env_ids]

       
        self.reset_idx(env_ids)

        if len(env_ids) != 0:
            self.pre_loc[env_ids, :] = self.root_states[env_ids, :3]
            _, _, cur_yaw = get_euler_xyz(self.root_states[:, 3:7])
            self.pre_yaw[env_ids] = cur_yaw[env_ids]
        
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        # self.move_same()
        # if self.cfg.domain_rand.random_pd:
        #     self.random_p = torch_rand_float(self.low_p_ratio, self.high_p_ratio, 
        #                             shape=(self.num_envs, self.num_dof), device=self.device)
        #     self.random_d = torch_rand_float(self.low_d_ratio, self.high_d_ratio, 
        #                             shape=(self.num_envs, self.num_dof), device=self.device)
        self.last_torques[:] = self.torques[:]

        self.ctrl_hist_deque_short.append(self.obs_buf[:, 11:23])
        self.ctrl_hist = torch.cat([t for t in self.ctrl_hist_deque_short],dim=1)

        self.dof_pos_hist_deque_short.append(self.obs_buf[:, 23:35])
        self.dof_pos_hist = torch.cat([t for t in self.dof_pos_hist_deque_short],dim=1)

        self.dof_vel_hist_deque_short.append(self.obs_buf[:, 35:47])
        self.dof_vel_hist = torch.cat([t for t in self.dof_vel_hist_deque_short],dim=1)

        self.base_ang_vel_hist_deque_short.append(self.obs_buf[:, 5:8])
        self.base_ang_vel_hist = torch.cat([t for t in self.base_ang_vel_hist_deque_short],dim=1)
        
        self.proj_gravity_hist_deque_short.append(self.obs_buf[:, 8:11])
        self.proj_gravity_hist = torch.cat([t for t in self.proj_gravity_hist_deque_short],dim=1)

        if self.num_history_long:

            self.long_history_deque.append(torch.cat([self.obs_buf[:, 11:23], 
                                                    self.obs_buf[:, 23:35],
                                                    self.obs_buf[:, 35:47],
                                                    self.obs_buf[:, 5:8],
                                                    self.obs_buf[:, 8:11]
                                                    ], dim=1))
        
            self.long_history = torch.cat([t for t in self.long_history_deque],dim=1)



            
        
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
        
        return env_ids, terminal_amp_states

    def _post_physics_step_callback(self):
        static = (self.commands[:, 0] == 0) & (self.commands[:, 1] == 0) & (self.commands[:, 2] == 0)
        low_speed = (torch.norm(self.base_lin_vel[:, :2], dim=1) < 0.2)
        self.time_to_stand_still += 1. * static
        # self.time_to_stand_still *= low_speed
        self.time_to_stand_still *= static * self.stop
        self.phase = torch.fmod(self.phase + self.dt, 1.0)



        env_ids = (
            self.episode_length_buf
            % (self.resampling_time / self.dt).to(torch.int32) == 0) \
            .nonzero(as_tuple=False).flatten()
        self.recommand_ids = env_ids


        # if len(env_ids):
        #     print("set")
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.commands.resampling_time == -1 :
            # print(self.commands)
            pass  # when the joystick is used, the self.commands variables are overridden
        else:
            self._resample_commands(env_ids)

            if ( self.cfg.domain_rand.push_robots and
                (self.common_step_counter
                % self.cfg.domain_rand.push_interval == 0)):
                # self._push_robots()
                random_number = round(random.uniform(0, 1), 2)
                if random_number > 1-self.cfg.domain_rand.push_ratio:
                    # print('push time=',self.common_step_counter)
                    self._push_robots()

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ , _, _= self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        self.pre_loc[:, :] = self.root_states[:, :3]
        _, _, cur_yaw = get_euler_xyz(self.root_states[:, 3:7])
        self.pre_yaw[:] = cur_yaw[:]
        return obs, privileged_obs
    
    
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
  
      
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)

        # reset robot states
        self._reset_system(env_ids)
        if hasattr(self, "_custom_reset"):
            self._custom_reset(env_ids)
        self._resample_commands(env_ids)
        # reset buffers
        self.ctrl_hist[env_ids] = 0.
        self.dof_pos_hist[env_ids] = 0.
        self.dof_vel_hist[env_ids] = 0.
        self.base_ang_vel_hist[env_ids] = 0.
        self.proj_gravity_hist[env_ids] = 0.

        self.feet_air_time[env_ids] = 0.
        self.feet_stand_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.last_torques[env_ids] = 0.

        # self.actions_record[env_ids] = 0.

        #reset deques
        for i in range(self.num_history_short):
            self.ctrl_hist_deque_short[i][env_ids] *= 0.
            self.dof_pos_hist_deque_short[i][env_ids] *= 0.
            self.dof_vel_hist_deque_short[i][env_ids] *= 0.
            self.base_ang_vel_hist_deque_short[i][env_ids] *= 0.
            self.proj_gravity_hist_deque_short[i][env_ids] *= 0.
            
        for i in range(self.num_history_long):
            self.long_history_deque[i][env_ids] *= 0.


       

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
    
    def _push_robots(self):
        # Randomly pushes the robots.
        # Emulates an impulse by setting a randomized base velocity.

        # max_vel = self.cfg.domain_rand.max_push_vel_xy
        # self.root_states[:, 7:8] = torch_rand_float(
        #     -max_vel, max_vel, (self.num_envs, 1), device=self.device)
        # self.gym.set_actor_root_state_tensor(
        #     self.sim, gymtorch.unwrap_tensor(self.root_states))
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)  # lin vel x/y
        self.root_states[:, 7:9] = self.rand_push_force[:, :2]

        self.rand_push_torque = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device)

        self.root_states[:, 10:13] = self.rand_push_torque

        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))
        
        print("pushed")

    def check_termination(self):
        """ Check if environments need to be reset
        """
        # Termination for contact
        term_contact = torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :],
                dim=-1)
        self.reset_buf = torch.any((term_contact > 1.), dim=1)


        # Termination for velocities, orientation, and low height
        self.reset_buf |= torch.any(
          torch.norm(self.base_lin_vel, dim=-1, keepdim=True) > 10., dim=1)

        self.reset_buf |= torch.any(
          torch.norm(self.base_ang_vel, dim=-1, keepdim=True) > 5., dim=1)

        self.reset_buf |= torch.any(
          torch.abs(self.projected_gravity[:, 0:1]) > 0.7, dim=1)

        self.reset_buf |= torch.any(
          torch.abs(self.projected_gravity[:, 1:2]) > 0.7, dim=1)


        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1).unsqueeze(1)
        self.reset_buf |= torch.any(base_height < 0.3, dim=1)

        # self.reset_buf |= torch.any(
        #     torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 20000., dim=1)


        # # no terminal reward for time-outs
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.resampling_time[env_ids] = torch_rand_float(self.min_sample_time, self.max_sample_time, (len(env_ids), 1), device=self.device).squeeze(1)
        
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.robot_height_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["robot_height"][0], self.command_ranges["robot_height"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.ang_vel_command:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.time_to_stand_still[env_ids] = 0.

        # set small commands to zero
        self.commands[env_ids, 0] *= (self.commands[env_ids, 0] > self.cfg.commands.lin_vel_x_clip)|(self.commands[env_ids, 0] < -self.cfg.commands.lin_vel_x_clip)
        self.commands[env_ids, 1] *= (self.commands[env_ids, 1] > self.cfg.commands.lin_vel_y_clip)|(self.commands[env_ids, 1] < -self.cfg.commands.lin_vel_y_clip)
        self.commands[env_ids, 2] *= (self.commands[env_ids, 2] > self.cfg.commands.ang_vel_yaw_clip)|(self.commands[env_ids, 2] < -self.cfg.commands.ang_vel_yaw_clip)
        # if len(env_ids):
        #     print(self.commands[:, 2])
        command_mode = torch.rand((len(env_ids), 1), device=self.device).squeeze(1)
        # if len(env_ids) > 0:
        #     self.commands[env_ids,:2] = torch.where((self.terrain_types[env_ids]<4).unsqueeze(1),
        #                                             self.commands[env_ids,:2],
        #                                             0.5*self.commands[env_ids,:2])

        if self.cfg.commands.always_walking:
            command_mode += 1.

        # 0-0.1 stand still (all = 0)
        # 0.1-0.2 turn (ang vel != 0)
        # 0.2-0.4 walk along y axis (y vel != 0)
        # 0.4-0.6 walk along x axis (x vel != 0)
        # 0.6-1 hybrid (all != 0)
      

        self.commands[env_ids, 0] *= command_mode >= 0.4 # x vel
        self.commands[env_ids, 1] *= ((command_mode >= 0.2) & (command_mode < 0.4)) | (command_mode >= 0.6) # y vel
        self.commands[env_ids, 2] *= ((command_mode >0.1) & (command_mode < 0.2)) | (command_mode >= 0.6) # ang vel


        self.max_feet_air_time[env_ids, :] =torch.zeros((len(env_ids), 2), device=self.device)
        self.max_feet_stand_time[env_ids, :] =torch.zeros((len(env_ids), 2), device=self.device)
        self.max_feet_height[env_ids, :] = torch.zeros((len(env_ids), 2), device=self.device)

        self.command_error_accumulate[env_ids, :] = torch.zeros((len(env_ids), 2), device=self.device, requires_grad=False)
        self.command_ang_error_accumulate[env_ids, :] = torch.zeros((len(env_ids), 1), device=self.device, requires_grad=False)
        contacts = self.contact_forces[:, self.feet_indices, 2] > 5.
        double_contact = torch.sum(1.*contacts, dim=-1) == 2
        self.time_to_stand_still[env_ids] += self.static_delay * double_contact[env_ids] * \
            (torch.norm(self.commands[env_ids, :3], dim=-1) == 0) #* \
            #(torch.norm(self.base_lin_vel[env_ids, :2], dim=-1) < 0.3)
        self.stop[env_ids,] = torch.randint(0,2,(len(env_ids),),device=self.device) 

        
        
            
    
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        
        # pd controller

        # if self.cfg.control.exp_avg_decay:
        #     self.action_avg = exp_avg_filter(self.actions, self.action_avg,
        #                                     self.cfg.control.exp_avg_decay)
        #     actions = self.action_avg

        if self.cfg.control.control_type=="P":
            torques = self.random_p * self.p_gains*(actions * self.cfg.control.action_scale \
                                    + self.default_dof_pos \
                                    - self.dof_pos) \
                    - self.random_d * self.d_gains*self.dof_vel

        elif self.cfg.control.control_type=="T":
            torques = actions * self.cfg.control.action_scale

        elif self.cfg.control.control_type=="Td":
            torques = actions * self.cfg.control.action_scale \
                        - self.random_d * self.d_gains*self.dof_vel

        else:
            raise NameError(f"Unknown controller type: {self.cfg.control.control_type}")
   
        if self.cfg.domain_rand.random_dof_friction:
            rng = self.cfg.domain_rand.dof_friction_range
            self.dof_friction = torch_rand_float(rng[0], rng[1], 
                                    shape=(self.num_envs, self.num_dof), device=self.device)
            torques = torch.where(self.dof_vel > 0., 
                torch.where(torch.logical_and(torques > 0, torques - self.dof_friction < 0), torch.zeros_like(torques), torques - self.dof_friction),
                torch.where(torch.logical_and(torques < 0, torques + self.dof_friction > 0), torch.zeros_like(torques), torques + self.dof_friction))
                #torques - self.dof_friction, torques + self.dof_friction)
            
            
            

        return torch.clip(torques, -self.torque_limits, self.torque_limits)
    
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
            self.friction_buf[env_id, :] = self.friction_coeffs[env_id]
        # default value of frictions are 1.0
        return props
    
    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
   
                if i == 5 or i == 11:
                    self.torque_limits[i] /= 2.
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r \
                                           *self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit


        rng = self.cfg.domain_rand.damping_range
        for i in range(len(props)):
            if self.cfg.domain_rand.random_damping:
                rd_num = np.random.uniform(rng[0], rng[1])
                self.damping_buf[env_id, i] = rd_num
                props["damping"][i] = rd_num
            else:
                self.damping_buf[env_id, i] = props["damping"][i].item()

            # if self.cfg.domain_rand.randomize_joint_armature:
            #     if self.cfg.domain_rand.randomize_joint_armature_each_joint:
            #         props["armature"][i] = self.joint_armatures[env_id, i]
            #     else:
            #         props["armature"][i] = self.joint_armatures[env_id, 0]


            if self.cfg.domain_rand.randomize_joint_armature:
                props["armature"][i] = self.joint_armatures[i] * self.random_armature[env_id, i]
            else:
                props["armature"][i] = self.joint_armatures[i]

        
        return props
    
    def _process_rigid_body_props(self, props, env_id):
        if env_id==0:
            m = 0
            for i, p in enumerate(props):
                m += p.mass
            #     print(f"Mass of body {i}: {p.mass} (before randomization)")
            # print(f"Total mass {m} (before randomization)")
            self.mass_total = m

        # randomize mass of all link
        if self.cfg.domain_rand.randomize_all_mass:
            for s in range(len(props)):
                rng = self.cfg.domain_rand.rd_mass_range
                rd_num = np.random.uniform(rng[0], rng[1])
                self.mass_mask[env_id, s] = rd_num
                props[s].mass *= rd_num
        
        # randomize base mass
        if self.cfg.domain_rand.randomize_load_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[-1].mass += np.random.uniform(rng[0], rng[1])
        
        # randomize com of all link other than base link
        if self.cfg.domain_rand.randomize_com:
            for s in range(len(props)-1):
                
                rng = self.cfg.domain_rand.rd_com_range
                rd_num = np.random.uniform(rng[0], rng[1])
                self.com_diff_x[env_id, s] = rd_num
                props[s].com.x += rd_num
                rd_num = np.random.uniform(rng[0], rng[1])
                self.com_diff_y[env_id, s] = rd_num
                props[s].com.y += rd_num
                rd_num = np.random.uniform(rng[0], rng[1])
                self.com_diff_z[env_id, s] = rd_num
                props[s].com.z += rd_num

        # randomize com of base link
        if self.cfg.domain_rand.randomize_base_com:
            rng = self.cfg.domain_rand.rd_base_com_range
            rd_num = np.random.uniform(rng[0], rng[1])
            self.com_diff_x[env_id, -1] = rd_num
            props[-1].com.x += rd_num
            rd_num = np.random.uniform(rng[0], rng[1])
            self.com_diff_y[env_id, -1] = rd_num
            props[-1].com.y += rd_num
            rd_num = np.random.uniform(rng[0], rng[1])
            self.com_diff_z[env_id, -1] = rd_num
            props[-1].com.z += rd_num

        # randomize inertia of all body
        if self.cfg.domain_rand.random_inertia:
            rng = self.cfg.domain_rand.inertia_range
            for s in range(len(props)):
                rd_num = np.random.uniform(rng[0], rng[1])
                self.inertia_mask_xx[env_id, s] = rd_num
                props[s].inertia.x.x *= rd_num
                
                rd_num = np.random.uniform(rng[0], rng[1])
                self.inertia_mask_xy[env_id, s] = rd_num
                props[s].inertia.x.y *= rd_num
                props[s].inertia.y.x *= rd_num

                rd_num = np.random.uniform(rng[0], rng[1])
                self.inertia_mask_xz[env_id, s] = rd_num
                props[s].inertia.x.z *= rd_num
                props[s].inertia.z.x *= rd_num

                rd_num = np.random.uniform(rng[0], rng[1])
                self.inertia_mask_yy[env_id, s] = rd_num
                props[s].inertia.y.y *= rd_num

                rd_num = np.random.uniform(rng[0], rng[1])
                self.inertia_mask_yz[env_id, s] = rd_num
                props[s].inertia.y.z *= rd_num
                props[s].inertia.z.y *= rd_num

                rd_num = np.random.uniform(rng[0], rng[1])
                self.inertia_mask_zz[env_id, s] = rd_num
                props[s].inertia.z.z *= rd_num
        
        
        return props

    def _reset_system(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids

        # todo: make separate methods for each reset type, cycle through `reset_mode` and call appropriate method. That way the base ones can be implemented once in legged_robot.
        """

        if hasattr(self, self.cfg.init_state.reset_mode):
            eval(f"self.{self.cfg.init_state.reset_mode}(env_ids)")
        else:
            raise NameError(f"Unknown default setup: {self.cfg.init_state.reset_mode}")

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # start base position shifted in X-Y plane
        if self.custom_origins:
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            # xy position within 1m of the center
            # self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device)
        else:
            self.root_states[env_ids] = self.root_states[env_ids]
            self.root_states[env_ids, :3] += self.env_origins[env_ids] 

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                    gymtorch.unwrap_tensor(self.root_states),
                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        
        #make robot on the ground

        self.gym.simulate(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # lower_foot = torch.where(self._rigid_body_pos[env_ids, self.feet_indices[0], 2] - self._get_foot_terrain_heights()[env_ids, 0] <= self._rigid_body_pos[env_ids, self.feet_indices[1], 2] - self._get_foot_terrain_heights()[env_ids, 1], \
        #         self._rigid_body_pos[env_ids, self.feet_indices[0], 2] - self._get_foot_terrain_heights()[env_ids, 0],\
        #                              self._rigid_body_pos[env_ids, self.feet_indices[1], 2] - self._get_foot_terrain_heights()[env_ids, 1])
        # # if len(env_ids) > 10:
        # #     print(self.root_states[0:10, 2])
        # #     print(self._rigid_body_pos[0:10, self.feet_indices[0], 2], "++++++")
        # #     print(self._rigid_body_pos[0:10, self.feet_indices[1], 2], "******")
        

        # # self.root_states[env_ids, 2] -= (lower_foot - 0.09 - 0.1*torch.rand_like(self.root_states[env_ids, 2]))
        # self.root_states[env_ids, 2] -= (lower_foot  - 0.1*torch.rand_like(self.root_states[env_ids, 2]))

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                    gymtorch.unwrap_tensor(self.root_states),
                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.simulate(self.sim)

    # * implement reset methods
    def reset_to_basic(self, env_ids):
        """
        Reset to a single initial state
        """
        #dof 
        self.dof_pos[env_ids] = self.default_dof_pos  #torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0 
        self.root_states[env_ids] = self.base_init_state

    def reset_to_range(self, env_ids):
        """
        Reset to a uniformly random distribution of states, sampled from a
        range for each state
        """
        # dof states
        self.dof_pos[env_ids] = random_sample(env_ids,
                                    self.dof_pos_range[:, 0],
                                    self.dof_pos_range[:, 1],
                                    device=self.device)
        self.dof_vel[env_ids] = random_sample(env_ids,
                        self.dof_vel_range[:, 0],
                        self.dof_vel_range[:, 1],
                        device=self.device)

        # base states
        random_com_pos = random_sample(env_ids,
                                    self.root_pos_range[:, 0],        
                                    self.root_pos_range[:, 1],
                                    device=self.device)

        quat = quat_from_euler_xyz(random_com_pos[:, 3],
                                        random_com_pos[:, 4],
                                        random_com_pos[:, 5]) 

        self.root_states[env_ids, 0:7] = torch.cat((random_com_pos[:, 0:3],
                                    quat_from_euler_xyz(random_com_pos[:, 3],
                                                        random_com_pos[:, 4],
                                                        random_com_pos[:, 5])),
                                                    1)
        self.root_states[env_ids, 7:13] = random_sample(env_ids,
                                    self.root_vel_range[:, 0],
                                    self.root_vel_range[:, 1],
                                    device=self.device)

    def mixed_reset(self, env_ids):
        rd_num = np.random.uniform(0., 1.)
        if self.cfg.init_state.range_pb:
            limit = self.cfg.init_state.range_pb
        else:
            limit = 0.5
        if rd_num <= limit:
            self.reset_to_range(env_ids)
        else:
            self.reset_to_basic(env_ids)


    
    
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # find ids for end effectors defined in env. specific config files
        ee_ids = []
        kp_ids = []
        for body_name in self.cfg.asset.end_effectors:
            ee_id = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], body_name)
            ee_ids.append(ee_id)
        for keypoint in self.cfg.asset.keypoints:
            kp_id = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], keypoint)
            kp_ids.append(kp_id)
        self.end_eff_ids = to_torch(ee_ids, device=self.device, dtype=torch.long)
        self.keypoint_ids = to_torch(kp_ids, device=self.device, dtype=torch.long)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim, "legged_robot")
        mass_matrix_tensor = self.gym.acquire_mass_matrix_tensor(self.sim, "legged_robot")

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        self.jacobians = gymtorch.wrap_tensor(jacobian_tensor)
        self.mass_matrices = gymtorch.wrap_tensor(mass_matrix_tensor)
        
        self._rigid_body_pos = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 0:3]
        self._rigid_body_ori = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 3:7]
        self._rigid_body_vel = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 7:10]
        self._rigid_body_ang = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 10:13]
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_pos = self.root_states[:, 0:3]
        self.base_quat = self.root_states[:, 3:7]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        #foot sensors
        # sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        # self.gym.refresh_force_sensor_tensor(self.sim)
        # force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor)
        # self.sensor_forces = force_sensor_readings.view(self.num_envs, 4, 6)[..., :3]

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        # self.noise_scale_vec = self._get_noise_scale_vec(self.cfg) # move to custom init
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx),
                                device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.],
                                device=self.device).repeat((self.num_envs, 1))

        # self.torques = torch.zeros(self.num_envs, self.num_actions,
        #                            dtype=torch.float, device=self.device,
        #                            requires_grad=False)
        # SE HWAN CRIME (?): why are the torques the same dimension as the output of the neural network?
        # They shouldn't need to be...

        self.torques = torch.zeros(self.num_envs, self.num_dof,
                                   dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_torques = torch.zeros_like(self.torques)
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float,
                                    device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float,
                                    device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions,
                                   dtype=torch.float, device=self.device,
                                   requires_grad=False)
        # * additional buffer for last ctrl: whatever is actually used for PD control (which can be shifted compared to action)
        self.ctrl_hist = torch.zeros(self.num_envs, self.num_actions*3,
                                     dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.dof_pos_hist = torch.zeros(self.num_envs, self.num_dof*3,
                                     dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.dof_vel_hist = torch.zeros(self.num_envs, self.num_dof*3,
                                     dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.base_ang_vel_hist = torch.zeros(self.num_envs, 9,
                                     dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.proj_gravity_hist = torch.zeros(self.num_envs, 9,
                                     dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.commands = torch.zeros(self.num_envs,
                                    self.cfg.commands.num_commands,
                                    dtype=torch.float, device=self.device,
                                    requires_grad=False) # x vel, y vel, yaw vel, height
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel,
                                            self.obs_scales.lin_vel,
                                            self.obs_scales.ang_vel],
                                           device=self.device,
                                           requires_grad=False,)
        self.feet_air_time = torch.zeros(self.num_envs,
                                         self.feet_indices.shape[0],
                                         dtype=torch.float,
                                         device=self.device,
                                         requires_grad=False)
        self.max_feet_air_time = torch.zeros(self.num_envs,
                                         self.feet_indices.shape[0],
                                         dtype=torch.float,
                                         device=self.device,
                                         requires_grad=False)
        self.feet_stand_time = torch.zeros(self.num_envs,
                                         self.feet_indices.shape[0],
                                         dtype=torch.float,
                                         device=self.device,
                                         requires_grad=False)
        self.max_feet_stand_time = torch.zeros(self.num_envs,
                                         self.feet_indices.shape[0],
                                         dtype=torch.float,
                                         device=self.device,
                                         requires_grad=False)
        self.max_feet_height = torch.zeros(self.num_envs,
                                         self.feet_indices.shape[0],
                                         dtype=torch.float,
                                         device=self.device,
                                         requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs,
                                         len(self.feet_indices),
                                         dtype=torch.bool,
                                         device=self.device,
                                         requires_grad=False)
        self.time_to_stand_still = torch.zeros(self.num_envs,
                                         dtype=torch.float,
                                         device=self.device,
                                         requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat,
                                                self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat,
                                                self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat,
                                                     self.gravity_vec)

        self.foot1_projected_gravity = quat_rotate_inverse(self._rigid_body_ori[:, self.feet_indices[0], :], self.gravity_vec)
        self.foot2_projected_gravity = quat_rotate_inverse(self._rigid_body_ori[:, self.feet_indices[1], :], self.gravity_vec)

        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        if self.cfg.control.exp_avg_decay:
            self.action_avg = torch.zeros(self.num_envs, self.num_actions,
                                            dtype=torch.float,
                                            device=self.device, requires_grad=False)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float,
                                           device=self.device,
                                           requires_grad=False)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = self.cfg.init_state.default_joint_angles[name]
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.rs_w = torch.max(self.p_gains)/self.p_gains
        # * check that init range highs and lows are consistent
        # * and repopulate to match 
        if self.cfg.init_state.reset_mode == "reset_to_range" or self.cfg.init_state.reset_mode == "mixed_reset":
            self.dof_pos_range = torch.zeros(self.num_dof, 2,
                                            dtype=torch.float,
                                            device=self.device,
                                            requires_grad=False)
            self.dof_vel_range = torch.zeros(self.num_dof, 2,
                                            dtype=torch.float,
                                            device=self.device,
                                            requires_grad=False)

            for joint, vals in self.cfg.init_state.dof_pos_range.items():
                for i in range(self.num_dof):
                    if joint in self.dof_names[i]:
                        self.dof_pos_range[i, :] = to_torch(vals)

            for joint, vals in self.cfg.init_state.dof_vel_range.items():
                for i in range(self.num_dof):
                    if joint in self.dof_names[i]:
                        self.dof_vel_range[i, :] = to_torch(vals)

            self.root_pos_range = torch.tensor(self.cfg.init_state.root_pos_range,
                    dtype=torch.float, device=self.device, requires_grad=False)
            self.root_vel_range = torch.tensor(self.cfg.init_state.root_vel_range,
                    dtype=torch.float, device=self.device, requires_grad=False)
            # todo check for consistency (low first, high second)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.cfg.terrain.horizontal_scale
        hf_params.row_scale = self.cfg.terrain.horizontal_scale
        hf_params.vertical_scale = self.cfg.terrain.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.cfg.terrain.border_size 
        hf_params.transform.p.y = -self.cfg.terrain.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution
        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _get_foot_terrain_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, 2, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = self._rigid_body_pos[env_ids, self.feet_indices, :3]
        else:
            points = self._rigid_body_pos[:, self.feet_indices, :3]

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        ###################  debug   ##############################
        # heights = heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        # self.gym.clear_lines(self.viewer)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)
        # sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        # for i in range(self.num_envs):
        #     heights4 = heights[i].cpu().numpy()
        #     for j in range(heights4.shape[0]):
        #         x = self._rigid_body_pos[i, self.feet_indices[j], 0].cpu().numpy()
        #         y = self._rigid_body_pos[i, self.feet_indices[j], 1].cpu().numpy()
        #         z = heights4[j]
        #         sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
        #         gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        
# ########################## REWARDS ######################## #

    # * "True" rewards * #

    def _reward_tracking_lin_vel(self):
        # Reward tracking specified linear velocity command
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        error *= 1./(1. + torch.abs(self.commands[:, :2]))
        # error = torch.sum(torch.square(error), dim=1)

        error = torch.where(self.time_to_stand_still > self.static_delay, 
                                    torch.sum(torch.abs(error), dim=1), \
                                    torch.sum(torch.square(error), dim=1))
        return torch.exp(-2. * error/self.cfg.rewards.tracking_sigma) * self.smooth_w

    def _reward_tracking_ang_vel(self):
        # Reward tracking yaw angular velocity command
        ang_vel_error = torch.where(self.time_to_stand_still > self.static_delay, 
                                    torch.abs((self.commands[:, 2] - self.base_ang_vel[:, 2])), \
                                    torch.square((self.commands[:, 2] - self.base_ang_vel[:, 2])))
        return torch.exp(-2. * ang_vel_error/self.cfg.rewards.tracking_sigma) * self.smooth_w
    
    def _reward_waiting(self):

        return 1. * torch.logical_or((((self.commands[:, :2] * self.base_lin_vel[:, :2]).sum(axis = 1))/(torch.norm(self.commands[:, :2],dim=1) + 1e-8) < 0.2) \
                    * (torch.norm(self.commands[:, :2],dim=1) !=  0.),
                    (torch.norm(self.base_lin_vel[:, :2]) > 0.2) * (torch.norm(self.commands[:, :2],dim=1) ==  0.)) 

    def _reward_tracking_avg(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        self.command_error_accumulate = 0.99 * self.command_error_accumulate + 0.01 * (self.base_lin_vel[:, :2] - self.commands[:, :2] * (((self.commands[:, :2] * self.base_lin_vel[:, :2]).sum(axis = 1))/torch.square(torch.norm(self.commands[:, :2],dim=1) + 1e-8)).unsqueeze(1))
        self.command_ang_error_accumulate = 0.99 * self.command_ang_error_accumulate + 0.01 * (self.commands[:,2] - self.base_ang_vel[:,2]).unsqueeze(1)
        # Tracking of linear velocity commands (xy axes)
        # self.command_error_accumulate = torch.where(self.time_to_stand_still.unsqueeze(1) >  0., 
        #                             self.command_error_accumulate * 0., \
        #                             self.command_error_accumulate
        #                             )

        # self.command_ang_error_accumulate = torch.where(self.time_to_stand_still >  0., 
        #                             self.command_ang_error_accumulate * 0., \
        #                             self.command_ang_error_accumulate
        #                             )        
       
        lin_vel_error = torch.norm(self.commands[:, :2] - self.commands[:, :2] * (((self.commands[:, :2] * self.base_lin_vel[:, :2]).sum(axis = 1))/torch.square(torch.norm(self.commands[:, :2],dim=1) + 1e-8)).unsqueeze(1)
                                               , dim=1)
  
        lin_vel_error2 = torch.norm(self.base_lin_vel[:, :2] - self.commands[:, :2] * (((self.commands[:, :2] * self.base_lin_vel[:, :2]).sum(axis = 1))/torch.square(torch.norm(self.commands[:, :2],dim=1) + 1e-8)).unsqueeze(1)
                                               , dim=1)
     
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10) * 1. + torch.exp(-torch.norm(self.command_error_accumulate, dim=1) * 50 * (self.time_to_stand_still <= self.static_delay))

        # Tracking of angular velocity commands (yaw)
        # ang_vel_error = 5 * torch.abs(
        #     self.command_ang_error_accumulate).squeeze(1)
        ang_vel_error = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-torch.abs(self.command_ang_error_accumulate).squeeze(1) * (self.time_to_stand_still <= self.static_delay) * 50)

        linear_error = 0.2 * (lin_vel_error + lin_vel_error2 + ang_vel_error)
        # print(self.command_error_accumulate) 下面这个是考虑
        return (lin_vel_error_exp + ang_vel_error_exp) / 2. * self.smooth_w - linear_error# - torch.norm(self.command_error_accumulate, dim=1) - torch.abs(self.command_ang_error_accumulate).squeeze(1)


    def _reward_torques(self):
        # Penalize torques
        # print(torch.exp(-0.02*(torch.sum(torch.abs(self.torques)/self.torque_limits, dim=1)/self.num_dof)))
 
        return torch.exp(-0.02*(torch.sum(torch.abs(self.torques)/self.torque_limits, dim=1)/self.num_dof))
    
    def _reward_torques_var(self):
   
        # Penalize torques
        # print("right: " ,self.torques[:,5],"left: ", self.torques[:,11])
        # print("***************")
        # print(self.fatigue)
        # print("@@@@@@@@@@@@@@@")
        # print(torch.exp(-torch.sum(torch.abs(self.fatigue[:,:6]-self.fatigue[:,6:])/((self.fatigue[:,:6]+self.fatigue[:,6:])/2.), dim=1)/6.))
        # print(torch.exp(-0.04*torch.sum(torch.abs(self.fatigue[:,:6]-self.fatigue[:,6:])/((self.fatigue[:,:6]+self.fatigue[:,6:])/6. + 1e-8), dim=1)))
        return torch.exp(-torch.sum(torch.abs(self.fatigue[:,:6]-self.fatigue[:,6:])/((self.fatigue[:,:6]+self.fatigue[:,6:])/2. + 1e-8), dim=1)/6.)
    
    def _reward_delta_torques(self):
        return torch.sum(torch.square((self.torques - self.last_torques)/(self.fatigue_torque + 1e-8)), dim=1)/self.num_dof
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = (-(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)) > 0. # lower limit
        out_of_limits += ((self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)) > 0.
        return torch.sum(out_of_limits * 1., dim=1)
    
    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum(((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.) > 0.) * 1., dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    # * Shaping rewards * #

    def _reward_base_height(self):
        # Reward tracking specified base height
        ############### change later ##################
        
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        # print(base_height)
        error = (base_height - (0.05 * self.commands[:,3] + 1.03)) # command[:,3] 是指heading
        error = error.flatten()
        return torch.exp(-1000. * torch.square(error)) * self.smooth_w
    
    def _reward_base_height_stand(self):
        # Reward tracking specified base height
        
        ############### change later ##################
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        # print(base_height)
        error = (base_height - (0.05 * self.commands[:,3] + 1.03))
        error = error.flatten()
        return -100. * torch.square(error) #* (self.time_to_stand_still > self.static_delay)

    def _reward_orientation(self):
        # Reward tracking upright orientation
        
        error = torch.where(self.time_to_stand_still > self.static_delay, 
                            torch.norm(self.projected_gravity[:, :2],dim=-1), \
                            torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1))
        # error = torch.norm(self.projected_gravity[:, :2],dim=-1)
        # error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return torch.exp(-2.*error/self.cfg.rewards.tracking_sigma) * self.smooth_w

    def _reward_joint_regularization(self):
        # Reward joint poses and symmetry
        
        error = 0.
        # Yaw joints regularization around 0
        error += self.sqrdexp(
            2.*(self.dof_pos[:, 1]) / self.cfg.normalization.obs_scales.dof_pos)
        # # print("reward")
        # print(self.sqrdexp(
        #     10.*(self.dof_pos[:, 0]) / self.cfg.normalization.obs_scales.dof_pos))
        # # print("dof pos 2")
        # # print(self.dof_pos[:, 3])
        error += self.sqrdexp(
            2.*(self.dof_pos[:, 7]) / self.cfg.normalization.obs_scales.dof_pos)
        # Ab/ad joint regularization around 0
        error += self.sqrdexp(
            2.*(self.dof_pos[:, 0])
            / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp(
            2.*(self.dof_pos[:, 6])
            / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp(
            2.*(self.dof_pos[:, 5])
            / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp(
            2.*(self.dof_pos[:, 11])
            / self.cfg.normalization.obs_scales.dof_pos)
        # Pitch joint symmetry
        # error += self.sqrdexp(
        #     ((self.dof_pos[:, 2] + self.dof_pos[:, 8]) / 2. - self.cfg.init_state.default_joint_angles['Joint-hip-r-pitch'])
        #     / self.cfg.normalization.obs_scales.dof_pos)
        # print(error/6)
        return error/4 * self.smooth_w
    

    def _reward_ankle_regularization(self):
        # Ankle joint regularization around 0
       
        error = torch.exp(-2.*torch.norm(self.foot1_projected_gravity[:, :2],dim=-1)/self.cfg.rewards.tracking_sigma)
        error += torch.exp(-2.*torch.norm(self.foot2_projected_gravity[:, :2],dim=-1)/self.cfg.rewards.tracking_sigma)
        # error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return error/2. * self.smooth_w
        # error = 0
        # error += self.sqrdexp(
        #     (10.*self.dof_pos[:, 5]) / self.cfg.normalization.obs_scales.dof_pos)
        # error += self.sqrdexp(
        #     (10.*self.dof_pos[:, 11]) / self.cfg.normalization.obs_scales.dof_pos)
        # # error += self.sqrdexp(
        # #     (self.dof_pos[:, 4]-self.cfg.init_state.default_joint_angles['Joint-ankel-l-pitch']) / self.cfg.normalization.obs_scales.dof_pos)
        # # error += self.sqrdexp(
        # #     (self.dof_pos[:, 10]-self.cfg.init_state.default_joint_angles['Joint-ankel-r-pitch']) / self.cfg.normalization.obs_scales.dof_pos)
        # return error / 4. * self.smooth_w

    # Added Reward ---------------------------------------------
    def _reward_feet_contact(self):
      
        # reward one-foot contact when moving
        contacts = self.contact_forces[:, self.feet_indices, 2] > 5.
        no_contact = torch.sum(1.*contacts, dim=1)==0
        single_contact = torch.sum(1.*contacts, dim=1)==1
        self.single_contact_last = torch.where(single_contact, self.single_contact_last + 1.,torch.zeros_like(self.single_contact_last))
        double_contact = torch.sum(1.*contacts, dim=1)==2

        move = (torch.norm(self.commands[:, :3], dim=1) > 0.) 
        self.double_contact_last = torch.where(double_contact, self.double_contact_last + 1.,torch.zeros_like(self.double_contact_last))
        # move_stand = torch.norm(foot_pos_left[:,:2] + foot_pos_right[:,:2], dim = 1) * low_speed * double_contact > 0.18
        move_stand = (self.double_contact_last > 10) * move
        # print(torch.where(self.time_to_stand_still > self.static_delay, double_contact * 1.,  #torch.ones_like(self.time_to_stand_still), \
        #                           1.*single_contact - 1. * double_contact))
        # print(self.single_contact_last)

     
        return torch.where(self.time_to_stand_still > self.static_delay, torch.ones_like(self.time_to_stand_still), \
                                  1.*single_contact) * self.smooth_w
        #- 2. * no_contact - 1. * single_contact - 1. * move_stand
        
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        first_swing = (self.feet_stand_time > 0.) * ~contact_filt

        self.feet_air_time += self.dt  # todo: pull this out into post-physics
        self.feet_stand_time += self.dt
        tabu = (self.commands[:, 0] == 0) & (self.commands[:, 1] == 0) & (self.commands[:, 2] == 0)
        T_air = 0.3
        rew_airTime = 4./self.feet_indices.size()[0] * torch.where(self.time_to_stand_still > self.static_delay, torch.sum(torch.ones_like(self.feet_air_time), dim=1)*self.smooth_w, \
                                  torch.sum(torch.where(self.feet_air_time < (T_air + 0.06), self.feet_air_time.clip(max=T_air)/T_air * self.feet_air_time/self.dt * first_contact, torch.zeros_like(self.feet_air_time))
                                            # + 9. * (self.feet_air_time - 0.4) * first_contact
                                            * self.smooth_w.unsqueeze(1) #* first_contact#, dim=1))
                                             , dim=1))  # reward only on first contact with the ground
        
        # rew_airTime = 6 * self.cfg.rewards.smooth_w * torch.where(self.time_to_stand_still > self.static_delay, 0. * torch.ones_like(self.time_to_stand_still), \
        #                           torch.sum(((self.feet_air_time - 0.4).clip(max=0.)) * first_contact#, dim=1))
        #                                      - tabu.unsqueeze(1) * 100.*(self.feet_air_time - 0.4).clip(min = 0.), dim=1))  # reward only on first contact with the ground
        if first_contact[0].any():
            print("swing:", self.feet_air_time[0] * first_contact[0])
        T_stand = (0.5/torch.norm(self.commands[:, :2],dim = 1)).clip(min = 0.1,max = 0.4).unsqueeze(1)

        rew_standTime = 4./self.feet_indices.size()[0] * torch.where(self.time_to_stand_still > self.static_delay, torch.sum(torch.ones_like(self.feet_stand_time), dim=1)*self.smooth_w, \
                                  torch.sum(torch.where(self.feet_stand_time < (T_stand + 0.06), self.feet_stand_time.clip(max=T_stand)/T_stand * self.feet_stand_time/self.dt*first_swing, torch.zeros_like(self.feet_stand_time))
                                            * self.smooth_w.unsqueeze(1) #* first_contact#, dim=1))
                                             , dim=1))  # reward only on first contact with the ground
        if first_swing[0].any():
            print("contanct:", self.feet_stand_time[0] * first_swing[0])
        self.feet_air_time *= ~contact_filt
        self.feet_stand_time *= contact_filt * (self.time_to_stand_still <= self.static_delay).unsqueeze(1) #这里需要区分是行走站立时间和永久站立
        self.max_feet_air_time = torch.where(self.max_feet_air_time > self.feet_air_time, self.max_feet_air_time, self.feet_air_time)
        self.max_feet_stand_time = torch.where(self.max_feet_stand_time > self.feet_stand_time, self.max_feet_stand_time, self.feet_stand_time)
        return rew_airTime #+ rew_standTime#- acc_reward/100.
    
    def _reward_feet_air_time_diff(self):
        error = torch.where(self.time_to_stand_still > self.static_delay, 
                            torch.zeros_like(torch.var(self.max_feet_air_time[:, :],dim=1)),
                            torch.var(self.max_feet_air_time[:, :],dim=1)
                            + torch.var(self.max_feet_stand_time[:, :],dim=1)
                #  + torch.var(self.max_feet_stand_time[:, :]/(torch.mean(self.max_feet_stand_time[:, :],dim=1).unsqueeze(1) + 1e-8),dim=1)
                 )
        # print("air:",self.max_feet_air_time[0, ]) /(torch.mean(self.max_feet_air_time[:, :],dim=1).unsqueeze(1) + 1e-8)
        # print("stand:",self.max_feet_stand_time[0,])
        # print("error:",error[0])
        return error
    
    def _reward_base_acc(self):
        # penalize large linear acc of root base
        base_lin_acc = (self.lin_vel2 - self.lin_vel1) / self.dt
        base_ang_acc = (self.ang_vel2 - self.ang_vel1) / self.dt
        # print(0.001*(torch.sum(torch.square(base_lin_acc[:, :]), dim=1) + 0.02*torch.sum(torch.square(base_ang_acc[:, :]), dim=1))[0])
        return (torch.exp(-0.01*torch.sum(torch.abs(base_lin_acc[:, :]), dim=1))/2. + torch.exp(-0.002*torch.sum(torch.abs(base_ang_acc[:, :]), dim=1))/2.)* self.smooth_w
        # return torch.exp(-0.01*(torch.sum(torch.abs(base_lin_acc[:, :]), dim=1) + torch.sum(torch.abs(base_ang_acc[:, :]), dim=1))) * self.smooth_w

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_dofpower(self):
        return self.cfg.rewards.impulse_w * torch.sum(torch.abs(self.dof_vel/self.dof_vel_limits*self.torques/self.torque_limits)/self.num_dof, dim=-1)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.exp(-2 * torch.square(self.base_lin_vel[:, 2]))

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.exp(-1.5*torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1))
    
    def _reward_target_position(self):
        # reward = torch.exp(-100. * torch.square(torch.norm((self.pre_loc - self.root_states[:, :3])[:, :2], dim=-1)))
        if not self.cfg.rewards.accumulate_target:
            error = torch.where(torch.norm(self.commands[:,:2],dim=1) == 0, 
                                -100. * torch.square(torch.norm((self.pre_loc - self.root_states[:, :3])[:, :2], dim=-1)), 
                                -25000. * torch.square(torch.norm((self.pre_loc - self.root_states[:, :3])[:, :2], dim=-1)))
        else:
            error = -10. * torch.square(torch.norm((self.pre_loc - self.root_states[:, :3])[:, :2], dim=-1))
        reward = torch.exp(error)# + error/50.
        # print("position error2:", self.pre_loc)
        # print("root position:", self.root_states[:, :3])
        # print("position reward:", reward)
        return reward * self.smooth_w
    
    def _reward_target_orientation(self):
        # reward = torch.exp(-250. * torch.abs(torch.cos((self.yaw - self.pre_yaw)) - 1.))
        if not self.cfg.rewards.accumulate_target:
            error = torch.where(self.commands[:,2]==0, 
                                -200. * torch.abs(torch.cos((self.yaw - self.pre_yaw)) - 1.), 
                                -50000. * torch.abs(torch.cos((self.yaw - self.pre_yaw)) - 1.))
        else:
            error = -20. * torch.abs(torch.cos((self.yaw - self.pre_yaw)) - 1.)
        # print("orientation reward:", reward)
        reward = torch.exp(error)# + error/50.
        return reward * self.smooth_w
    
    def _reward_foot_pos(self):
        ypos = 0.189 / 2.
        xpos = 0.
        
        left_foot_sub = self._rigid_body_pos[:, self.feet_indices[0], :3] - self.root_states[:, :3]
        right_foot_sub = self._rigid_body_pos[:, self.feet_indices[1], :3] - self.root_states[:, :3]
        left_foot_sub_b = quat_rotate_inverse(self.base_quat, left_foot_sub)
        right_foot_sub_b = quat_rotate_inverse(self.base_quat, right_foot_sub)
        left_foot_sub_b[:,2] = 0.
        right_foot_sub_b[:,2] = 0.
        # print("left_foot_sub_b: ",left_foot_sub_b)
        # print("right_foot_sub_b: ",right_foot_sub_b)

        left_foot_error = torch.cat(((left_foot_sub_b[:, 0] - xpos).unsqueeze(1),
                                     (left_foot_sub_b[:, 1] - ypos).unsqueeze(1),), dim=-1)
        right_foot_error = torch.cat(((right_foot_sub_b[:, 0] - xpos).unsqueeze(1),
                                     (right_foot_sub_b[:, 1] + ypos).unsqueeze(1),), dim=-1)
        
        rew = -.5 * torch.norm(left_foot_error, dim=-1) * (self.time_to_stand_still > self.static_delay)
        # print(right_foot_error)
        rew += -.5 * torch.norm(right_foot_error, dim=-1) * (self.time_to_stand_still > self.static_delay)
        # left_foot_error = torch.cat(((left_foot_sub_b[:, 0] - xpos).unsqueeze(1),
        #                              3. * (left_foot_sub_b[:, 1] - ypos).unsqueeze(1),), dim=-1)
        # right_foot_error = torch.cat(((right_foot_sub_b[:, 0] - xpos).unsqueeze(1),
        #                              3. * (right_foot_sub_b[:, 1] + ypos).unsqueeze(1),), dim=-1)
        
        # rew = -.5 * torch.norm(left_foot_error, dim=-1) * torch.where(self.time_to_stand_still > self.static_delay, 1., 0.3)
        # # print(right_foot_error)
        # rew += -.5 * torch.norm(right_foot_error, dim=-1) * torch.where(self.time_to_stand_still > self.static_delay, 1., 0.3)
        
        #knee distance
        foot_pos = self._rigid_body_pos[:, self.knee_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = 0.18
        max_df = 1. / 2
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        n_out = 0.
        n_out += torch.abs(d_min) + torch.abs(d_max)
        # rew += (-torch.abs(d_min) * 20. - torch.abs(d_max) * 20.)

        #foot distance
        left_foot_pos = self._rigid_body_pos[:, self.feet_indices[0], :2]
        right_foot_pos = self._rigid_body_pos[:, self.feet_indices[1], :2]

        foot_dist = torch.norm(left_foot_pos - right_foot_pos, dim=1)
        fd = 0.18
        max_df = 1.
        # print(foot_dist)
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        # rew += (torch.exp(-torch.abs(d_min) * 100.) + torch.exp(-torch.abs(d_max) * 100.))#\
            # + torch.exp(-500. * torch.square(torch.norm((right_foot_pos+right_foot_pos)/2. - self.root_states[:, :2], dim=-1)) * (self.time_to_stand_still > self.static_delay))


        foot_pos_left = quat_rotate_inverse(self.base_quat, self._rigid_body_pos[:, self.feet_indices[0], :] - self.base_pos)
        foot_pos_right = quat_rotate_inverse(self.base_quat, self._rigid_body_pos[:, self.feet_indices[1], :] - self.base_pos)
        d_min_left = torch.clamp(foot_pos_left[:,1] - ypos + 0.02, -0.5, 0.)
        d_min_right = torch.clamp(-foot_pos_right[:,1] - ypos + 0.02, -0.5, 0.)
        n_out += torch.abs(d_min_left) + torch.abs(d_min_right)
        # rew += (-torch.abs(d_min_left) * 20. + -torch.abs(d_min_right) * 20.)
        # print(torch.exp(rew))
        move = (torch.norm(self.commands[:, :3], dim=1) > 0.) 
        contacts = self.contact_forces[:, self.feet_indices, 2] > 5.
        double_contact = torch.sum(1.*contacts, dim=1)==2
        self.double_contact_last = torch.where(double_contact, self.double_contact_last + 1.,torch.zeros_like(self.double_contact_last))
        # move_stand = torch.norm(foot_pos_left[:,:2] + foot_pos_right[:,:2], dim = 1) * low_speed * double_contact > 0.18
        move_stand = (self.double_contact_last > 25) * move
        rew += (1. - self.cfg.rewards.smooth_w) * torch.where(move_stand, 
                           -0.1 * torch.ones((self.num_envs,),device=self.device),
                            torch.zeros((self.num_envs,),device=self.device))
        return rew -0.1 * (n_out > 0.)

    
    def _reward_foot_slip(self):
        """
        penalize foot slip, including x,y linear velocity and yaw angular velocity, when contacting ground
        """  
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self._rigid_body_vel[:, self.feet_indices, :2], dim=2) * contact
        foot_ang_vel = torch.norm(self._rigid_body_ang[:, self.feet_indices, 2].unsqueeze(1), dim=2) * contact
        rew = torch.exp(-torch.sum(torch.square(foot_speed_norm), dim=1)) \
                + torch.exp(-torch.sum(torch.square(foot_ang_vel), dim=1))

        return rew * self.smooth_w
    
    def _reward_foot_smooth(self):
        # foot_speed = self._rigid_body_vel[:, self.feet_indices, :3]
        # ratio = self.contact_forces[:, self.feet_indices, :]
        # ratio[:,:,:2] = 0.
        # ratio[:,:,2] = 1. * (ratio[:,:,2]/50.>1.)
        pre_foot_speed = self._rigid_body_vel0[:, self.feet_indices, :3]
        foot_speed1 = quat_rotate_inverse(self.base_quat, self.foot_speed[:,0,:])/(1. + torch.abs(self.base_lin_vel[:, :3]))
        foot_speed2 = quat_rotate_inverse(self.base_quat, self.foot_speed[:,1,:])/(1. + torch.abs(self.base_lin_vel[:, :3]))

        foot_acc1 = quat_rotate_inverse(self.base_quat, (self.foot_speed[:,0,:] - pre_foot_speed[:,0,:]))/(1. + torch.abs(self.base_lin_vel[:, :3]))
        foot_acc2 = quat_rotate_inverse(self.base_quat, (self.foot_speed[:,1,:] - pre_foot_speed[:,1,:]))/(1. + torch.abs(self.base_lin_vel[:, :3]))

        
        foot_reward = torch.sum(
                                # torch.square(foot_speed1)
                                # + torch.square(foot_speed2)
                                torch.square(foot_acc1)
                                + torch.square(foot_acc2), dim= -1)
        #             torch.sum(torch.abs(foot_acc2/(1. + torch.abs(self.base_lin_vel[:, :3])) * ratio[:,1]), dim= -1)

        # foot_reward = torch.sum(
        #     torch.square(torch.norm(self.foot_speed[:,:,:],dim=-1))
        #     + 4. * torch.square(torch.norm(self.foot_speed[:,:,:] - pre_foot_speed[:,:,:],dim=-1))
        #     , dim= -1) # 
        # foot_acc2 = (self.foot_speed[:,1,:] - pre_foot_speed[:,1,:])
        # foot_acc = torch.sum(torch.abs(foot_acc1/(1. + torch.abs(self.base_lin_vel[:, :3])) * ratio[:,0]), dim= -1) + \
        #             torch.sum(torch.abs(foot_acc2/(1. + torch.abs(self.base_lin_vel[:, :3])) * ratio[:,1]), dim= -1)
        # /(1. + torch.abs(self.commands_body.unsqueeze(1)))
        # if self.foot_height[0,0]<0.1:
        # self.max_acc = torch.where(self.max_acc>foot_acc,self.max_acc,foot_acc)
        # print("feet_air_time:", self.feet_air_time[0,]>0)
        print("height:", self.foot_height[0,0])
        print("speed:" ,self.foot_speed[0,0])
        print("force: ", self.contact_forces[0, self.feet_indices[0]], "\n")

        # print(ratio[0,0])
        # print(foot_reward)
        return foot_reward
    
    def _reward_foot_impulse(self):

        return self.cfg.rewards.impulse_w * torch.sum(
                    torch.sum(
                        torch.abs(
                            ((self.contact_forces[:, self.feet_indices, :]).clip(min=-1600.,max = 1600)) 
                            * ((torch.square(self.foot_speed) ).clip(min = 0)) #- 0.03
                        ),
                    dim=-1)
                ,dim=-1)

    
    def _reward_foot_height(self):
        #### change for terrain #####################
        # foot_speed = self._rigid_body_vel[:, self.feet_indices, :3]
        not_contact = self.contact_forces[:, self.feet_indices, 2] < 5.
        self.max_feet_height = torch.where(self.foot_height > self.max_feet_height,self.foot_height,self.max_feet_height)
        self.max_feet_height *= not_contact
        # print(self.max_feet_height[0])
        # foot_height_rew = torch.sum(torch.exp(-10. * torch.sqrt(torch.norm(self.foot_speed[:,:,:2],dim=-1))
        #                                               * torch.square(0.3 - foot_height) * not_contact),dim=-1)
        # print(torch.sum(torch.sqrt(torch.norm(self.foot_speed[:,:,:2],dim=-1)/(torch.norm(self.base_lin_vel[:,:2],dim=-1)+1e-8)
        #                                         + torch.abs(self._rigid_body_ang[:, self.feet_indices, 2])/torch.abs(self.base_ang_vel[:,2]+1e-8))
        #                                               * torch.square(0.2 - self.foot_height) * not_contact, dim=-1))
        # print("no /:", torch.sum(torch.sqrt(torch.norm(self.foot_speed[:,:,:2],dim=-1) + torch.abs(self._rigid_body_ang[:, self.feet_indices, 2]))))

        # 这个奖励为啥很奇怪
        foot_height_rew = torch.sum((torch.norm(self.foot_speed[:,:,:2],dim=-1)
                                                + torch.abs(self._rigid_body_ang[:, self.feet_indices, 2]))
                                                      * torch.square((0.1 - self.foot_height).clip(min=0.)), dim=-1) #

        foot_height_rew2 = torch.sum(((torch.abs(self.foot_speed[:,:,2]) - 0.3).clip(min = 0.))
                                                      * torch.exp(-1000. * torch.square(self.foot_height)), dim=-1)
                                                # + torch.abs(self._rigid_body_ang[:, self.feet_indices, 2])
        pre_foot_speed = self._rigid_body_vel0[:, self.feet_indices, :3]

        foot_height_rew3 = torch.sum(torch.where(self.foot_height < self.max_feet_height/2., 
                            (self.foot_speed[:,:,2] - self._rigid_body_vel0[:, self.feet_indices, 2]) * (self.foot_speed[:,:,2] < 0.) * not_contact,
                            torch.zeros_like(self.foot_speed[:,:,2])),dim=1)
        # if self.foot_height[0,0] < 0.17:
        #     print("***************************************")
        #     print("height: ", self.foot_height[0,0].item())
        #     print(" speed: ", (torch.norm(self.foot_speed[0,0,:2],dim=-1)
        #                                         + torch.abs(self._rigid_body_ang[0, self.feet_indices[0], 2])).item())
        #     print("   rew: ", torch.exp(-100.*torch.sum((torch.norm(self.foot_speed[0,0,:2],dim=-1)
        #                                         + torch.abs(self._rigid_body_ang[0, self.feet_indices[0], 2]))
        #                                               * torch.square((0.1 - self.foot_height[0,0]).clip(min=0.)), dim=-1)).item(), "\n")
        return torch.exp(-100.*foot_height_rew) * self.smooth_w# + self.cfg.rewards.smooth_w * 5. * foot_height_rew3#- 6. * foot_height_rew2

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             2 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_action_rate_exp(self):
        # Penalize changes in actions
        nact = self.num_actions
        rew = torch.exp(-1. * torch.sum(torch.square(self.actions*self.cfg.control.action_scale \
                            - self.ctrl_hist[:, -self.num_actions:] \
                        ), dim=1))
        return rew * (1.+(self.time_to_stand_still>self.static_delay)*0.)
    
    def _reward_action_rate2_exp(self):
        # Penalize changes in actions
        nact = self.num_actions
        rew = torch.exp(-0.5 * torch.sum(torch.square(self.actions*self.cfg.control.action_scale \
                        - 2.*self.ctrl_hist[:, -self.num_actions:]  \
                        + self.ctrl_hist[:, -2*self.num_actions:-self.num_actions]  \
                        ), dim=1))
        return rew * (1.+(self.time_to_stand_still>self.static_delay)*0.)
    
    def _reward_dof_acc_exp(self):
        # Penalize dof accelerations
        rew = torch.exp(-0.002 * torch.sum(torch.square((self.dof_vel_hist[:, -self.num_dof:] - self.dof_vel)), dim=1) )
        return rew * (1.+(self.time_to_stand_still>self.static_delay)*0.)   

    def _reward_dof_vel_exp(self):
        # Reward zero dof velocities
        rew = torch.exp(-0.00048 * torch.sum(torch.square(self.dof_vel), dim=1))
        # print(self.time_to_stand_still[0])
        return rew * (1.+(self.time_to_stand_still>self.static_delay)*0.)
    # * Potential-based rewards * #

    def pre_physics_step(self):
        self.rwd_oriPrev = self._reward_orientation()
        self.rwd_baseHeightPrev = self._reward_base_height()
        self.rwd_jointRegPrev = self._reward_joint_regularization()
        self.rwd_standStillPrev = self._reward_stand_still()
        self.rwd_ankleRegPrev = self._reward_ankle_regularization()
        self.lin_vel1[:] = self.base_lin_vel
        self.ang_vel1[:] = self.base_ang_vel
        self.contact_force0[:] = self.contact_force1[:]
        self.contact_force1[:] = self.contact_forces[:, self.feet_indices, 2]
        self._rigid_body_vel0[:] = self._rigid_body_vel[:]
    

    def _reward_ori_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_orientation() - self.rwd_oriPrev)
        return delta_phi / self.dt_step

    def _reward_jointReg_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_joint_regularization() - self.rwd_jointRegPrev)
        return delta_phi / self.dt_step

    def _reward_baseHeight_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_base_height() - self.rwd_baseHeightPrev)
        return delta_phi / self.dt_step
    
    def _reward_ankleReg_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_ankle_regularization() - self.rwd_ankleRegPrev)
        return delta_phi / self.dt_step
        
# ##################### HELPER FUNCTIONS ################################## #

    def sqrdexp(self, x):
        """ shorthand helper for squared exponential
        """
        return torch.exp(-torch.square(x)/self.cfg.rewards.tracking_sigma)

    def smooth_sqr_wave(self, phase):
        p = 2.*torch.pi*phase * self.phase_freq
        return torch.sin(p) / \
            (2*torch.sqrt(torch.sin(p)**2. + self.eps**2.)) + 1./2.
    

    
    def analyze_max_vel(self):
        x = torch.max(self.base_lin_vel[:, 0]).unsqueeze(0)
        y = torch.max(self.base_lin_vel[:, 1]).unsqueeze(0)
        z = torch.max(self.base_lin_vel[:, 2]).unsqueeze(0)
        roll = torch.max(self.base_ang_vel[:, 0]).unsqueeze(0)
        pitch = torch.max(self.base_ang_vel[:, 1]).unsqueeze(0)
        yaw = torch.max(self.base_ang_vel[:, 2]).unsqueeze(0)
        cur_max = torch.cat((x,y,z,roll,pitch,yaw))
        true_max = torch.where(self.max_vel > cur_max, self.max_vel, cur_max)
        self.max_vel[:] = true_max
        print(true_max)

    def print_vel(self):
        print("linear:")
        print(self.base_lin_vel)
        print("angular:")
        print(self.base_ang_vel)

    def print_feet_force(self):
        print(self.contact_forces[:, self.feet_indices, 2])

    def move_same(self):
        ##### env_ids need to change
        # env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # print(env_ids.shape)


        # self.root_states[env_ids, :] = self.root_states[0, :] #### 1 to end
        # self.dof_pos[env_ids, :] = self.dof_pos[0]
        # self.dof_vel[env_ids, :] = self.dof_vel[0]
        # if (len(env_ids) > 0):
        for i1 in range(self.dof_pos.shape[0]):
            self.dof_pos[i1, :] = self.dof_pos[0, :]
        for i1 in range(self.dof_vel.shape[0]):
            self.dof_vel[i1, :] = self.dof_vel[0, :]
        for i1 in range(self.root_states.shape[0]):
            self.root_states[i1, 3:] = self.root_states[0, 3:]
        env_ids_int32 = to_torch(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
             16, 17, 18, 19, 20, 21, 22, 23, 24,
             25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
             36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
             # 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
             # 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
             ],
            device=self.device, dtype=torch.int32)
        # env_ids_int32 = env_ids.to(dtype=torch.int32)
        # print('force move same')
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        

    def print_rs(self):
        r_s = 0.
        nact = self.num_actions
        r_s -= 4000. * torch.square(self.actions*self.cfg.control.action_scale \
                            - self.ctrl_hist[:, :nact] \
                        )
        r_s -= 2000. * torch.square(self.actions*self.cfg.control.action_scale \
                        - 2.*self.ctrl_hist[:, :nact]  \
                        + self.ctrl_hist[:, nact:2*nact]  \
                        )
        r_s -= torch.square(self.dof_vel)
        r_s -= 4. * torch.square((self.dof_vel_hist[:, :self.num_dof] - self.dof_vel))
        return torch.exp(0.00012*torch.sum(r_s, dim=-1))
    

    def draw_sphere(self, location, color):
        self.gym.clear_lines(self.viewer)
        sphere_geom = gymutil.WireframeSphereGeometry(0.03, 8, 8, None, color)
        pose = gymapi.Transform(gymapi.Vec3(location[0, 0], location[0, 1], location[0, 2]), r=None)
        gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[0], pose)