"""
Configuration file for "fixed arm" (FA) humanoid environment
with potential-based rewards implemented
"""

import glob
import torch
from gpugym.envs.base.legged_robot_config \
    import LeggedRobotCfg, LeggedRobotCfgPPO
	
MOTION_FILES = glob.glob('/home/usr/project/RL_GROUP-main/pbrs_rl/datasets/mocap_motions/*')

use_tcn = True
get_heights = True

class PBRS_OGHR_V4_Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env): 
            
        amp_motion_files = MOTION_FILES
        reference_state_initialization = False
        reference_state_initialization_prob = 0.85		
        include_history_steps = None  # Number of steps of history to include.
		

        num_envs = 4096
        num_observations = 173
        num_actions = 12
        episode_length_s = 70
        num_privileged_obs = 379
        num_history_short = 3
        num_history_long = 100 if use_tcn else 0
        size_history_long = 42 # = sum(dimensions of all short history) = 12+12+12+3+3
        num_observations = num_observations + ((num_history_long*size_history_long) if use_tcn else 0)
        num_privileged_obs += ((187) if get_heights else 0)

    class terrain(LeggedRobotCfg.terrain):
        curriculum = True
        mesh_type = 'plane' # 'trimesh'  #
        measure_heights = ((True) if get_heights else 0)
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # terrain_proportions = [0.5, 0.5]
        static_friction = 0.1
        dynamic_friction = 0.1
        has_low_field = False

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4
        resampling_time = 5.
        robot_height_command = True
        ang_vel_command = True
        always_walking = False
        lin_vel_clip = 0.3
        lin_vel_x_clip = 0.2
        lin_vel_y_clip = 0.06
        ang_vel_yaw_clip = 0.3
        static_delay = 0.
        resampling_range = [3., 15.] # for random resample time

        class ranges:
            # TRAINING COMMAND RANGES #
            lin_vel_x = [-1.2, 1.2]        # min max [m/s]
            lin_vel_y = [-0.6, 0.6]   # min max [m/s]
            ang_vel_yaw = [-1., 1.]     # min max [rad/s]
            robot_height = [-1., 1.]     # min max [scale]

            # PLAY COMMAND RANGES #
            # lin_vel_x = [3., 3.]    # min max [m/s]
            # lin_vel_y = [-0., 0.]     # min max [m/s]
            # ang_vel_yaw = [2, 2]      # min max [rad/s]
            # heading = [0, 0]

    class init_state(LeggedRobotCfg.init_state):
        reset_mode = 'mixed_reset'
        range_pb = 0.5
        penetration_check = False
        pos = [0., 0., 1.122]        # x,y,z [m]  
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]   # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]   # x,y,z [rad/s]

        # ranges for [x, y, z, roll, pitch, yaw]
        root_pos_range = [
            [0., 0.],
            [0., 0.],
            [1.122, 1.122],      
            [-torch.pi/10, torch.pi/10],
            [-torch.pi/10, torch.pi/10],
            [-torch.pi/10, torch.pi/10]
        ]

        # ranges for [v_x, v_y, v_z, w_x, w_y, w_z]
        root_vel_range = [
            [-.5, .5],
            [-.5, .5],
            [-.5, .5],
            [-.5, .5],
            [-.5, .5],
            [-.5, .5]
        ]

        default_joint_angles = {
            'Joint-hip-r-roll': 0.,
            'Joint-hip-r-yaw': 0.,
            'Joint-hip-r-pitch': 0.305913,
            'Joint-knee-r-pitch': -0.670418,
            'Joint-ankel-r-pitch': 0.371265,
            'Joint-ankel-r-roll': 0.,

            'Joint-hip-l-roll': 0.,
            'Joint-hip-l-yaw': 0.,
            'Joint-hip-l-pitch': 0.305913,
            'Joint-knee-l-pitch': -0.670418,
            'Joint-ankel-l-pitch': 0.371265,
            'Joint-ankel-l-roll': 0.,
        }

        dof_pos_range = {
            'Joint-hip-r-roll': [-0.2, 0.2],
            'Joint-hip-r-yaw': [-0.1, 0.1],
            'Joint-hip-r-pitch': [0.12, 0.52],
            'Joint-knee-r-pitch': [-0.72, -0.62],
            'Joint-ankel-r-pitch': [0.07, 0.67],
            'Joint-ankel-r-roll': [-0.1, 0.1],

            'Joint-hip-l-roll': [-0.2, 0.2],
            'Joint-hip-l-yaw': [-0.1, 0.1],
            'Joint-hip-l-pitch': [0.12, 0.52],
            'Joint-knee-l-pitch': [-0.72, -0.62],
            'Joint-ankel-l-pitch': [0.07, 0.67],
            'Joint-ankel-l-roll': [-0.1, 0.1],
        }

        dof_vel_range = {
            'Joint-hip-r-roll':[-0.1, 0.1],
            'Joint-hip-r-yaw': [-0.1, 0.1],
            'Joint-hip-r-pitch': [-0.1, 0.1],
            'Joint-knee-r-pitch': [-0.1, 0.1],
            'Joint-ankel-r-pitch': [-0.1, 0.1],
            'Joint-ankel-r-roll': [-0.1, 0.1],

            'Joint-hip-l-roll': [-0.1, 0.1],
            'Joint-hip-l-yaw': [-0.1, 0.1],
            'Joint-hip-l-pitch': [-0.1, 0.1],
            'Joint-knee-l-pitch': [-0.1, 0.1],
            'Joint-ankel-l-pitch': [-0.1, 0.1],
            'Joint-ankel-l-roll': [-0.1, 0.1],
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P' # P: position, V: velocity, T: torques
        # stiffness and damping for joints
        stiffness = {
            'Joint-hip-r-roll': 400.,
            'Joint-hip-r-yaw': 200.,
            'Joint-hip-r-pitch': 400.,
            'Joint-knee-r-pitch': 400.,
            'Joint-ankel-r-pitch': 192.,
            'Joint-ankel-r-roll': 192.,

            'Joint-hip-l-roll': 400.,
            'Joint-hip-l-yaw': 200.,
            'Joint-hip-l-pitch': 400.,
            'Joint-knee-l-pitch': 400.,
            'Joint-ankel-l-pitch': 192.,
            'Joint-ankel-l-roll': 192.,

            #old p
            # 'Joint-hip-r-roll': 300.,
            # 'Joint-hip-r-yaw': 200.,
            # 'Joint-hip-r-pitch': 200.,
            # 'Joint-knee-r-pitch': 400.,
            # 'Joint-ankel-r-pitch': 120.,
            # 'Joint-ankel-r-roll': 120.,

            # 'Joint-hip-l-roll': 300.,
            # 'Joint-hip-l-yaw': 200.,
            # 'Joint-hip-l-pitch': 200.,
            # 'Joint-knee-l-pitch': 400.,
            # 'Joint-ankel-l-pitch': 120.,
            # 'Joint-ankel-l-roll': 120.,
        }
        damping = {
            'Joint-hip-r-roll': 2.,
            'Joint-hip-r-yaw': 2.,
            'Joint-hip-r-pitch': 2.,
            'Joint-knee-r-pitch': 4.,
            'Joint-ankel-r-pitch': 0.64,
            'Joint-ankel-r-roll': 0.64,

            'Joint-hip-l-roll': 2.,
            'Joint-hip-l-yaw': 2.,
            'Joint-hip-l-pitch': 2.,
            'Joint-knee-l-pitch': 4.,
            'Joint-ankel-l-pitch': 0.64,
            'Joint-ankel-l-roll': 0.64,
        }

        action_scale = 0.4 #1.0
        exp_avg_decay = 0.05
        decimation = 20

    class domain_rand(LeggedRobotCfg.domain_rand):


        randomize_friction = False
        # friction_range = [0.5, 1.25]
        friction_range = [0.3, 3.]

        #load mass randomize
        randomize_load_mass = False
        added_mass_range = [-10., 10.]

        randomize_all_mass = False
        rd_mass_range = [0.5, 1.5]

        randomize_com = False
        rd_com_range = [-0.05, 0.05]

        randomize_base_com = False
        rd_base_com_range = [-0.1, 0.1]

        push_robots = False
        push_interval_s = 2
        push_ratio= 0.4
        max_push_vel_xy = 0.5
        max_push_ang_vel = 0.4

        random_pd = False
        p_range = [0.7, 1.3]
        d_range = [0.7, 1.3]

        random_damping = False
        damping_range = [0.3, 4.0]

        random_inertia = False
        inertia_range = [0.7, 1.3]

        dynamic_randomization = 0.02 ############################### first our delay second another dist and entropy #################
        comm_delay = False
        comm_delay_range = [0, 26] # will exclude the upper limit

        random_dof_friction = False
        dof_friction_range = [0., 5.]

        randomize_joint_armature = False
        armature_range = [0.9, 1.1]

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}'\
            '/resources/robots/OGHR/urdf/OGHR_wholeBody_Simplified(rotor12dof).urdf'
        keypoints = ["base_link"]
        end_effectors = ['Link-ankel-r-roll', 'Link-ankel-l-roll']
        foot_name = ['Link-ankel-l-roll', 'Link-ankel-r-roll']
        knee_name = ['Link-knee-l-pitch', 'Link-knee-r-pitch']

        terminate_after_contacts_on = [
            "base_link",
        ]

        disable_gravity = False
        disable_actions = False
        disable_motors = False

        # (1: disable, 0: enable...bitwise filter)
        self_collisions = 0
        collapse_fixed_joints = False
        flip_visual_attachments = False

        # Check GymDofDriveModeFlags
        # (0: none, 1: pos tgt, 2: vel target, 3: effort)
        default_dof_drive_mode = 3

    class rewards(LeggedRobotCfg.rewards):
        # ! "Incorrect" specification of height
        # base_height_target = 0.7
        base_height_target = 1.05
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.8
        max_contact_force = 1500.

        smooth_w = 0.001   #1


        impulse_w = 1.

        # negative total rewards clipped at zero (avoids early termination)
        only_positive_rewards = False
        tracking_sigma = 0.5
        accumulate_target = False

        class scales(LeggedRobotCfg.rewards.scales):
            # * "True" rewards * #
            # reward for task
            # target_position = 4.
            tracking_lin_vel = 10.
            # target_orientation = 2.
            tracking_ang_vel = 5.
            orientation = 5. 
            tracking_avg = 10.
            # waiting = -1
            lin_vel_z = 2.5
            ang_vel_xy = 2.5
            ###################################
            #reward for trajectory
            
            

            # foot regularization
            foot_height = 6.
            feet_air_time = 1.5
            feet_contact = 5.0 #1.0
            foot_slip = 1.0
            # foot_force = 1.0
            # foot_pos = 2.

            # reward for smooth
            torques = 2.
            torques_var = 0.5
            base_acc = 5.
            foot_impulse = -0.02
            dofpower = -20.
            # foot_acc = -0.1 #5.
            ##############################
            # action_rate_exp = 2.
            # action_rate2_exp = 2.
            # dof_acc_exp = 2.
            # dof_vel_exp = 2.
            # dof_acc = -2.5e-6
            # dof_vel = -0.01
            # action_rate = -0.1
            # action_rate2 = -0.1
            

            

            #reward for safety
            # dof_pos_limits = -20
            # torque_limits = -2e-2


            # feet_contact_forces = -1e-2
            # termination = -200

            #--------------------


            ################################
            # stumble = -2
            
            #reward for beauty
            # * Shaping rewards * #
            # Sweep values: [0.5, 2.5, 10, 25., 50.]
            # Default: 5.0
            # orientation = 5.0

            # Sweep values: [0.2, 1.0, 4.0, 10., 20.]
            # Default: 2.0
            # base_height = 2.0

            # Sweep values: [0.1, 0.5, 2.0, 5.0, 10.]
            # Default: 1.0
            # joint_regularization = 1.0

            # * PBRS rewards * #
            # Sweep values: [0.1, 0.5, 2.0, 5.0, 10.]
            # Default: 1.0
            # ori_pb = 2.0

            # Sweep values: [0.1, 0.5, 2.0, 5.0, 10.]
            # Default: 1.0

            # baseHeight_pb = 2.
            # base_height_stand = 2.



            # Sweep values: [0.1, 0.5, 2.0, 5.0, 10.]
            # Default: 1.0

            # jointReg_pb = 2.0
            # ankleReg_pb = 2.0



            # dofpower_pb = 2.0
            # footimpulse_pb = 2.0



    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            base_z = 1./0.6565 

        clip_observations = 100.
        clip_actions = 10.

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.003 # 0.005
            dof_vel = 0.16 # 0.01
            ang_vel = 0.1
            gravity = 0.016 # 0.05
            base_z = 0.05
            lin_vel = 0.016 # 0.1
            in_contact = 0.1
            height_measurements = 0.1

    class sim(LeggedRobotCfg.sim):
        dt = 0.001
        substeps = 1
        gravity = [0., 0., -9.81]
        class physx:
            max_depenetration_velocity = 10.0


class PBRS_OGHR_V4_CfgPPO(LeggedRobotCfgPPO):
    do_wandb = True
    seed = -1

    class algorithm(LeggedRobotCfgPPO.algorithm):
        # algorithm training hyperparameters
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        amp_replay_buffer_size = 1000000
        num_learning_epochs = 5
        num_mini_batches = 4    # minibatch size = num_envs*nsteps/nminibatches
        learning_rate = 1.e-5
        schedule = 'adaptive'   # could be adaptive, fixed
        gamma = 0.98
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        weight_decay = 0

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCriticTCN' if use_tcn else 'ActorCritic'
        algorithm_class_name = 'AMPPPO'
        num_steps_per_env = 24
        max_iterations = 30000
        run_name = 'OGHR_TCN_2024' if use_tcn else 'OGHR_V4_2024'
        experiment_name = 'PBRS_OGHR_Locomotion_TCN' if use_tcn else 'PBRS_OGHR_Locomotion_V4'
        # run_name = 'OGHR_TCN_2024_highsp'
        # experiment_name = "PBRS_OGHR_Locomotion_TCN_highsp"
        save_interval = 50
        plot_input_gradients = False
        plot_parameter_gradients = False


        
        max_iterations = 500000 # number of policy updates
        amp_reward_coef = 2.0
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.3
        amp_discr_hidden_dims = [1024, 512]

        min_normalized_std = [0.05, 
                              0.05, 
                              0.05,0.05, 0.05,0.02, 0.05, 
                              0.05,0.05, 0.05,0.02, 0.05 ] 		
		
		
		
		
		

    class policy(LeggedRobotCfgPPO.policy):
        actor_hidden_dims = [256, 256, 256]
        critic_hidden_dims = [256, 256, 256]
        # (elu, relu, selu, crelu, lrelu, tanh, sigmoid)
        activation = 'elu'
        TCN_activation = 'relu'
        conv_dims = [(42, 32, 6, 5), (32, 16, 4, 2)]
        period_length = 100
        
        
        
        
        
        
"""
Configuration file for cassie
"""




