"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Franka Cube Pick
----------------
Use Jacobian matrix and inverse kinematics control of Franka robot to pick up a box.
Damped Least Squares method from: https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
"""

import os
import shutil
import cv2
import csv
import h5py
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch
import random
import time
import copy

from torch.utils.tensorboard import SummaryWriter



class Gym_env():
    def __init__(self):
        
        self.num_envs           = 1

        self.seed               = 8
        self.sim_device         = "cpu"
        self.env                = None
        self.gym                = None     #self.gym = gymapi.acquire_gym()
        self.sim                = None     #self.sim = self.gym.create_sim(args['compute_device_id'], args['graphics_device_id'], args['physics_engine'], sim_params)
        
        self.viewer             = None
        self.envs               = []
        
        self.piper_dof_states   = []
        self.piper_body_states  = []
        self.piper_handles      = []
        self.piper_velocity_target = []
        self.piper_hand         = "link6"
        self.saved_dof_states   = None
        
        self.cube_states        = []
        self.cube_handles       = []
        
        self.states_bf          = []
        
        self.cube_pose          = [0.3, 0.2, 0.0]        # Target position
        self.goal_rot           = [0.0, 0.0, 0.0, -1.0]   # Target orientation (quaternion)

        self.sphere_geom        = None

        self.asset_root  =  "/home/ucluser/isaacgym/assets"
        # self.asset_root  =  "/home/wee_ucl/workspace/Piper_RL/assets/"
        # self.asset_root  = "/home/ucluser/isaacgym/assets"
        self.success_dataset_dir= None
        self.failure_dataset_dir= None
        self.piper_lower_limits = []
        self.piper_upper_limits = []
        self.piper_mids         = []
        self.piper_num_dofs     = None
        
        self.envs               = []
        self.tray_handles       = []
        self.piper_handles      = []
        self.box_handles        = []
        self.camera_handles     = []
        self.tray_idxs          = []
        self.box_idxs           = []
        self.unfinished_box_idxs= []
        self.hand_idxs          = []
        self.init_pos_list      = []
        self.init_rot_list      = []
        self.writers            = []
        self.dof_states         = None
        self.init_dof_states    = None
        self.rb_states          = None
        self.camera_props       = gymapi.CameraProperties()
        
        
        
        self.time_counter       = 0
        self.time_ep            = 0
        self.goal_dist_initial  = 0
        
        self.table_dims         = gymapi.Vec3(0.6, 1.0, 0.01)
        self.table_pose         = gymapi.Transform()
        self.cube_size          = 0.04
        self.tray_pose          = gymapi.Transform()
        self.tray_color         = None
        self.num_box            = 1
        
        torch.set_printoptions(precision=4, sci_mode=False)

        # GPU configuration
        if self.sim_device == 'cuda':
            print("CUDA IS AVAILABLE")
            GPU_num = torch.cuda.current_device()
            self.device = torch.device("cuda:{}".format(GPU_num))
        else:
            self.device = "cpu"
        
        for j in range(self.num_envs):
            color_writer_1, color_writer_2 = self.create_video_writer(j)
            self.writers.append([color_writer_1, color_writer_2])
    
    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
    
    def parse_device_str(self, device_str):
        # defaults
        device = 'cpu'
        device_id = 0

        if device_str == 'cpu' or device_str == 'cuda':
            device = device_str
            device_id = 0
        else:
            device_args = device_str.split(':')
            assert len(device_args) == 2 and device_args[0] == 'cuda', f'Invalid device string "{device_str}"'
            device, device_id_s = device_args
            try:
                device_id = int(device_id_s)
            except ValueError:
                raise ValueError(f'Invalid device string "{device_str}". Cannot parse "{device_id}"" as a valid device id')
        return device, device_id
        
    def init_gym(self):
        """
            Create a parametrized empty gym environment 
        """
            
        # Initialize gym
        self.gym = gymapi.acquire_gym()

        args = dict(
            compute_device_id=0, 
            flex=False, 
            graphics_device_id=0, 
            num_envs=1, 
            num_threads=0, 
            physics_engine= gymapi.SIM_PHYSX, 
            physx=False, 
            pipeline='gpu', 
            sim_device='cpu', 
            sim_device_type='cuda', 
            slices=0, 
            subscenes=0, 
            use_gpu=False, 
            use_gpu_pipeline=False
        )
        
        # configure sim
        sim_params                  = gymapi.SimParams()
        sim_params.up_axis          = gymapi.UP_AXIS_Z
        sim_params.gravity          = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.dt               = 1.0 / 60.0
        sim_params.substeps         = 2
        sim_params.use_gpu_pipeline = args['use_gpu_pipeline']
        if args['physics_engine'] == gymapi.SIM_PHYSX:
            sim_params.physx.solver_type                    = 1
            sim_params.physx.num_position_iterations        = 8
            sim_params.physx.num_velocity_iterations        = 1
            sim_params.physx.rest_offset                    = 0.0
            sim_params.physx.contact_offset                 = 0.001
            sim_params.physx.friction_offset_threshold      = 0.001
            sim_params.physx.friction_correlation_distance  = 0.0005
            sim_params.physx.num_threads                    = args['num_threads']
            sim_params.physx.use_gpu                        = args['use_gpu']
        else:
            raise Exception("This example can only be used with PhysX")


        #TODO: Consider where sim should be initialized
        self.sim = self.gym.create_sim(args['compute_device_id'], args['graphics_device_id'], args['physics_engine'], sim_params)

        if self.sim is None:
            raise Exception("Failed to create sim")


        # Create viewer

        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise Exception("Failed to create viewer")

        # Add ground plane
        plane_params        = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        
        # return gym, sim, viewer

    def load_piper(self):
        """
            Create a piper asset
        """
        piper_asset_file = "urdf/piper_description/urdf/piper_description.urdf"
        # piper_description_chain = Chain.from_urdf_file(self.asset_root + "/" + piper_asset_file)
        asset_options                           = gymapi.AssetOptions()
        asset_options.fix_base_link             = True
        asset_options.flip_visual_attachments   = True
        asset_options.armature                  = 0.01
        piper_asset                             = self.gym.load_asset(self.sim, self.asset_root, piper_asset_file, asset_options)
        print("Loading asset '%s' from '%s'" % (piper_asset_file, self.asset_root))
        
        return piper_asset  #, piper_description_chain

    def load_table(self):
        """
            Create a table asset
        """
        table_asset_options                 = gymapi.AssetOptions()
        table_asset_options.fix_base_link   = True
        table_asset_options.armature        = 0.01
        table_asset                         = self.gym.create_box(self.sim, self.table_dims.x, self.table_dims.y, self.table_dims.z, table_asset_options)
        self.table_pose.p                   = gymapi.Vec3(0.5, 0.0, 0.5 * self.table_dims.z)
        print("Creating asset '%s' " % ("table"))
        
        return table_asset
    
    def load_cube(self):
        """
            Create a cube asset 
        """
        cube_size                    = self.cube_size
        cube_dim                     = [cube_size] * 3
        cube_asset_options           = gymapi.AssetOptions()
        cube_asset_options.density   = 1000
        cube_asset_options.armature  = 0.01
        cube_asset                   = self.gym.create_box(self.sim, cube_size, cube_size, cube_size, cube_asset_options)
        cube_pose                    = gymapi.Transform()
        cube_pose_np                 = np.array((0,0,0))
        print("Creating asset '%s' " % ("cube1"))
        
        return cube_asset

    def load_tray(self):
        """
            Create a tray asset
        """
        tray_dim                            = [0.15, 0.15, 0.005] #small
        # tray_dim = [0.2, 0.2, 0.01] #original
        self.tray_color                     = gymapi.Vec3(0.24, 0.35, 0.8)
        tray_asset_file                     = "urdf/tray/traybox_smaller.urdf"
        tray_asset_options                  = gymapi.AssetOptions()
        tray_asset_options.armature         = 0.01
        tray_asset_options.density          = 8000
        tray_asset_options.override_inertia = True
        tray_asset                          = self.gym.load_asset(self.sim, self.asset_root, tray_asset_file, tray_asset_options)
        x               = 0.3 #corner.x  # + table_dims.x * 0.2
        y               = 0.1 #corner.y + table_dims.y * 0.8
        z               = tray_dim[2]
        
        self.tray_pose.p     = gymapi.Vec3(x, y, z)
        print("Loading asset '%s' from '%s'" % (tray_asset_file, self.asset_root))
        
        return tray_asset
    
    def create_piper_env(self):
        """Create a simulation environment for PiPER 

        Parameters
        ----------
        args : dict
            simulation-related parameters, consisting of the following keys:
            
            -- sim_device            Physics Device in PyTorch-like syntax
            -- pipeline              Tensor API pipeline (cpu/gpu)
            - graphics_device_id    Graphics Device ID
            - physics_engine        Use FleX or PhysX for physics
            - num_threads           Number of cores used by PhysX
            --subscenes             Number of PhysX subscenes to simulate in parallel
            --slices                Number of client threads that process env slices
            - num_envs              Number of environments to create
            - total_time            Time to reach the target pose
            - use_gpu
            - use_gpu_pipeline
            - compute_device_id
            - 

        Returns
        -------
        envs
            a list of all simulated environments
        sim
            a simulation instance with Piper robots with objects
        viewer
            viewer of the gym environment
        """
        
        # Time to wait in seconds before moving robot
        next_piper_update_time = 1.0
        # Set up the env grid
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        piper_pose = gymapi.Transform()
        piper_pose.p = gymapi.Vec3(0, 0.0, 0.0)
        piper_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

        cube_pose = gymapi.Transform()
        cube_pose.p = gymapi.Vec3(self.cube_pose[0], self.cube_pose[1], self.cube_pose[2])
        cube_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
        
        goal_pose = gymapi.Transform()
        goal_pose.p = gymapi.Vec3(self.cube_pose[0], self.cube_pose[1], self.cube_pose[2])
        goal_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
        
        # Create an wireframe sphere
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        self.sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(1, 0, 0))
        axes_geom = gymutil.AxesGeometry(0.1)

        #create empty gym environment  
        self.set_seed(self.seed)
        self.init_gym()
        piper_asset = self.load_piper()
        cube_asset  = self.load_cube()
        tray_asset  = self.load_tray()
        table_asset = self.load_table()

        piper_pose     = gymapi.Transform()
        piper_pose.p   = gymapi.Vec3(0, 0, 0)
        
        # get joint limits and ranges for piper
        piper_dof_props             = self.gym.get_asset_dof_properties(piper_asset)
        piper_lower_limits          = piper_dof_props['lower']
        piper_upper_limits          = piper_dof_props['upper']
        piper_ranges                = piper_upper_limits - piper_lower_limits
        piper_mids                  = 0.5 * (piper_upper_limits + piper_lower_limits)
        
        # default dof states and position targets
        self.piper_num_dofs         = self.gym.get_asset_dof_count(piper_asset)
        default_dof_pos             = np.zeros(self.piper_num_dofs, dtype=np.float32)
        default_dof_pos[:6]         = piper_mids[:6]
        
        # grippers open
        default_dof_pos[6:]         = piper_upper_limits[6:]
        default_dof_state           = np.zeros(self.piper_num_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"]    = default_dof_pos
        
        

        piper_dof_props["driveMode"][:6].fill(gymapi.DOF_MODE_POS)
        piper_dof_props["stiffness"][:6].fill(400.0)
        piper_dof_props["damping"][:6].fill(40.0)
        
        piper_dof_props["driveMode"][6:].fill(gymapi.DOF_MODE_POS)
        piper_dof_props["stiffness"][6:].fill(800.0)
        piper_dof_props["damping"][6:].fill(40.0)
        
        table_pose      = gymapi.Transform()
        table_pose.p    = gymapi.Vec3(0.35, 0.0, 0.5 * self.table_dims.z )

        box_pose        = gymapi.Transform()

        
        
        self.camera_props.width      = 640
        self.camera_props.height     = 480
        camera_1_position       = gymapi.Vec3(0.4, 0.5, 0.5)
        camera_1_target         = gymapi.Vec3(0, 0, 0)
        camera_2_position       = gymapi.Vec3(0.4, - 0.5, 0.6)
        camera_2_target         = gymapi.Vec3(0, 0, 0)


        # unfinished_color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
        # finished_color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
        
        print("Creating %d environments" % self.num_envs)
        num_per_row = int(math.sqrt(self.num_envs))

        def random_box_pose():
            box_pose.p.x = table_pose.p.x + np.random.uniform(-0.1, 0.1)
            box_pose.p.y = table_pose.p.y + np.random.uniform(-0.2, 0.2)
            box_pose.p.z = self.table_dims.z + 0.5 * self.cube_size
            # box_pose_np     = np.array([box_pose.p.x,box_pose.p.y,box_pose.p.z])
            # init_box_pose = np.zeros(1, dtype=gymapi.RigidBodyState.dtype)
            # init_box_pose['pose']['p'][0] =(box_pose.p.x, box_pose.p.y, box_pose.p.z)
            box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
            return box_pose

        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)

            # add table
            # table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)

            # add tray
            tray_handle     = self.gym.create_actor(env, tray_asset, self.tray_pose, "tray", i, 0)
            self.tray_handles.append(tray_handle)
            self.gym.set_rigid_body_color(env, self.tray_handles[i], 0, gymapi.MESH_VISUAL_AND_COLLISION, self.tray_color)
            # get global index of tray in rigid body state tensor
            tray_idx        = self.gym.get_actor_rigid_body_index(env, tray_handle, 0, gymapi.DOMAIN_SIM)
            self.tray_idxs.append(tray_idx)
            
            # add box
            self.box_handles.append([])
            self.box_idxs.append([])
            
            for n in range(self.num_box):
                box_handle          = self.gym.create_actor(env, cube_asset, random_box_pose(), "box_" + str(n), i, 0)
                unfinished_color    = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
                self.gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, unfinished_color) #color
                self.box_handles[i].append(box_handle)
                
                # get global index of box in rigid body state tensor
                box_idx = self.gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
                self.box_idxs[i].append(box_idx)
            
            

            # add piper
            piper_handle = self.gym.create_actor(env, piper_asset, piper_pose, "piper", i, 2)
            self.piper_handles.append(piper_handle)
            
            # set dof properties
            self.gym.set_actor_dof_properties(env, piper_handle, piper_dof_props)

            # set initial dof states
            self.gym.set_actor_dof_states(env, piper_handle, default_dof_state, gymapi.STATE_ALL)

            # set initial position targets
            self.gym.set_actor_dof_position_targets(env, piper_handle, default_dof_pos)

            # get inital hand pose
            hand_handle     = self.gym.find_actor_rigid_body_handle(env, piper_handle, "piper_hand")
            hand_pose       = self.gym.get_rigid_transform(env, hand_handle)
            self.init_pos_list.append([0.1, 0.1, 0.3])
            # init_pos_list.append([tray_pose.p.x, tray_pose.p.y, tray_pose.p.z])
            self.init_rot_list.append([-0.95, -0.25, 0.0, 0.0])
            # init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
            # init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])
            
            # get global index of hand in rigid body state tensor
            hand_idx        = self.gym.find_actor_rigid_body_index(env, piper_handle, "piper_hand", gymapi.DOMAIN_SIM)
            self.hand_idxs.append(hand_idx)
            
            # add camera
            cam_1 = self.gym.create_camera_sensor(env, self.camera_props)
            cam_2 = self.gym.create_camera_sensor(env, self.camera_props)
            #set the location of camera sensor
            self.gym.set_camera_location(cam_1, env, camera_1_position, camera_1_target)
            self.gym.set_camera_location(cam_2, env, camera_2_position, camera_2_target)
            self.camera_handles.append([])
            self.camera_handles[i].append(cam_1)
            self.camera_handles[i].append(cam_2)

        self.unfinished_box_idxs = copy.deepcopy(self.box_idxs)
        
        # Point camera at environments
        cam_pos = gymapi.Vec3(4, 3, 3)
        cam_target = gymapi.Vec3(-4, -3, 0)

        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        
        # ==== prepare tensors =====
        # from now on, we will use the tensor API that can run on CPU or GPU
        self.gym.prepare_sim(self.sim)
        
        # initial hand position and orientation tensors
        init_pos = torch.Tensor(self.init_pos_list).view(self.num_envs, 3).to(self.device)
        init_rot = torch.Tensor(self.init_rot_list).view(self.num_envs, 4).to(self.device)
        
        # get rigid body state tensor
        _rb_states      = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states       = gymtorch.wrap_tensor(_rb_states)
        # print("rb_state shape", rb_states.shape)
        # print("rb_state shape", rb_states.shape[0])
        # print("rb_state shape", rb_states.shape[1])

        rb_states_clone = self.rb_states.clone()
        # print("device: ", rb_states_clone.device)
        # print("rb_states_clone[tray_idxs, :] =", rb_states_clone[tray_idxs])
        init_tray_rot   = rb_states_clone[self.tray_idxs, 3:7]
        init_cube_rot   = rb_states_clone[self.box_idxs, 3:7]
        print("box_ids:", self.box_idxs)

        # get dof state tensor
        _dof_states             = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states         = gymtorch.wrap_tensor(_dof_states)  #torch.Size([24, 2])
        self.init_dof_states    = self.dof_states.clone().view(self.num_envs, 8, 2)
                
        print("Issacgym piper simulation sytem is completed")

    
    def step(self, predicted_actions):
        # self.time_envs[:] += 1
        # self.t+=1
        self.step_physics()
        # print("step physics")
        self.refresh_tensors()
        # print("refresh_tensors")
        self.apply_rl_actions(predicted_actions)
        self.render() 
        #get the images
        self.finish_ep()
        
        return 
    
    def wait(self):
        self.step_physics()
        self.render()
    
    def get_observations(self):
        
        joint_positions     = self.dof_states[:, 0].view(self.num_envs, 8)
        # print(f"joint_positions = {joint_positions}")
        return joint_positions.tolist()
    
    def get_states(self):
        '''return observations of the environment's states
        
        Return
        ------
        obs: list()
            [goal_position x3, goal_orientation x4, end_effector_position x3, end_effector_orientation x4]
        '''
        #TODO: add other states in the environment like velocities, torques, positions of each joint.
        
        
        goal_position                       = self.cube_pose #list(3)    #TODO: change this when the goal_pos refers to the real cube
        goal_position_normalized            = [(goal_position[0]  + 0.65) / 1.3 ,
                                               (goal_position[1]  + 0.75) / 1.5 ,
                                               (goal_position[2]  + 0.65 )/ 1.3 ]
        # goal_orientation            = self.goal_rot  #list(4)   #TODO: change this when the goal_rot refers to the real cube
        joint_angles                        = [0.0] * 6
        joint_velocities_normalized         = [0.0] * 6
        joint_angles[:]                     = self.piper_dof_states['pos'][:6]   #list(6) 
        joint_angles[:]             = (joint_angles[:] - self.piper_lower_limits[:]) / (self.piper_upper_limits[:] - self.piper_lower_limits[:])
        # print("normalized joint angles= ", joint_angles_normalized)
        # for j in range(len(joint_angles_normalized)):
        #     if joint_angles_normalized[j] >1.0:
        #         print("joint ", j, "exceeds limit= ", joint_angles[j])
                
        
        joint_velocities_normalized[:]      = (self.piper_dof_states['vel'][:6] + 3.0) / 6.0  #list(6) 
        
        """print("(joint vel) =", joint_velocities)
        print(f"joint_angles (len={len(joint_angles)}): {joint_angles[0]}, joint_vel (len={len(joint_velocities)}) : {joint_velocities[0]}")
    
        
        print(f"EE_position: {end_effector_position}")
        
        
        print("size EE_pose", len(ee_p_dicts))
        
        print(f"body length: {len(self.piper_body_states['pose']['p'])}")
        print("EE_pos:", ee_p_dicts[-1])
        ee_position_x = [p['x'] for p in ee_p_dicts]
        ee_position_y = [p['y'] for p in ee_p_dicts]
        ee_position_z = [p['z'] for p in ee_p_dicts]
        end_effector_position = ee_position_x + ee_position_y + ee_position_z"""
        
        # print("EE_position_bf", end_effect
        end_effector_position               = self.piper_body_states['pose']['p'][-1] 
        end_effector_position_normalized    = [(end_effector_position['x']  + 0.65) / 1.3 ,
                                               (end_effector_position['y']  + 0.75) / 1.5 ,
                                               (end_effector_position['z']  + 0.65 )/ 1.3 ]
        end_effector_velocity               = (self.piper_body_states['vel']['linear'][-1])
        end_effector_velocity               = [end_effector_velocity['x'],
                                               end_effector_velocity['y'],
                                               end_effector_velocity['z']]
        # print("type(joint vel) =", type(joint_velocities), "type(ee_vel)", type(end_effector_velocity), "type piper dof state", type(self.piper_dof_states['vel']))
        end_effector_velocity_normalized     = [(end_effector_velocity[i] + 3.0) / 6.0 for i in range(3)]
        # print("(ee vel normalizeds  ) =", end_effector_velocity_normalized)
        
        velocity_target =  [(self.piper_velocity_target[i] + 3.0 )/ 6.0 for i in range(len(self.piper_velocity_target))]
        """print("velo_target=", velocity_target)
        print("len=", len(velocity_target))
        end_effector_position = [list(pos) for pos in end_effector_position] 
        print("EE_position_af", end_effector_position)
        print("EE_vel=", end_effector_velocity)
        end_effector_orientation    = self.piper_body_states['pose']['r'] #dict
        
        print("size EE_rot", len(ee_r_dicts))
        
        end_effector_orientation = self.piper_body_states['pose']['r'][-3]      # list of 9 dicts
        end_effector_orientation    = [end_effector_orientation['x'],
                                       end_effector_orientation['y'],
                                       end_effector_orientation['z'],
                                       end_effector_orientation['w']]
        end_effector_orientation = [list(rot) for rot in end_effector_orientation]  
        print("EE_orientation_af", end_effector_orientation)
        
        obs = [goal_position, goal_orientation, end_effector_position, end_effector_orientation]
        obs = np.array(goal_position +  goal_orientation +  end_effector_position + end_effector_orientation)
        -------------------3-----------------4---------------------3-----------------------4--------------------------3---------------------6--------------6----------------#
        obs = np.array(goal_position + goal_orientation + end_effector_position + end_effector_orientation + end_effector_velocity + joint_angles + joint_velocities, dtype=np.float32)
        """
        #--------------------------3----------------------------3-----------------------------3-----------------------------------6----------------------------6----------------#
        
        # print("lens=", len(goal_position_normalized), len(end_effector_position_normalized), len(end_effector_velocity_normalized), len(joint_angles), len(joint_velocities_normalized))
        obs = np.array(goal_position_normalized + end_effector_position_normalized + end_effector_velocity_normalized + joint_angles + joint_velocities_normalized + velocity_target, dtype=np.float32)
        
        # print("obs = ", obs)
        obs_tensor = torch.from_numpy(obs).to("cuda:0")
        # print(f"type of obs_tensor = {type(obs_tensor)}, device = {obs_tensor.get_device()}")
        # obs.flatten()
        # if self.debug:
            # print("OBS:", obs)
            # print("size:", len(obs))
        return obs, obs_tensor
        # return self.cube_states, self.piper_dof_states, self.piper_body_states
     
    def step_physics(self):
        # Step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        
    def refresh_tensors(self):
        # refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

    def render(self):
        # Step rendering
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        
        # self.image_processing(writers)
        
    def finish_ep(self):
        self.gym.draw_viewer(self.viewer, self.sim, False)
        self.gym.sync_frame_time(self.sim)
        self.gym.end_access_image_tensors(self.sim)

    def reset(self):
        print('#-----------------RESETTING ENV-----------------#')
        
        # print("restart_indices", restart_indices)
        # print("restart_indices", restart_indices.shape)
        # states_buffer                   = self.dof_states.clone().view(self.num_envs, 8, 2)
        # states_buffer                   = self.init_dof_states
        # dof_state                       = states_buffer.view(self.num_envs * 8,2)
        random_pose                     = self.random_pos_tensor()
        
        # tray_reset_idxs                 = torch.tensor(self.tray_idxs, device=self.device)
        # box_reset_idxs                  = torch.tensor(self.box_idxs, device=self.device)
        
        # print("rb_states[box_reset_idxs, :3] ", rb_states[box_reset_idxs, :3])
        # print("rb_states[box_reset_idxs, :3] shape ", rb_states[box_reset_idxs, :3].shape)
        # print("random_pose[restart_indices, 1:num_box + 1, :3]", random_pose[restart_indices, 1:num_box + 1, :3])
        # print("random_pose[restart_indices, 1:num_box + 1, :3] shape", random_pose[restart_indices, 1:num_box + 1, :3].reshape(-1, 3).shape)
        self.rb_states[self.tray_idxs, :3]  = random_pose[:, 0, :3]
        self.rb_states[self.tray_idxs, 3:7] = torch.tensor([0., 0., 0., 1.0], device=self.device)
        # for n in range(num_box):
        # rb_states[box_reset_idxs, :3]   = random_pose[restart_indices, 1:num_box + 1, :3]
        self.rb_states[self.box_idxs, :3]   = random_pose[:, 1:self.num_box + 1, :3] #.reshape(-1, 3)
        
        self.gym.set_rigid_body_state_tensor(self.sim, gymtorch.unwrap_tensor(self.rb_states))
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.init_dof_states))

    def random_pos_tensor(self):
        """
            Return: torch.cat((tray_pose, boxes_pose), dim=1)
        """
        r_min = 0.2
        r_max = 0.4
        theta = 70
        radius_tensor   = torch.FloatTensor(self.num_envs, self.num_box + 1).uniform_(r_min, r_max).to(self.device)       # self.num_envs x num_objs (3)
        theta_tensor    = torch.FloatTensor(self.num_envs, self.num_box + 1).uniform_(- theta/180 * np.pi, theta/180 * np.pi).to(self.device)   
        x_tensor   = radius_tensor * torch.cos(theta_tensor)           # self.num_envs x 1
        y_tensor   = radius_tensor * torch.sin(theta_tensor)           # self.num_envs x 1
        z_tensor_tray   = torch.full([self.num_envs], self.table_dims.z).to(self.device)                   # self.num_envs x 1
        z_tensor_cube   = torch.full([self.num_envs], self.table_dims.z + 0.5 * self.cube_size).to(self.device)   # self.num_envs x 1
        
        # random_pose     = torch.stack((torch.stack([x_tensor[:, 0], y_tensor[:, 0], z_tensor_tray], dim=1), 
        #                                torch.stack([x_tensor[:, 1], y_tensor[:, 1], z_tensor_cube], dim=1)), dim=1).to(self.device)  
        tray_pose    = torch.stack([x_tensor[:, 0], y_tensor[:, 0], z_tensor_tray], dim=1).unsqueeze(1)
        boxes_pose  = torch.stack([torch.stack([x_tensor[:, j+1], y_tensor[:, j+1], z_tensor_cube], dim=1) for j in range(self.num_box)], dim=1)
        
        # print("tray_pose dim", tray_pose.shape)
        # print("boxes_pose dim", boxes_pose.shape)
        # print("tray_pose dim", tray_pose)
        # print("boxes_pose dim", boxes_pose)
        random_pose     = torch.cat((tray_pose, boxes_pose), dim=1)
        
        # print("random pose: ", random_pose)
        # print("random pose shape: ", random_pose.shape)
        
        return random_pose

    def apply_rl_actions(self, predicted_actions=None):
        """Specify the actions to be performed by the rl agent(s).

        If no actions are provided at any given step, the rl agents default to
        performing actions specified by SUMO.

        Parameters
        ----------
        predicted_actions : array_like
            list of actions provided by the RL algorithm
        """
        if predicted_actions is None:
            return
        
        # 1.   to torch, float32, correct device
        if isinstance(predicted_actions, np.ndarray):
            action_tensor = torch.from_numpy(predicted_actions.astype(np.float32))
        else:                           # already a tensor
            action_tensor = predicted_actions.float().cpu()
            
        action_tensor = action_tensor.flatten().contiguous()
        expected = self.num_envs * 8
        if action_tensor.numel() != expected:
            raise ValueError(f"Expected {expected} targets, got {action_tensor.numel()}")

        # print(action_tensor.device, action_tensor.shape, action_tensor.is_contiguous())

        # print(f"action_tensor = {action_tensor}")
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(action_tensor))
            
    def apply_rl_actions_force(self, rl_actions=None):
        
        # Step the physics
        # self.gym.simulate(self.sim)
        # self.gym.fetch_results(self.sim, True)
        
        # ignore if no actions are issued
        if rl_actions is None:
            return

        for i in range(self.num_envs):
            # action_np = rl_clipped[i].detach().cpu().numpy().astype(np.float32)
            # print(f"rl_actions= {rl_actions}, dim= {len(rl_actions)}, type= {type(rl_actions)}")
            
            rl_actions      =   np.concatenate((rl_actions, [0.0, 0.0]))
            
            # action_np = rl_actions.astype(np.float32) * 1000.0 / 3 #100.0
            action_np       =   rl_actions.astype(np.float32) * 3
            # print(f"rl_actions {action_np}")
            # action_np[1] =   abs(action_np[1])
            action_np[1] = action_np[1] * 2.0
            action_np[2] = action_np[2] * 1.5
            # action_np[-1] = 0.0
            # action_np[-2] = 0.0
            # action_np[6] =   abs(action_np[6])
            # action_np[7] = - abs(action_np[7])
            # print("action = ", action_np)
            force_tensor =   torch.from_numpy(action_np).to("cpu")
            
            # print(f"force tensor = {force_tensor}")
            # self.gym.set_actor_dof_position_targets(self.envs[i], self.piper_handles[i], action_np)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(force_tensor))
    
    def apply_rl_actions_velocity(self, rl_actions=None):
        
        # Step the physics
        # self.gym.simulate(self.sim)
        # self.gym.fetch_results(self.sim, True)
        
        # ignore if no actions are issued
        if rl_actions is None:
            return

        for i in range(self.num_envs):
            # action_np = rl_clipped[i].detach().cpu().numpy().astype(np.float32)
            # print(f"rl_actions= {rl_actions}, dim= {len(rl_actions)}, type= {type(rl_actions)}")
            
            rl_actions      =   np.concatenate((rl_actions, [0.0, 0.0]))
            
            # action_np = rl_actions.astype(np.float32) * 1000.0 / 3 #100.0
            action_np       =   rl_actions.astype(np.float32) * self.action_scale
            # print(f"rl_actions {action_np}")
            # action_np[1] =   abs(action_np[1])
            # action_np[1] = action_np[1] * 2.0
            # action_np[2] = action_np[2] * 1.5
            # action_np[-1] = 0.0
            # action_np[-2] = 0.0
            # action_np[6] =   abs(action_np[6])
            # action_np[7] = - abs(action_np[7])
            # print("action = ", action_np)
            velocity_tensor =   torch.from_numpy(action_np).to("cpu")
            
            # print(f"force tensor = {force_tensor}")
            # self.gym.set_actor_dof_position_targets(self.envs[i], self.piper_handles[i], action_np)
            self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(velocity_tensor))
            
        
    def create_dataset_dirs(self):
        try:
            os.makedirs("dataset/success_seed_" + str(self.seed), exist_ok=True)
            os.makedirs("dataset/failure_seed_" + str(self.seed), exist_ok=True)
            os.makedirs("videos/success_seed_" + str(self.seed), exist_ok=True)
            os.makedirs("videos/failure_seed_" + str(self.seed), exist_ok=True)
        except FileExistsError:
            pass
                
    def create_video_writer(self, env): 
        """
            create a new pair of video writers in an envronment
            Return: color_writer_1, color_writer_2 
        """
        color_writer_1  = cv2.VideoWriter(f'color_1_env_{env}.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))
        color_writer_2  = cv2.VideoWriter(f'color_2_env_{env}.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))
        return color_writer_1, color_writer_2 
    # depth_writer_1    = cv2.VideoWriter('depth_1.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))
    # depth_writer_2    = cv2.VideoWriter('depth_2.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))
    
    def image_capture(self):
        self.render()
        color_image_1   = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[0][0], gymapi.IMAGE_COLOR)
        color_image_2   = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[0][1], gymapi.IMAGE_COLOR)
        # depth_image_1   = gym.get_camera_image(sim, envs[0], camera_handles[0][0], gymapi.IMAGE_DEPTH)
        # depth_image_2   = gym.get_camera_image(sim, envs[0], camera_handles[0][1], gymapi.IMAGE_DEPTH)
        # print("depth_image_shape",depth_image_1)
        
        img_np_1 = color_image_1.reshape((self.camera_props.height, self.camera_props.width, 4))
        img_np_2 = color_image_2.reshape((self.camera_props.height, self.camera_props.width, 4))
        # print("distance",depth_image_1[100][100])
        # depth_colormap_1 = cv2.convertScaleAbs(depth_image_1, alpha=1)
        # Normalize to [0, 1] or [-1, 1] if needed
        
        # depth_norm_1 = (depth_image_1 - np.min(depth_image_1)) / (np.max(depth_image_1) - np.min(depth_image_1))
        # depth_norm_2 = (depth_image_2 - np.min(depth_image_2)) / (np.max(depth_image_2) - np.min(depth_image_2))
        
        
        # print("depth_image_shape",depth_colormap_1)
        # print("distance",depth_colormap_1[100][100])
        # depth_colormap_2 = cv2.convertScaleAbs(depth_image_2, alpha=1)
        # print("capture!")
        rgb_image_1 = img_np_1[:, :, :3]
        rgb_image_2 = img_np_2[:, :, :3]
        # print("size of color", rgb_image_1.shape)
        # cv2.imshow("cam1", np.asanyarray(rgb_image_1))
        # cv2.imshow("cam2", np.asanyarray(rgb_image_2))
        cv2.waitKey(1)
        # self.finish_ep()
        
        return [rgb_image_1, rgb_image_2]   
     
    def image_processing(self, writers):
        for i in range(self.num_envs):
            color_image_1   = self.gym.get_camera_image(self.sim, self.envs[i], self.camera_handles[i][0], gymapi.IMAGE_COLOR)
            color_image_2   = self.gym.get_camera_image(self.sim, self.envs[i], self.camera_handles[i][1], gymapi.IMAGE_COLOR)
            # depth_image_1   = gym.get_camera_image(sim, envs[0], camera_handles[0][0], gymapi.IMAGE_DEPTH)
            # depth_image_2   = gym.get_camera_image(sim, envs[0], camera_handles[0][1], gymapi.IMAGE_DEPTH)
            # print("depth_image_shape",depth_image_1)
            
            img_np_1 = color_image_1.reshape((self.camera_props.height, self.camera_props.width, 4))
            img_np_2 = color_image_2.reshape((self.camera_props.height, self.camera_props.width, 4))
            # print("distance",depth_image_1[100][100])
            # depth_colormap_1 = cv2.convertScaleAbs(depth_image_1, alpha=1)
            # Normalize to [0, 1] or [-1, 1] if needed
            
            # depth_norm_1 = (depth_image_1 - np.min(depth_image_1)) / (np.max(depth_image_1) - np.min(depth_image_1))
            # depth_norm_2 = (depth_image_2 - np.min(depth_image_2)) / (np.max(depth_image_2) - np.min(depth_image_2))
            
            
            # print("depth_image_shape",depth_colormap_1)
            # print("distance",depth_colormap_1[100][100])
            # depth_colormap_2 = cv2.convertScaleAbs(depth_image_2, alpha=1)

            rgb_image_1 = img_np_1[:, :, :3]
            rgb_image_2 = img_np_2[:, :, :3]

            # rgb_image_1 = rgb_image_1.astype(np.uint8)    # Ensure type
            # rgb_image_2 = rgb_image_2.astype(np.uint8)    # Ensure type
            
            # bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

            # print("color image", bgr)
            # print("color image shape", img_np.shape)
            # cv2.imshow("Color", bgr_image)
            # cv2.imshow("Depth_1",depth_colormap_1)
            # cv2.imshow("Depth_2",depth_image_2)
            writers[i][0].write(rgb_image_1)
            writers[i][1].write(rgb_image_2)
        
        # depth_writer_1.write(depth_colormap)
        # depth_writer_1.write(depth_colormap_1)  #This is for visualization 
        # depth_writer_2.write(depth_colormap_2)  #This is for visualization 
        
        # depth_writer_1.write(cv2.cvtColor(depth_colormap_1, cv2.COLOR_GRAY2BGR))  
        # depth_writer_2.write(cv2.cvtColor(depth_colormap_2, cv2.COLOR_GRAY2BGR)) 

    def store_joints_states(self, env: int, ep: int, joints_data: object, success: bool):
        """
        file = h5py.File("dataset/joint_angles.h5py", 'w')
        group = file.create_group(f"env_{env}_ep_{ep}")
        dataset = file.store_joints_states()
        """
        if success:
            with open(f"dataset/success_seed_{self.seed}/env_{env}_ep_{ep}.csv", 'w') as f:
                writer = csv.writer(f)
                writer.writerows(joints_data.numpy())
        else:
            with open(f"dataset/failure_seed_{self.seed}/env_{env}_ep_{ep}.csv", 'w') as f:
                writer = csv.writer(f)
                writer.writerows(joints_data.numpy())


    def stop_simulation(self):
        print("Done")

        # print(f"Total frozen resets: {frozen_counter}")
        # print(f"Total successes: {success_counter}")
        # cleanup
        # color_writer_1.release()
        # color_writer_2.release()
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        cv2.destroyAllWindows()

