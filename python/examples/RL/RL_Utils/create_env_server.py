"""
Create piper environment
----------------
Create an isaacgym environment to train piper models for performing tasks 
"""

import math
import random
import time
import numpy as np
import ast
import sys
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import json
import torch
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from paho.mqtt import client as mqtt_client 
from threading import Lock

from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

# sys.path.append('/home/ucluser/IsaacGymEnvs/isaacgymenvs')
# from RL_Utils.torch_jit_utils import to_torch, get_axis_params, tensor_clamp, tf_vector, tf_combine


class Gym_env_server():
    def __init__(self, args):
        self.args = args
        self.sim_device         = args['sim_device']
        self.pipeline           = args['pipeline']
        self.graphics_device_id = args['graphics_device_id']
        self.physics            = args['physics_engine']  #flex/ physx
        self.num_threads        = args['num_threads']
        self.subscenes          = args['subscenes']
        self.slices             = args['slices']
        self.num_envs           = args['num_envs']
        self.dist_reward_scale  = args['dist_reward_scale']
        self.rot_reward_scale   = args['rot_reward_scale']
        self.stiffness          = args['stiffness']
        self.damping            = args['damping']
        self.debug              = args['debug']
        self.dt                 = args['dt']
        self.debug_interval     = args['debug_interval']
        self.warmup             = args['warmup']
        self.action_scale       = args['action_scale']
        
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
        self.goal_set           = [[0.350,0.45,0.01],[0.62,0.1,0.08],[-0.5,0.1,-0.31],[-0.5,0.1,0.31],[-0.48,0.176,0.302],[0.244,0.112,-0.369],
                                    [0.214,0.162,0.178],[-0.04,-0.031,-0.188],[0.423,0.297,-0.376],[0.112,0.297,-0.555],[-0.23,0.297,-0.517],
                                    [-0.559,0.297,-0.089],[-0.074,0.672,-0.031],[0.150,0.472,-0.111],[0.107,0.416,0.110],[0.01,0.632,0.030],[0.106,0.494,-0.068],
                                    [0.154,0.359,-0.04],[-0.110,0.422,0.114],[0.063,0.422,0.145],[-0.106,0.422,-0.118],[0.093,0.443,0.059],
                                    [0.301,0.455,0.308],[-0.37,0.455,0.213],[-0.076,0.441,0.089],[0.222,0.610,-0.111],[0.266,0.333,-0.159],
                                    [0.021,0.483,0.122],[-0.065,0.483,-0.106],[0.123,0.483,0.017],[0.280,0.289,-0.038],[0.282,0.582,0.044],
                                    [0.109,0.304,0.047],[-0.111,0.304,0.043],[0.285,0.388,-0.267],[0.248,0.35,0.245],[-0.11,0.326,0.499],
                                    [-0.133,0.2333,0.489],[0.492,0.329,-0.115],[0.140,0.590,0.254],[-0.023,0.271,-0.304],[-0.384,0.462,-0.165],[0.157,0.459,0.032]]
        
        self.sphere_geom        = None
        self.asset_root         = "/home/ucluser/isaacgym/assets"
        
        self.piper_lower_limits = []
        self.piper_upper_limits = []
        self.piper_mids         = []
        self.piper_num_dofs     = 8
        
        self.writer             = SummaryWriter('logs', comment="")
        
        self.time_counter       = 0
        self.time_ep            = 0
        self.goal_dist_initial  = 0
        
        # self.sims_per_step = 100\
    
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
        
    def create_gym_env(self):
        """Create a parametrized empty gym environment 

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
        gym
            gym instance
        sim
            create an empty simulation with a plane
        viewer
            viewer of the gym environment
        """
            
        # Initialize gym
        self.gym = gymapi.acquire_gym()
            
        # configure sim
        
        # set common parameters
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        
        if self.debug:
            print("sim_device:", self.sim_device)
            
        sim_device_type, compute_device_id = self.parse_device_str(self.sim_device)
        
        # pipeline = self.pipeline.lower()  #use when using line-arguments
        
        # assert (pipeline == 'cpu' or pipeline in ('gpu', 'cuda')), f"Invalid pipeline '{self.pipeline}'. Should be either cpu or gpu."
        use_gpu_pipeline = (self.pipeline in ('gpu', 'cuda'))

        if sim_device_type != 'cuda' and self.physics=='flex':
            print("Can't use Flex with CPU. Changing sim device to 'cuda:0'")
            self.sim_device = 'cuda:0'
            sim_device_type, compute_device_id = self.parse_device_str(self.sim_device)

        if (sim_device_type != 'cuda' and self.pipeline == 'gpu'):
            print("Can't use GPU pipeline with CPU Physics. Changing pipeline to 'CPU'.")
            self.pipeline = 'cpu'
            use_gpu_pipeline = False

        # Default to PhysX
        if self.physics == 'physx':
            physics_engine = gymapi.SIM_PHYSX
            use_gpu = (sim_device_type == 'cuda')
        elif self.physics == 'flex':
            physics_engine = gymapi.SIM_FLEX

        if self.slices is None:
            self.slices = self.subscenes
          
        # set Flex-specific parameters
        if physics_engine == gymapi.SIM_FLEX:
            sim_params.flex.solver_type = 5
            sim_params.flex.num_outer_iterations = 4
            sim_params.flex.num_inner_iterations = 15
            sim_params.flex.relaxation = 0.75
            sim_params.flex.warm_start = 0.8
        # set PhysX-specific parameters
        elif physics_engine == gymapi.SIM_PHYSX:
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 4
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.num_threads = self.num_threads
            sim_params.physx.use_gpu = use_gpu

        # sim_params.use_gpu_pipeline = use_gpu_pipeline  
        sim_params.use_gpu_pipeline = False
        
        if use_gpu_pipeline:
            print("WARNING: Forcing CPU pipeline.")

        #TODO: Consider where sim should be initialized
        self.sim = self.gym.create_sim(compute_device_id, self.graphics_device_id, physics_engine, sim_params)

        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        # Create viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            print("*** Failed to create viewer")
            quit()

        # Add ground plane
        plane_params = gymapi.PlaneParams()
        self.gym.add_ground(self.sim, plane_params)

        self.gym.prepare_sim(self.sim)
        # return gym, sim, viewer

    def load_piper(self):
        # load piper asset
        # asset_root = "../../assets"
        piper_asset_file = "urdf/piper_description/urdf/piper_description.urdf"
        # piper_description_chain = Chain.from_urdf_file(self.asset_root + "/" + piper_asset_file)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.armature = 0.01
        print("Loading asset '%s' from '%s'" % (piper_asset_file, self.asset_root))
        piper_asset = self.gym.load_asset(self.sim, self.asset_root, piper_asset_file, asset_options)
        return piper_asset  #, piper_description_chain

    def load_cube(self):
        # load cube asset
        cube_asset_file = "urdf/cube.urdf"
        print("Loading asset '%s' from '%s'" % (cube_asset_file, self.asset_root))
        cube_asset = self.gym.load_asset(self.sim, self.asset_root, cube_asset_file, gymapi.AssetOptions())
        return cube_asset


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
        
        # params = dict()
        # params['num_envs'] = args.num_envs
        # params['graphics_device_id'] = args.graphics_device_id
        # params['physics_engine'] = args.physics_engine
        # params['num_threads'] = args.num_threads
        # params['total_time'] = args.total_time
        # params['use_gpu'] = args.use_gpu
        # params['use_gpu_pipeline'] = args.use_gpu_pipeline
        # params['compute_device_id'] = args.compute_device_id
        # params['sim_device'] = args.sim_device
        
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
        self.create_gym_env()
        piper_asset = self.load_piper()
        cube_asset  = self.load_cube()
        
        print("Creating %d environments" % self.num_envs)
        num_per_row = int(math.sqrt(self.num_envs))

        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)
            
            # add pipers and cubes into the gym simulation 
            piper_handle = self.gym.create_actor(env, piper_asset, piper_pose, "piper", i , -1)
            # cube_handle = self.gym.create_actor(env, cube_asset, cube_pose, "cube", i , -1)
            
            self.gym.enable_actor_dof_force_sensors(env, piper_handle)
            
            body_dict = self.gym.get_actor_rigid_body_dict(env, piper_handle)
            
            gymutil.draw_lines(self.sphere_geom, self.gym, self.viewer, self.envs[i], goal_pose)

            self.piper_handles.append(piper_handle)
            # self.cube_handles.append(cube_handle)

        # get joint limits and ranges for piper
        piper_dof_props = self.gym.get_actor_dof_properties(self.envs[0], self.piper_handles[0])
        self.piper_lower_limits = piper_dof_props['lower'][:6]
        self.piper_upper_limits = piper_dof_props['upper'][:6]
        piper_ranges = self.piper_upper_limits - self.piper_lower_limits
        self.piper_mids = 0.5 * (self.piper_upper_limits + self.piper_lower_limits)
        self.piper_num_dofs = len(piper_dof_props) - 2

        # override default stiffness and damping values
        piper_dof_props['stiffness'].fill(self.stiffness)
        piper_dof_props['damping'].fill(self.damping)  

        # Now focus just one robot 
        piper_dof_props["driveMode"][:] =  gymapi.DOF_MODE_VEL  #gymapi.DOF_MODE_POS  #A DoF that is set to a specific drive mode will ignore drive commands for other modes.

        # piper_dof_props['stiffness'][:] = 1e10
        # piper_dof_props['damping'][:] = 10.0        #200.0
        for i in range(self.num_envs):
            self.gym.set_actor_dof_properties(self.envs[i], self.piper_handles[i], piper_dof_props)

            # Set piper pose so that each joint is in the middle of its actuation range
            piper_dof_states = self.gym.get_actor_dof_states(self.envs[i], self.piper_handles[i], gymapi.STATE_POS)
            for j in range(self.piper_num_dofs):
                piper_dof_states['pos'][j] = self.piper_mids[j]
            self.gym.set_actor_dof_states(self.envs[i], self.piper_handles[i], piper_dof_states, gymapi.STATE_POS)

        # Point camera at environments
        cam_pos = gymapi.Vec3(4, 3, 3)
        cam_target = gymapi.Vec3(-4, -3, 0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        # print("dof_states shape", dof_states.shape)
        # Set all positions to mid-range for each DOF (first 6 joints)
        dof_states[:self.piper_num_dofs, 0] = torch.tensor(self.piper_mids[:self.piper_num_dofs], device=dof_states.device)
        # print("dof_states shape1", dof_states[:self.piper_num_dofs, 0].shape)
        # Set all velocities to zero for each DOF (first 6 joints)
        dof_states[:self.piper_num_dofs, 1] = 0.0
        self.saved_dof_states = dof_states.clone()
        
        # dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        # dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        # print("dof_states shape", dof_states.shape)
        # dof_states[:, 0][:] = self.piper_mids[:] # positions
        # print("dof_states shape1", dof_states[:, 0].shape)
        # dof_states[:, 1][:] = 0.0  # velocities
        # self.saved_dof_states = dof_states.clone()
        
        print("Issacgym piper simulation is completed")
        
        # self.random_new_goal()


    def step(self, rl_actions):
        self.time_counter += 1
        self.time_ep += 1

        # advance the simulation in the simulator by one step
        self.step_physics()
        
        self.refresh()
        
        #take actions given by RL agent
        # self.apply_rl_actions(rl_actions)
        if rl_actions is None:
            # self.apply_rl_actions_force(np.zeros(6))
            self.apply_rl_actions_velocity(np.zeros(6))
            # print("NONE ACTION IS ACTIVATED")
            self.update()
            
            self.render()
            states, state_tensor = self.get_states()
            # print(f"states: {states}")
            return state_tensor, 0, False
        else:    
            # self.apply_rl_actions_force(rl_actions)
            self.apply_rl_actions_velocity(rl_actions)
        
        

        # render a frame
        self.render()
        
        self.update()
        # actions = np.array(rl_actions) * 0.1
        
        # print("action=" ,actions)
        # print("velocity=", self.piper_dof_states['vel'])
        # diff = []
        # for j in range(len(actions)):
            # diff[j] = actions[j] - self.piper_dof_states['vel'][j]
        # diff = [actions[j] - self.piper_dof_states['vel'][j] for j in range(len(actions))]
        # print("diff=", diff)
        # print("mean=", np.array(diff).mean())
        
        
        
        states_np, states_tensor = self.get_states()
        # print("states")
        # print(', '.join(f'{q:.2f}' for q in states_np))
        # print(f"type = {type(states_tensor)}")

        # test if the environment should terminate due to a collision or the
        # time horizon being met
        # done = (time_counter >= self.num_envs * (1_steps + env_params.horizon) ) #or crash)
        # done = (self.time_counter >= horizon )
        #TODO: Need to set the proper done conditions. Like grab it successfully or crash (HOW??)
        
        # compute the info for each agent
        infos = {}

        # compute the reward
        # rl_clipped = clip_actions(rl_actions)   #TODO: Make sure whether clip_actions() function is necessary
        reward, result = self.compute_reward(states_np, rl_actions)  #TODO: fix this when crash is availalbe fail=crash 

        return states_tensor, reward, result  # infos
    
    def wait(self):
            # self.time_counter += 1
            # self.step_counter += 1

            # self.gym.clear_lines(self.viewer)

            self.render()
    
            
    def random_new_goal(self, randomness=True):
        
        if randomness:
            max_radius = 0.5
            min_radius = 0.2
            # Random point in spherical coordinates
            r = np.random.uniform(min_radius, max_radius)  # Random radius [0, 0.8]
            theta = np.random.uniform(0, 2 * np.pi)  # Random azimuthal angle [0, 2pi]
            phi = np.random.uniform(0, np.pi / 2)  # Random polar angle [0, pi/2] (upper hemisphere)

            # Convert spherical to Cartesian coordinates
            x = r * np.sin(phi) * np.cos(theta)
            z = r * np.sin(phi) * np.sin(theta)
            y = r * np.cos(phi) 
            # z = np.clip(z, 0.1, 0.8)
        else:
            # set_length = len(self.goal_set)
            goal_pose = random.choice(self.goal_set)
            x,y,z   = goal_pose[0],goal_pose[1],goal_pose[2]
  
        self.cube_pose = [x,y,z]    
        # Random orientation as a normalized quaternion
        rand_quat = np.random.randn(4)
        rand_quat /= np.linalg.norm(rand_quat)

        self.goal_rot = [rand_quat[0], rand_quat[1], rand_quat[2], rand_quat[3]]
        goal_pose = gymapi.Transform()
        goal_pose.p = gymapi.Vec3(x, y, z)
        goal_pose.r = gymapi.Quat(rand_quat[0], rand_quat[1], rand_quat[2], rand_quat[3])
        print(f"New goal pose: {self.cube_pose[0]:.2f}, {self.cube_pose[1]:.2f}, {self.cube_pose[2]:.2f}")
        for i in range(self.num_envs):
            gymutil.draw_lines(self.sphere_geom, self.gym, self.viewer, self.envs[i], goal_pose)
        # self.render()
    
    #COMPLETE
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
        # print("(joint vel) =", joint_velocities)
        # print(f"joint_angles (len={len(joint_angles)}): {joint_angles[0]}, joint_vel (len={len(joint_velocities)}) : {joint_velocities[0]}")
    
        
        # print(f"EE_position: {end_effector_position}")
        
        
        # print("size EE_pose", len(ee_p_dicts))
        
        # print(f"body length: {len(self.piper_body_states['pose']['p'])}")
        # print("EE_pos:", ee_p_dicts[-1])
        # ee_position_x = [p['x'] for p in ee_p_dicts]
        # ee_position_y = [p['y'] for p in ee_p_dicts]
        # ee_position_z = [p['z'] for p in ee_p_dicts]
        # end_effector_position = ee_position_x + ee_position_y + ee_position_z
        
        # print("EE_position_bf", end_effect
        end_effector_position               = self.piper_body_states['pose']['p'][-3] 
        end_effector_position_normalized    = [(end_effector_position['x']  + 0.65) / 1.3 ,
                                               (end_effector_position['y']  + 0.75) / 1.5 ,
                                               (end_effector_position['z']  + 0.65 )/ 1.3 ]
        end_effector_velocity               = (self.piper_body_states['vel']['linear'][-3])
        end_effector_velocity               = [end_effector_velocity['x'],
                                               end_effector_velocity['y'],
                                               end_effector_velocity['z']]
        # print("type(joint vel) =", type(joint_velocities), "type(ee_vel)", type(end_effector_velocity), "type piper dof state", type(self.piper_dof_states['vel']))
        end_effector_velocity_normalized     = [(end_effector_velocity[i] + 3.0) / 6.0 for i in range(3)]
        # print("(ee vel normalizeds  ) =", end_effector_velocity_normalized)
        
        velocity_target =  [(self.piper_velocity_target[i] + 3.0 )/ 6.0 for i in range(len(self.piper_velocity_target))]
        # print("velo_target=", velocity_target)
        # print("len=", len(velocity_target))
        # end_effector_position = [list(pos) for pos in end_effector_position] 
        # print("EE_position_af", end_effector_position)
        # print("EE_vel=", end_effector_velocity)
        # end_effector_orientation    = self.piper_body_states['pose']['r'] #dict
        
        # print("size EE_rot", len(ee_r_dicts))
        
        # end_effector_orientation = self.piper_body_states['pose']['r'][-3]      # list of 9 dicts
        # end_effector_orientation    = [end_effector_orientation['x'],
                                    #    end_effector_orientation['y'],
                                    #    end_effector_orientation['z'],
                                    #    end_effector_orientation['w']]
        # end_effector_orientation = [list(rot) for rot in end_effector_orientation]  
        # print("EE_orientation_af", end_effector_orientation)
        
        # obs = [goal_position, goal_orientation, end_effector_position, end_effector_orientation]
        # obs = np.array(goal_position +  goal_orientation +  end_effector_position + end_effector_orientation)
        #-------------------3-----------------4---------------------3-----------------------4--------------------------3---------------------6--------------6----------------#
        # obs = np.array(goal_position + goal_orientation + end_effector_position + end_effector_orientation + end_effector_velocity + joint_angles + joint_velocities, dtype=np.float32)
        
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
        
        
    #COMPLETE  
    def update(self):
        '''Update the state variables
        
        Definitions
        ----------
        piper_dof_states: 
            np.ndarray[RigidBodyState] [('pose',[('p',[x,y,z]),('r', [(x,y,z,w)])]), ('vel', [('linear',[x,y,z]),('angular', [x,y,z])])]
        cube_states:   
            np.ndarray[RigidBodyState] [('pose',[('p',[x,y,z]),('r', [(x,y,z,w)])])
        '''
        for i in range(self.num_envs):
            # self.cube_states        = self.gym.get_actor_rigid_body_states(self.envs[i],    self.cube_handles[i],   gymapi.STATE_POS) 
            self.piper_dof_states       = self.gym.get_actor_dof_states(self.envs[i],           self.piper_handles[i],  gymapi.STATE_ALL)  
            #gymapi.DofState  ([('pos', '<f4'), ('vel', '<f4')]):   piper_dof_state['pos'][i]
            self.piper_body_states      = self.gym.get_actor_rigid_body_states( self.envs[i],   self.piper_handles[i],  gymapi.STATE_ALL)  
            
            self.piper_forces           = self.gym.get_actor_dof_forces(self.envs[i], self.piper_handles[i])
            
            self.piper_velocity_target  = self.gym.get_actor_dof_velocity_targets(self.envs[i], self.piper_handles[i])
            
            # print(f"forces: {self.piper_forces}")
            
    #COMPLETE    
    def refresh(self):
        # self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        # self.gym.refresh_jacobian_tensors(self.sim)
        # self.gym.refresh_mass_matrix_tensors(self.sim)
    #COMPLETE
    
    def step_physics(self):
        # Step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

    def render(self):
        # Step rendering
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, False)
        self.gym.sync_frame_time(self.sim)

    def init_episode(self):
        self.step_physics()
        self.render()
        self.update()
        
        states, states_tensor = self.get_states()
        current_EE_pose     = states[3:6]
        self.goal_dist_initial = np.linalg.norm(np.array(current_EE_pose) - np.array(self.cube_pose))
        
        return states_tensor


    def reset(self):
        # Reset all DOF states (positions and velocities) to zero and clear forces
        # Acquire the full DOF state tensor, modify and set it back
        # dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        # dof_states = gymtorch.wrap_tensor(self.dof_state_tensor)
        # print(f"dof_states: {dof_states[:, 0]}")
        # dof_states=self.dof_states
        # dof_states[:, 0][:] = 0.0  # positions
        # dof_states[:, 1][:] = 0.0  # velocities
        print('#-----------------RESETTING ENV-----------------#')
        self.step_physics()
        
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.saved_dof_states))
        
        # self.apply_rl_actions_force(np.zeros(6))
        self.apply_rl_actions_velocity(np.zeros(6))
        self.render()

        # Clear any graphical debug lines and choose a new goal
        self.gym.clear_lines(self.viewer)
        self.time_ep = 0
        self.random_new_goal(True)
        # Allow physics to settle at zero configuration

        # time.sleep(1)
        self.update()

            
    #COMPLETE
    def apply_rl_actions(self, rl_actions=None):
        """Specify the actions to be performed by the rl agent(s).

        If no actions are provided at any given step, the rl agents default to
        performing actions specified by SUMO.

        Parameters
        ----------
        rl_actions : array_like
            list of actions provided by the RL algorithm
        """
        # ignore if no actions are issued
        if rl_actions is None:
            return
        
        # if self.debug:  # and self.time_counter % self.debug_interval == 0 :
            # print("rl_actions:", rl_actions)
            
        # rl_clipped = self.clip_actions(rl_actions)
        # if self.debug and self.time_counter % self.debug_interval == 0 :
            # print("rl_clipped:", rl_clipped)
        for i in range(self.num_envs):
            # action_np = rl_clipped[i].detach().cpu().numpy().astype(np.float32)
            # print(f"rl_actions= {rl_actions}, type= {type(rl_actions)}")
            action_np = rl_actions.astype(np.float32)       #rl_actions[i].detach().cpu().numpy().astype(np.float32)
            # print(f"action_np = {action_np}, type= {type(action_np)}")
            self.gym.set_actor_dof_position_targets(self.envs[i], self.piper_handles[i], action_np)
            
        
        # self._apply_rl_actions(rl_clipped)
    
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
    
    def apply_rl_actions_force_warmup(self, rl_actions=None):

        # ignore if no actions are issued
        if rl_actions is None:
            return

        for i in range(self.num_envs):
            # action_np = rl_clipped[i].detach().cpu().numpy().astype(np.float32)
            # print(f"rl_actions= {rl_actions}, type= {type(rl_actions)}")

            # action_np = rl_actions.astype(np.float32) * 1000.0 / 3 #100.0
            # arr             =   np.array([0.0, 0.0])
            rl_actions       =   np.concatenate((rl_actions, [0.0, 0.0]))
            
            action_np       =   rl_actions.astype(np.float32) * 100
            
            force_tensor =   torch.from_numpy(action_np).to("cpu")
            # action_np[1] = action_np[1] * 1.5
            # action_np[2] = action_np[2] * 1.5
            # action_np[4] = action_np[4] * 2
            # action_np[-1] = 0.0
            # action_np[-2] = 0.0
            # print(f"force tensor = {force_tensor}")
            # self.gym.set_actor_dof_position_targets(self.envs[i], self.piper_handles[i], action_np)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(force_tensor))
            
            
    #ALMOST COMPLETE: waiting for adding option for clipping or not and change the type of action, not to use tensor
    def clip_actions(self, rl_actions=None):
        """Clip the actions passed from the RL agent.

        Parameters
        ----------
        rl_actions : array_like
            list of actions provided by the RL algorithm

        Returns
        -------
        array_like
            The rl_actions clipped according to the box or boxes
        """
        # ignore if no actions are issued
        if rl_actions is None:
            return

        # clip according to the action space requirements
        rl_actions_clipped = torch.clamp(rl_actions, 
                                         min=torch.tensor(self.piper_lower_limits, device=rl_actions.device), 
                                         max=torch.tensor(self.piper_upper_limits, device=rl_actions.device))
        
        
        # if not np.array_equal(rl_actions, rl_actions_clipped): 
            # print("actions is clipped") 
            
        return rl_actions_clipped

    #COMPLETE
    def compute_reward(self, states, rl_actions=None, **kwargs):
        """Reward function for the RL agent(s).

        MUST BE implemented in new environments.
        Defaults to 0 for non-implemented environments.

        Parameters
        ----------
        rl_actions : array_like
            actions performed by rl vehicles
        kwargs : dict
            other parameters of interest. Contains a "fail" element, which
            is True if a vehicle crashed, and False otherwise
            
        Reward designs
        ----------
            - reward:
                - finish the task before timeout
            - penalties:
                - Euclidean distance between the desired position and the end-effector's position  
                - Difference between the desired posture and its real posture (rotations)
                - Too much acceleration
                - Crash

        Returns
        -------
        reward : float
        """
        if rl_actions is None:
            return 0, False
        
        state = np.copy(states)
        #TODO: come back here to update when the states is more detailed
        #state = array[(self.cube_states, self.piper_dof_states, self.piper_body_states)]
        
        #Goal end-effector's position: input of the RL agent 
        #TODO: the goal position now is the cube position, the orientation is the grabbing posture
        goal_pose           = self.cube_pose  #list
        goal_pose_tensor    = torch.tensor(goal_pose)
        goal_rot            =  self.goal_rot
        goal_rot_tensor     = torch.tensor(goal_rot)
        
        
        #-------------------3-----------------3----------------------3---------------------6--------------6----------------#
        # obs = np.array(goal_position + end_effector_position + end_effector_velocity + joint_angles + joint_velocities, dtype=np.float32)
        
        
        #Current end-effector's position
        #TODO: right now, it is assumed that the end-effector is the link8, but in the future, it should be in the middle of the gripper
        # [goal_position 0-2, goal_orientation 3-6, end_effector_position 7-9, end_effector_orientation 10-13]
        
        current_EE_pose     = state[3:6]
        # current_EE_pose = state[2]['pose']['p'][7]  #dict
        # current_EE_pose = [current_EE_pose['x'], 
        #                    current_EE_pose['y'], 
        #                    current_EE_pose['z']]  #list
        current_EE_pose_tensor = torch.tensor(current_EE_pose)
        
        #piper_body_states: state[2]['pose'/'vel']['p'/'r'][link: 0,1,2,3,4,5,6,7][x,y,z: 0,1,2]
        
        # distance from EE to the goal
        dist = torch.norm(current_EE_pose_tensor - goal_pose_tensor, p=2, dim=-1)
        
        # dist_reward = 1.0 / (1.0 + (2 * dist) ** 2)
        
        dist_reward = (2 * self.goal_dist_initial) / (self.goal_dist_initial + dist) - 1
        dist_reward = dist_reward.detach().cpu().item() 
        height_reward = 0
        if current_EE_pose[1] <= 0.1:
            height_reward = current_EE_pose[1] - 0.1
        
        
        # dist_reward = - 0.5 * (dist ** 2)
        # dist_reward = dist
        # dist_reward *= dist_reward
        # dist_reward = torch.where(dist <= 0.2, 1 - dist, - dist_reward)
        
        # dist_reward = torch.where(dist <= 0.2, 2 * dist_reward, dist_reward)
        
        
        #Current end-effector's rotations
        '''current_EE_rot = state[26:30]
        current_EE_rot_tensor = torch.tensor(current_EE_rot)'''
        # current_EE_rot = state[2]['pose']['r'][7]
        # current_EE_rot_tensor = torch.tensor([current_EE_rot['x'],
        #                                       current_EE_rot['y'],
        #                                       current_EE_rot['z'],
        #                                       current_EE_rot['w']])
        
        joint_velocity_targets   =  state[21:29]
        mean_value = abs(joint_velocity_targets).mean()
        # if self.time_counter % self.debug_interval == 0: print("mean_value=", mean_value)
        if mean_value > 3:
            velo_reward = - 0.5 * ((mean_value - 3) ** 2)
        else:
            velo_reward = 0.0


        
        
        rewards = self.dist_reward_scale * dist_reward + height_reward * 0.5 + velo_reward * 1.0  #- math.sqrt(self.time_counter)
        
        self.writer.add_scalar('Reward per step', rewards, self.time_ep)
        
        
        
        if (dist <= 0.10):
            success  = True
            # print("DONE DIST = ", dist.item())
            rewards += 1500
        else:
            success  = False
            rewards -= 1
            # print("NOT DONE DIST = ", dist.item())            
        if self.debug and self.time_counter % self.debug_interval == 0:
            print(f"step: {self.time_counter}       dist= {dist:.3f}")
            print(f"rewards: {rewards:3f}   dist_reward: {dist_reward:.3f}  height_reward: {height_reward:.3f}  velo_reward: {velo_reward:.3f}") #rot_reward: {rot_reward:.3f}")
        

        return rewards, success
    
    
    #COMPLETE
    def stop_simulation(self):
        print("Done")
        self.gym.destroy_sim(self.sim)
