"""
Create piper environment
----------------
Create an isaacgym environment to train piper models for performing tasks 
"""

import math
import numpy as np
import ast
import sys
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgymenvs.tasks.base.vec_task import VecTask
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


class Gym_env(VecTask):
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
        self.headless           = args['headless']
        self.dt                 = args['dt']
        self.debug_interval     = args['debug_interval']
        self.warmup             = args['warmup']
        
        self.env                = None
        self.gym                = None     #self.gym = gymapi.acquire_gym()
        self.sim                = None     #self.sim = self.gym.create_sim(args['compute_device_id'], args['graphics_device_id'], args['physics_engine'], sim_params)
        
        self.viewer             = None
        self.envs               = []
        
        self.piper_dof_states   = []
        self.piper_body_states  = []
        self.piper_handles      = []
        self.piper_hand         = "link6"
        
        self.cube_states        = []
        self.cube_handles       = []
        
        self.states_bf          = []
        
        self.cube_pose          = [0.3, 0.2, 0.0]        # Target position
        self.goal_rot           = [0.0, 0.0, 0.0, -1.0]   # Target orientation (quaternion)
        self.sphere_geom        = None
        self.asset_root         = "/home/ucluser/isaacgym/assets"
        
        self.piper_lower_limits = []
        self.piper_upper_limits = []
        self.piper_mids         = []
        self.piper_num_dofs     = 8
        
        self.writer             = SummaryWriter('logs', comment="")
        
        self.time_counter       = 0
        self.time_ep            = 0
        super().__init__(config                 = self.args, 
                         device_type            ='cuda', 
                         numEnvs                = self.num_envs,
                         numObservations        = 1,
                         numActions             = 6
                        #  sim_device             =self.sim_device, 
                        #  graphics_device_id     =self.graphics_device_id, 
                        #  headless               =self.headless, 
                         )
        
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
        piper_description_chain = Chain.from_urdf_file(self.asset_root + "/" + piper_asset_file)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.armature = 0.01
        print("Loading asset '%s' from '%s'" % (piper_asset_file, self.asset_root))
        piper_asset = self.gym.load_asset(self.sim, self.asset_root, piper_asset_file, asset_options)
        return piper_asset, piper_description_chain

    def load_cube(self):
        # load cube asset
        cube_asset_file = "urdf/cube.urdf"
        print("Loading asset '%s' from '%s'" % (cube_asset_file, self.asset_root))
        cube_asset = self.gym.load_asset(self.sim, self.asset_root, cube_asset_file, gymapi.AssetOptions())
        return cube_asset


    def create_piper_env(self):
        
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
        piper_asset, piper_chain = self.load_piper()
        cube_asset  = self.load_cube()
        
        print("Creating %d environments" % self.num_envs)
        num_per_row = int(math.sqrt(self.num_envs))

        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)
            
            # add pipers and cubes into the gym simulation 
            piper_handle = self.gym.create_actor(env, piper_asset, piper_pose, "piper", i , -1)
            cube_handle = self.gym.create_actor(env, cube_asset, cube_pose, "cube", i , -1)
            
            self.gym.enable_actor_dof_force_sensors(env, piper_handle)
            
            body_dict = self.gym.get_actor_rigid_body_dict(env, piper_handle)
            
            gymutil.draw_lines(self.sphere_geom, self.gym, self.viewer, self.envs[i], goal_pose)

            self.piper_handles.append(piper_handle)
            self.cube_handles.append(cube_handle)

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
        piper_dof_props["driveMode"][:] =  gymapi.DOF_MODE_EFFORT  #gymapi.DOF_MODE_POS  #A DoF that is set to a specific drive mode will ignore drive commands for other modes.

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
        
        print("Issacgym piper simulation is completed")
        
        # self.random_new_goal()

        
    def step(self, rl_actions):
        # for _ in range(self.num_envs): #TODO recheck this for-loop again
        self.time_counter += 1
        self.time_ep += 1
        
        
        # advance the simulation in the simulator by one step
        self.refresh()
                
        #take actions given by RL agent
        # self.apply_rl_actions(rl_actions)
        self.apply_rl_actions_force(rl_actions)
        
        # store new observations in the vehicles and traffic lights class
        self.update()
        
        #----TODO: DURING WARMING UP, LET THE AGENT LEARN FROM MOVING ROBOTIC ARM USING POSITION    

        # crash encodes whether the simulator experienced a collision
        # crash = self.check_collision()

        # stop collecting new simulation steps if there is a collision
        # if crash:
            # break

        # render a frame
        self.render()

        states = self.get_states()
        #TODO: print out the structure of the states
        
        # random new goal
        # if self.time_counter % 20 == 0:
            # self.random_new_goal()
        
        # collect information of the state of the network based on the
        # environment class used
        # self.state = np.asarray(states).flatten
        
        # if self.debug:
            # print("stored obs:", self.state)
        # collect observation new state associated with action
        next_observation = np.copy(states)

        # compute the reward
        # rl_clipped = clip_actions(rl_actions)   #TODO: Make sure whether clip_actions() function is necessary
        reward, done = self.compute_reward(next_observation, rl_actions)  #TODO: fix this when crash is availalbe fail=crash 

        return next_observation, reward, done # infos
    
    def wait(self):
        self.render()
    
            
    def random_new_goal(self):
        
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
        
        self.cube_pose = [x, y, z]

        

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

        goal_position               = self.cube_pose #list()   #TODO: change this when the goal_pos refers to the real cube
        goal_orientation            = self.goal_rot  #list()   #TODO: change this when the goal_rot refers to the real cube
        # joint_angles                = self.piper_dof_states['vel']['linear'][-3] #list() * 8
        
        # I forgot that the End-effector position is the last tuple from self.piper_body_states['pose']['p']
        # this is my computing reward function, the problem is that the size of state is enormously big, can you fix state[a:b} that I used to be what I want? because I cannot actually keep track of the position in the state vector. Do you know which location is the position (x,y,z) of the link8 (last link of the
        end_effector_position       = self.piper_body_states['pose']['p'][-3]
        end_effector_velocity       = self.piper_body_states['vel']['linear'][-3]
        # print("size EE_pose", len(ee_p_dicts))
        # print(f"body length: {len(self.piper_body_states['pose']['p'])}")
        # print("EE_pos:", ee_p_dicts[-1])
        # ee_position_x = [p['x'] for p in ee_p_dicts]
        # ee_position_y = [p['y'] for p in ee_p_dicts]
        # ee_position_z = [p['z'] for p in ee_p_dicts]
        # end_effector_position = ee_position_x + ee_position_y + ee_position_z
        
        # print("EE_position_bf", end_effector_position)
        end_effector_position       = [end_effector_position['x'],
                                       end_effector_position['y'],
                                       end_effector_position['z']]
        end_effector_velocity       = [end_effector_velocity['x'],
                                       end_effector_velocity['y'],
                                       end_effector_velocity['z']]
        # end_effector_position = [list(pos) for pos in end_effector_position] 
        # print("EE_position_af", end_effector_position)
        
        # end_effector_orientation    = self.piper_body_states['pose']['r'] #dict
        end_effector_orientation = self.piper_body_states['pose']['r'][-3]      # list of 9 dicts
        # print("size EE_rot", len(ee_r_dicts))
        
        # print("EE_pos:", ee_r_dicts[-1])
        # ee_rot_x = [r['x'] for r in ee_r_dicts]
        # ee_rot_y = [r['y'] for r in ee_r_dicts]
        # ee_rot_z = [r['z'] for r in ee_r_dicts]
        # ee_rot_w = [r['w'] for r in ee_r_dicts]
        # end_effector_orientation = ee_rot_x + ee_rot_y + ee_rot_z + ee_rot_w
        # print("EE_orientation_bf", end_effector_orientation)
        end_effector_orientation    = [end_effector_orientation['x'],
                                       end_effector_orientation['y'],
                                       end_effector_orientation['z'],
                                       end_effector_orientation['w']]
        # end_effector_orientation = [list(rot) for rot in end_effector_orientation]  
        # print("EE_orientation_af", end_effector_orientation)
        
        
        # obs = [goal_position, goal_orientation, end_effector_position, end_effector_orientation]
        # obs = np.array(goal_position +  goal_orientation +  end_effector_position + end_effector_orientation)
        obs = np.array(goal_position + goal_orientation + end_effector_position + end_effector_orientation + end_effector_velocity, dtype=np.float32)
        # obs.flatten()
        # if self.debug:
            # print("OBS:", obs)
            # print("size:", len(obs))
        return obs
        # return self.cube_states, self.piper_dof_states, self.piper_body_states
        
        
    #COMPLETE  
    def update(self):
        for i in range(self.num_envs):
            self.cube_states        = self.gym.get_actor_rigid_body_states(self.envs[i],    self.cube_handles[i],   gymapi.STATE_POS) 
            self.piper_dof_states   = self.gym.get_actor_dof_states(self.envs[i],           self.piper_handles[i],  gymapi.STATE_ALL)  
            #gymapi.DofState  ([('pos', '<f4'), ('vel', '<f4')]):   piper_dof_state['pos'][i]
            self.piper_body_states  = self.gym.get_actor_rigid_body_states( self.envs[i],   self.piper_handles[i],  gymapi.STATE_ALL)  
            
            self.piper_forces = self.gym.get_actor_dof_forces(self.envs[i], self.piper_handles[i])
            # print(f"forces: {self.piper_forces}")
            
    #COMPLETE    
    def refresh(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        # self.gym.refresh_jacobian_tensors(self.sim)
        # self.gym.refresh_mass_matrix_tensors(self.sim)
        
    #COMPLETE
    def render(self):
        # Step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # Step rendering
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, False)
        self.gym.sync_frame_time(self.sim)

    
    def reset(self):
        self.gym.clear_lines(self.viewer)
        for i in range(self.num_envs):
            # Set piper pose so that each joint is in the middle of its actuation range
            piper_dof_states = self.gym.get_actor_dof_states(self.envs[i], self.piper_handles[i], gymapi.STATE_POS)
            for j in range(self.piper_num_dofs):
                piper_dof_states['pos'][j] = self.piper_mids[j]
            self.gym.set_actor_dof_states(self.envs[i], self.piper_handles[i], piper_dof_states, gymapi.STATE_POS)
            # self.gym.set_actor_dof_position_targets(self.envs[i], self.piper_handles[i], piper_dof_states['pos'])
        
        self.time_ep     = 0   
        self.random_new_goal() 
        t_now   = self.gym.get_sim_time(self.sim)
        t_0     = self.gym.get_sim_time(self.sim)
        while True:
            if t_now - t_0  >= 1:
                break
            self.render()
            t_now   = self.gym.get_sim_time(self.sim)
              
    def apply_rl_actions_force(self, rl_actions=None):

        # ignore if no actions are issued
        if rl_actions is None:
            return

        for i in range(self.num_envs):
            # action_np = rl_clipped[i].detach().cpu().numpy().astype(np.float32)
            # print(f"rl_actions= {rl_actions}, type= {type(rl_actions)}")
            rl_actions       =   np.concatenate((rl_actions, [0.0, 0.0]))
            
            # action_np = rl_actions.astype(np.float32) * 1000.0 / 3 #100.0
            action_np    =   rl_actions.astype(np.float32) * 150
            
            # action_np[1] =   abs(action_np[1])
            # action_np[1] = action_np[1] * 1.5
            # action_np[2] = action_np[2] * 1.5
            # action_np[-1] = 0.0
            # action_np[-2] = 0.0
            # action_np[6] =   abs(action_np[6])
            # action_np[7] = - abs(action_np[7])
            # print("action = ", action_np)
            force_tensor =   torch.from_numpy(action_np).to("cpu")
            
            # print(f"force tensor = {force_tensor}")
            # self.gym.set_actor_dof_position_targets(self.envs[i], self.piper_handles[i], action_np)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(force_tensor))
    
    #COMPLETE
    def compute_reward(self, states, rl_actions=None, **kwargs):
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
        
        #Current end-effector's position
        #TODO: right now, it is assumed that the end-effector is the link8, but in the future, it should be in the middle of the gripper
        # [goal_position 0-2, goal_orientation 3-6, end_effector_position 7-9, end_effector_orientation 10-13]
        
        current_EE_pose     = state[7:10]
        # current_EE_pose = state[2]['pose']['p'][7]  #dict
        # current_EE_pose = [current_EE_pose['x'], 
        #                    current_EE_pose['y'], 
        #                    current_EE_pose['z']]  #list
        current_EE_pose_tensor = torch.tensor(current_EE_pose)
        
        #piper_body_states: state[2]['pose'/'vel']['p'/'r'][link: 0,1,2,3,4,5,6,7][x,y,z: 0,1,2]
        
        # distance from EE to the goal
        dist = torch.norm(current_EE_pose_tensor - goal_pose_tensor, p=2, dim=-1)
        
        # dist_reward = 1.0 / (1.0 + dist ** 2)
        # dist_reward = - 0.5 * (dist ** 2)
        dist_reward = dist
        # dist_reward *= dist_reward
        dist_reward = torch.where(dist <= 0.2, 1 - dist, - dist_reward)
        
        #Current end-effector's rotations
        current_EE_rot = state[10:14]
        current_EE_rot_tensor = torch.tensor(current_EE_rot)
        # current_EE_rot = state[2]['pose']['r'][7]
        # current_EE_rot_tensor = torch.tensor([current_EE_rot['x'],
        #                                       current_EE_rot['y'],
        #                                       current_EE_rot['z'],
        #                                       current_EE_rot['w']])
        
        # Normalize the quaternions 
        current_EE_rot_tensor = F.normalize(current_EE_rot_tensor, dim=0)
        goal_rot_tensor = F.normalize(goal_rot_tensor, dim=0)
        
        dot_product = torch.sum(current_EE_rot_tensor * goal_rot_tensor)
        dot_product = torch.abs(dot_product)
        rot_reward = 0.5 * (dot_product ** 2)
        rot_reward = torch.where(dot_product > 0.8, rot_reward * 2, rot_reward)
        
        #----Detach from GPU and move to cpu and extract only one value out of the tensors
        rot_reward = rot_reward.detach().cpu().item()
        dist_reward = dist_reward.detach().cpu().item()
        rewards = self.dist_reward_scale * dist_reward + self.rot_reward_scale * rot_reward - 1  #- math.sqrt(self.time_counter)
        self.writer.add_scalar('Reward per step', rewards, self.time_ep)
        
        done = (dist <= 0.15)
        
        if done:
            print("DONE DIST = ", dist.item())
            rewards += 800
            
            
        if self.debug and self.time_counter % self.debug_interval == 0:
            print(f"step: {self.time_counter},  dist= {dist:.3f}")
            print(f"rewards: {rewards:3f}  dist_reward: {dist_reward:.3f} rot_reward: {rot_reward:.3f}")
        

        return rewards, done
    
    #COMPLETE
    def stop_simulation(self):
        print("Done")
        # client.loop_stop() 
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

