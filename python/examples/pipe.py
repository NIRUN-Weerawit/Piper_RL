"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


piper Attractor
----------------
Positional control of piper panda robot with a target attractor that the robot tries to reach
"""

import math
import numpy as np
import ast


from isaacgym import gymapi
from isaacgym import gymutil
import json

from isaacgym import gymtorch
import torch


# Initialize gym
gym = gymapi.acquire_gym()

# Parse arguments
args = gymutil.parse_arguments(description="Piper Example",
                               custom_parameters=[
                                   {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"}])
# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.solver_type = 5
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 15
    sim_params.flex.relaxation = 0.75
    sim_params.flex.warm_start = 0.8
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# Create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# Add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# Load piper asset
asset_root = "../../assets"
piper_asset_file = "urdf/piper_description/urdf/piper_description.urdf"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = True
asset_options.armature = 0.01

print("Loading asset '%s' from '%s'" % (piper_asset_file, asset_root))
piper_asset = gym.load_asset(
    sim, asset_root, piper_asset_file, asset_options)

# Set up the env grid
num_envs = args.num_envs
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# Some common handles for later use
envs = []
piper_handles = []
piper_hand = "link6"

# Attractor setup
attractor_handles = []
attractor_properties = gymapi.AttractorProperties()
attractor_properties.stiffness = 5e5
attractor_properties.damping = 5e3

# Make attractor in all axes
attractor_properties.axes = gymapi.AXIS_ALL
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0.0, 0.0)
pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

# Create helper geometry used for visualization
# Create an wireframe axis
axes_geom = gymutil.AxesGeometry(0.1)
# Create an wireframe sphere
sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
sphere_pose = gymapi.Transform(r=sphere_rot)
sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(1, 0, 0))

print("Creating %d environments" % num_envs)
num_per_row = int(math.sqrt(num_envs))

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)
    
    # add piper
    piper_handle = gym.create_actor(env, piper_asset, pose, "piper", i , 2)
    body_dict = gym.get_actor_rigid_body_dict(env, piper_handle)
    # print("body_dict:", body_dict)
    props = gym.get_actor_rigid_body_states(env, piper_handle, gymapi.STATE_POS)
    hand_handle = body = gym.find_actor_rigid_body_handle(env, piper_handle, piper_hand)



    # Draw axes and sphere at attractor location
    # gymutil.draw_lines(axes_geom, gym, viewer, env, attractor_properties.target)
    # gymutil.draw_lines(sphere_geom, gym, viewer, env, attractor_properties.target)

    piper_handles.append(piper_handle)
    # attractor_handle = gym.create_rigid_body_attractor(env, attractor_properties)
    # attractor_handles.append(attractor_handle)
# get joint limits and ranges for piper
# get joint limits and ranges for piper
piper_dof_props = gym.get_actor_dof_properties(envs[0], piper_handles[0])
piper_lower_limits = piper_dof_props['lower']
piper_upper_limits = piper_dof_props['upper']
piper_ranges = piper_upper_limits - piper_lower_limits
piper_mids = 0.5 * (piper_upper_limits + piper_lower_limits)
piper_num_dofs = len(piper_dof_props)

# override default stiffness and damping values
# piper_dof_props['stiffness'].fill(1000.0)
# piper_dof_props['damping'].fill(200.0)  #1000.0

# # # Give a desired pose for first 2 robot joints to improve stability
# Now focus just one robot 
piper_dof_props["driveMode"] = gymapi.DOF_MODE_EFFORT  #DOF_MODE_POS

# piper_dof_props["driveMode"][7:] = gymapi.DOF_MODE_POS
piper_dof_props['stiffness'] = 1000
piper_dof_props['damping'] = 200.0        #200.0
# print("after stiffness and damping are added", piper_dof_props)
for i in range(num_envs):
    gym.set_actor_dof_properties(envs[i], piper_handles[i], piper_dof_props)


# print("type of piper_dof_props", type(piper_dof_props))
print(piper_dof_props.dtype.names)
print(piper_dof_props)
print("piper_dof_props(lower)", piper_dof_props['lower'])
print("piper_dof_props(upper)", piper_dof_props['upper'])
# print("lower: %f " % piper_lower_limits[0])
# print("upper: %f " % piper_upper_limits[0])
print("piper_num_dofs: %d " % piper_num_dofs)
# piper_dof_states = []
    

global step 
global val
global counter
global random_action
step = 0 
counter = 0
random_action = [60.0] * 6
val = -20
# Moving animation
def update_piper(t):
    global step 
    global val
    global counter
    global random_action
    gym.clear_lines(viewer)
    step += 1

    for i in range(num_envs):
        # Update attractor target from current piper state
        # attractor_properties = gym.get_attractor_properties(envs[i], attractor_handles[i]

        # gym.set_attractor_target(envs[i], attractor_handles[i], pose)
        piper_dof_states = gym.get_actor_dof_states(envs[i], piper_handles[i], gymapi.STATE_POS)
        
        
        # print(f"len = {len(random_action)}")
        if step % 50 == 0:
            val = -val
        
        random_action[counter] += val
        print(f"random_action = {random_action}")
        if step % 5 == 0:
            counter +=1
        if counter == 6:
            counter = 0

        for i in range(num_envs):
           
            rl_actions      =   np.concatenate((random_action, [0.0, 0.0]))
            
            # action_np = rl_actions.astype(np.float32) * 1000.0 / 3 #100.0
            action_np       =   rl_actions.astype(np.float32)
            
            force_tensor =   torch.from_numpy(action_np).to("cpu")
            
            # print(f"force tensor = {force_tensor}")
            # self.gym.set_actor_dof_position_targets(self.envs[i], self.piper_handles[i], action_np)
            gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(force_tensor))
        # print("len(piper_dof_states): ", len(piper_dof_states))

# Point camera at environments
cam_pos = gymapi.Vec3(4, 3, 3)
cam_target = gymapi.Vec3(-4, -3, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# Time to wait in seconds before moving robot
next_piper_update_time = 1.0


while not gym.query_viewer_has_closed(viewer):
    
    # Every 0.01 seconds the pose of the attactor is updated
    t = gym.get_sim_time(sim)
    if t >= next_piper_update_time:
        update_piper(t)
        next_piper_update_time += 0.5

    # Step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Step rendering
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

print("Done")
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
