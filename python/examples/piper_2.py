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

from isaacgym import gymapi
from isaacgym import gymutil
import json
import torch
from paho.mqtt import client as mqtt_client 
from threading import Lock

broker = "sora2.uclab.jp"
port = 1883
# topic = "robot/79bf476b-936c-460e-9329-adc8dcfd61b6-vzkhpl2-viewer"
# topic = "robot/79bf476b-936c-460e-9329-adc8dcfd61b6-zlxz5rn-viewer"
# topic = "control/9fb1b055-400a-4de5-a527-e9c0f12709ef-8m63esn"
# topic = "robot/79bf476b-936c-460e-9329-adc8dcfd61b6-5fq6dyu-viewer"
# topic = "robot/9fb1b055-400a-4de5-a527-e9c0f12709ef-ziabl8y-viewer "
# topic = "control/36ec7d19-7a14-4f67-8284-d71e5909797a-y5igbua" #cobotta
client_id = 'PiPER-control-wee'

def connect_mqtt() -> mqtt_client:
    def on_connect(client, userdata, flags, rc, properties):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)
    client = mqtt_client.Client(client_id=client_id, callback_api_version=mqtt_client.CallbackAPIVersion.VERSION2)

    # client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


vr_joints = [0]*7
vr_joints_lock = Lock()
    
client = connect_mqtt()
def subscribe(client: mqtt_client):
    
    def on_message(client, userdata, msg):
        global vr_joints
        data = msg.payload.decode()
        data = json.loads(data)['joints']
        with vr_joints_lock:
            vr_joints[:] = data
        
        # print("type", type(vr_joints))
        # print("vr_joint", vr_joints)
        # print('len(vr):', len(vr_joints))
        # print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")

    client.subscribe(topic)
    client.on_message = on_message

# Initialize gym
gym = gymapi.acquire_gym()

# Parse arguments
args = gymutil.parse_arguments(description="Piper Example",
                               custom_parameters=[
                                   {"name": "--num_envs",   "type": int,    "default": 1,   "help": "Number of environments to create"},
                                   {"name": "--total_time", "type": float,  "default": 3.0, "help": "Time to reach the target pose"}])
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

# Create piper asset
asset_root = "../../assets"
piper_asset_file = "urdf/piper_description/urdf/piper_description.urdf"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = True
asset_options.armature = 0.01
print("Loading asset '%s' from '%s'" % (piper_asset_file, asset_root))
piper_asset = gym.load_asset(
    sim, asset_root, piper_asset_file, asset_options)

# load cube asset
cube_asset_file = "urdf/cube.urdf"
print("Loading asset '%s' from '%s'" % (cube_asset_file, asset_root))
cube_asset = gym.load_asset(sim, asset_root, cube_asset_file, gymapi.AssetOptions())


# Set up the env grid
num_envs = args.num_envs
total_time = args.total_time
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# Some common handles for later use
envs = []
piper_handles = []
piper_hand = "link6"
cube_handles = []

# Attractor setup
attractor_handles = []
attractor_initial_states = []
attractor_properties = gymapi.AttractorProperties()
attractor_properties.stiffness = 5e5
attractor_properties.damping = 5e3

# Make attractor in all axes
attractor_properties.axes = gymapi.AXIS_ALL
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0.0, 0.0)
pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

cube_pose = gymapi.Transform()
cube_pose.p = gymapi.Vec3(0.3, 0.5, 0.0)
cube_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

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
    piper_handle = gym.create_actor(env, piper_asset, pose, "piper", i , -1)
    cube_handle = gym.create_actor(env, cube_asset, cube_pose, "cube", i , -1)
    
    body_dict = gym.get_actor_rigid_body_dict(env, piper_handle)
    # print("body_dict:", body_dict)
    props = gym.get_actor_rigid_body_states(env, piper_handle, gymapi.STATE_POS)
    hand_handle = body = gym.find_actor_rigid_body_handle(env, piper_handle, piper_hand)
    
    # Initialize the attractor
    attractor_properties.target = props['pose'][:][body_dict[piper_hand]]
    attractor_properties.target.p.y -= 0.05
    attractor_properties.target.p.z = 0.03
    attractor_properties.target.p.x = 0.1 
    attractor_properties.rigid_handle = hand_handle

    # Draw axes and sphere at attractor location
    gymutil.draw_lines(axes_geom, gym, viewer, env, attractor_properties.target)
    gymutil.draw_lines(sphere_geom, gym, viewer, env, attractor_properties.target)

    piper_handles.append(piper_handle)
    cube_handles.append(cube_handle)
    # attractor_handle = gym.create_rigid_body_attractor(env, attractor_properties)
    # attractor_handles.append(attractor_handle)


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
piper_dof_props["driveMode"] = gymapi.DOF_MODE_POS  #DOF_MODE_POS

# piper_dof_props["driveMode"][7:] = gymapi.DOF_MODE_POS
piper_dof_props['stiffness'] = 1e10
piper_dof_props['damping'] = 10.0        #200.0
# print("after stiffness and damping are added", piper_dof_props)
for i in range(num_envs):
    gym.set_actor_dof_properties(envs[i], piper_handles[i], piper_dof_props)

    # Set piper pose so that each joint is in the middle of its actuation range
    piper_dof_states = gym.get_actor_dof_states(envs[i], piper_handles[i], gymapi.STATE_NONE)
    for j in range(piper_num_dofs):
        piper_dof_states['pos'][j] = piper_mids[j]
    gym.set_actor_dof_states(envs[i], piper_handles[i], piper_dof_states, gymapi.STATE_POS)
    
    # attractor_initial_state = gym.get_attractor_properties(envs[i], attractor_handles[i])
    # attractor_initial_states.append(attractor_initial_state)
    
# print("type of piper_dof_props", type(piper_dof_props))
print(piper_dof_props.dtype.names)
print(piper_dof_props)
print("piper_dof_props(lower)", piper_dof_props['lower'])
print("piper_dof_props(upper)", piper_dof_props['upper'])
# print("lower: %f " % piper_lower_limits[0])
# print("upper: %f " % piper_upper_limits[0])
print("piper_num_dofs: %d " % piper_num_dofs)
# piper_dof_states = []

# piper_dof_states = gym.get_actor_dof_states(envs[i], piper_handles[i], gymapi.STATE_NONE)
# print("piper_dof_states bf: ", piper_dof_states['pos'])
# states_bf = piper_dof_states['pos'].copy()
# for j in range(len(states_bf)):
    # states_bf[j] = (piper_upper_limits[j] - states_bf[j]) * 0.01 / 10 + states_bf[j]
# for i in range(num_envs):
    # gym.set_actor_dof_position_targets(envs[i], piper_handles[i], piper_upper_limits[i])
# Moving animation
def update_piper(t):
    
    gym.clear_lines(viewer)
    
    with vr_joints_lock:
        joints_copy = vr_joints.copy()
    for i in range(num_envs):
        # Update attractor target from current piper state
        # attractor_properties = gym.get_attractor_properties(envs[i], attractor_handles[i])
        # pose = attractor_properties.target
        # # gym.set_attractor_target(envs[i], attractor_handles[i], pose)
        # piper_dof_states = gym.get_actor_dof_states(envs[i], piper_handles[i], gymapi.STATE_NONE)
        cube_states = gym.get_actor_rigid_body_states(envs[i], cube_handles[i], gymapi.STATE_POS)
        piper_dof_states = gym.get_actor_dof_states(envs[i], piper_handles[i], gymapi.STATE_NONE)
        piper_body_states = gym.get_actor_rigid_body_states(envs[i], piper_handles[i], gymapi.STATE_ALL)
        print("size piper", len(piper_body_states))
        print("piper_body_states: ", piper_body_states['pose']['p'])
        print("piper_body_states: ", piper_body_states['pose']['p'][-1])
        pos = piper_body_states['pose']['p'][7]
        goal = [0.5, 0.0, 0.0]
        pos = [pos['x'], pos['y'], pos['z']]
        pos = np.array(pos)
        # pos = torch.from_numpy(pos)
        goal = np.array(goal)
        # print("type of pos", type(pos))
        # print("type of goal", type(goal))
        # diff = np.subtract(pos, goal)
        # diff_tensor = torch.from_numpy(diff)
        # nor = torch.norm(diff_tensor, p=2, dim=-1)
        # print(nor.item())
        # print("piper_dof_states bf: ", piper_dof_states['vel'])
        
        states_bf = piper_dof_states['pos'].copy()
        
        
        # print("piper_dof_states af: ", states_bf)
        # for j in range(len(states_bf)):
        #     states_bf[j] = (piper_upper_limits[j] - states_bf[j]) * 0.01 / 10 + states_bf[j]
        
        # Update attractor target from current piper state
        # attractor_properties = gym.get_attractor_properties(envs[i], attractor_handles[i])
        # pose = attractor_properties.target
        # print("attractor_pose: ", pose.p)
        # print("pose: ", pose.r.dtype.names)
        cube_pos = cube_states['pose'][0][0]
        cube_rot = cube_states['pose'][0][1]
        pose.p.x = (cube_pos[0] - pose.p.x) * 0.01 / total_time + pose.p.x 
        pose.p.y = (cube_pos[1] - pose.p.y) * 0.01 / total_time + pose.p.y 
        pose.p.z = (cube_pos[2] - pose.p.z) * 0.01 / total_time + pose.p.z 
        # pose.r   = cube_rot
        
        # piper_dof_states['pos'][0] = 2.618
        # piper_dof_states['pos'][1] = 0.218
        # piper_dof_states['pos'][2] = -2.11
        # piper_dof_states['pos'][3] = 1.014 
        # piper_dof_states['pos'][4] = 1.112 
        # piper_dof_states['pos'][5] = 2.465 
        # piper_dof_states['pos'][6] = 0.04 
        # piper_dof_states['pos'][7] = -0.04
        
        # states_bf[0] = (2.618 - states_bf[0]) * 0.01 / total_time + states_bf[0]
        # states_bf[1] = (0.281 - states_bf[1]) * 0.01 / total_time + states_bf[1]
        # states_bf[2] = (-2.11 - states_bf[2]) * 0.01 / total_time + states_bf[2]
        # states_bf[3] = (1.014 - states_bf[3]) * 0.01 / total_time + states_bf[3]
        # states_bf[4] = (1.112 - states_bf[4]) * 0.01 / total_time + states_bf[4]
        # states_bf[5] = (2.465 - states_bf[5]) * 0.01 / total_time + states_bf[5]
        # states_bf[6] = (0.04 - states_bf[6]) * 0.01 / total_time + states_bf[6]
        # states_bf[7] = (-0.04 - states_bf[7]) * 0.01 / total_time + states_bf[7]
        # print("piper_dof_states af: ", piper_dof_states['pos'])
        # gym.set_actor_dof_position_targets(envs[i], piper_handles[i], piper_dof_states['pos'])
        # gym.set_actor_dof_position_targets(envs[i], piper_handles[i], states_bf)
        # gym.set_attractor_target(envs[i], attractor_handles[i], pose)
        
        # gym.set_actor_dof_position_targets(envs[i], piper_handles[i], states_bf)
        # Draw axes and sphere at attractor location
        # gymutil.draw_lines(axes_geom, gym, viewer, envs[i], pose)
        # gymutil.draw_lines(sphere_geom, gym, viewer, envs[i], pose)
        
        
        # print("names:", cube_states.dtype.names)
        # print('cube: ', cube_states['pose'][0][0])
        
        # print("piper_dof_states bf: ", piper_dof_states['pos'])
        # states_bf = piper_dof_states['pos'].copy()
        # print("piper_dof_states bf: ", states_bf)
        # if states_bf[2] > 0:
        #     states_bf[2] = -0.5
        # elif states_bf[2] > -2.697:
        # states_bf[2] = states_bf[2] - 0.01
        # else:
            # states_bf[2] = states_bf[2]  
        # print("piper_dof_states af: ", states_bf)
        # piper_dof_states['pos'][2] = states_bf[2]
        # piper_dof_states['vel'][2] = 0
        # print("len(piper_dof_states): ", len(piper_dof_states))
        # print("len vr_joints:", len(vr_joints))
        # for n in range(0,6): 
            # 
        # print("vr_joints: ", vr_joints )
        
        # print("piper_dof_states af: ", piper_dof_states['pos'])
        # gym.set_actor_dof_states(envs[i], piper_handles[i], piper_dof_states, gymapi.STATE_POS)
        gym.set_actor_dof_position_targets(envs[i], piper_handles[i], piper_dof_states['pos'])
        # Draw axes and sphere at attractor location
        # gymutil.draw_lines(axes_geom, gym, viewer, envs[i], pose)
        # gymutil.draw_lines(sphere_geom, gym, viewer, envs[i], pose)
        # piper_dof_states = gym.get_actor_dof_states(envs[i], piper_handles[i], gymapi.STATE_NONE)
        # print(piper_dof_states.dtype.names)
        # print(piper_dof_states)


# for i in range(num_envs):
#     Set updated stiffness and damping properties
#     gym.set_actor_dof_properties(envs[i], piper_handles[i], piper_dof_props)

#     Set piper pose so that each joint is in the middle of its actuation range
#     piper_dof_states = gym.get_actor_dof_states(envs[i], piper_handles[i], gymapi.STATE_NONE)
#     for j in range(piper_num_dofs):
#         piper_dof_states['pos'][j] = piper_mids[j]
#     gym.set_actor_dof_states(envs[i], piper_handles[i], piper_dof_states, gymapi.STATE_POS)

# Point camera at environments
cam_pos = gymapi.Vec3(4, 3, 3)
cam_target = gymapi.Vec3(-4, -3, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# Time to wait in seconds before moving robot
next_piper_update_time = 1.0


# client.loop_forever()
# subscribe(client)
# client.loop_start()

while not gym.query_viewer_has_closed(viewer):
    # Update jacobian and mass matrix
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)
    # Every 0.01 seconds the pose of the attactor is updated
    t = gym.get_sim_time(sim)
    if t >= next_piper_update_time:
        update_piper(t)
        next_piper_update_time += 0.01
    
    # Step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Step rendering
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

print("Done")
# client.loop_stop() 
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
