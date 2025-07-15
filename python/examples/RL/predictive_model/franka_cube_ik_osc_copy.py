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


def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def cube_grasping_yaw(q, corners):
    """ returns horizontal rotation required to grasp cube """
    
    rc = quat_rotate(q, corners)
    yaw = (torch.atan2(rc[:, 1], rc[:, 0]) - 0.25 * math.pi) % (0.5 * math.pi)
    theta = 0.5 * yaw
    w = theta.cos()
    x = torch.zeros_like(w)
    y = torch.zeros_like(w)
    z = theta.sin()
    yaw_quats = torch.stack([x, y, z, w], dim=-1)
    return yaw_quats


def control_ik(dpose):
    global damping, j_eef, num_envs
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 6)
    return u


def control_osc(dpose):
    global kp, kd, kp_null, kd_null, default_dof_pos_tensor, mm, j_eef, num_envs, dof_pos, dof_vel, hand_vel
    mm_inv = torch.inverse(mm)
    m_eef_inv = j_eef @ mm_inv @ torch.transpose(j_eef, 1, 2)
    m_eef = torch.inverse(m_eef_inv)
    # print(hand_vel.shape, dpose.shape)
    u = torch.transpose(j_eef, 1, 2) @ m_eef @ (
        kp * dpose - kd * hand_vel.unsqueeze(-1))

    # Nullspace control torques `u_null` prevents large changes in joint configuration
    # They are added into the nullspace of OSC so that the end effector orientation remains constant
    # roboticsproceedings.org/rss07/p31.pdf
    j_eef_inv = m_eef @ j_eef @ mm_inv
    u_null = kd_null * -dof_vel + kp_null * (
        (default_dof_pos_tensor.view(1, -1, 1) - dof_pos + np.pi) % (2 * np.pi) - np.pi)
    u_null = u_null[:, :6]
    u_null = mm @ u_null
    u += (torch.eye(6, device=device).unsqueeze(0) - torch.transpose(j_eef, 1, 2) @ j_eef_inv) @ u_null
    return u.squeeze(-1)




torch.set_printoptions(precision=4, sci_mode=False)

# acquire gym interface
gym = gymapi.acquire_gym()

# parse arguments

# Add custom arguments
custom_parameters = [
    {"name": "--controller",    "type": str,    "default": "ik",    "help": "Controller to use for Franka. Options are {ik, osc}"},
    {"name": "--num_envs",      "type": int,    "default": 1,       "help": "Number of environments to create"},
    {"name": "--num_box",       "type": int,    "default": 1,       "help": "Number of boxes in the environment"},
    {"name": "--record",        "type": bool,   "default": True,    "help": "Whether to record or not"},
    {"name": "--seed",          "type": int,    "default": 10,       "help": "Set random seed for the simulation"},
    {"name": "--headless",      "type": bool,    "default": True,       "help": "Set random seed for the simulation"},
]
args = gymutil.parse_arguments(
    description="Piper Jacobian Inverse Kinematics (IK) + Operational Space Control (OSC) Example",
    custom_parameters=custom_parameters,
)

# set random seed
np.random.seed(args.seed)

# Grab controller
controller  = args.controller
assert controller in {"ik", "osc"}, f"Invalid controller specified -- options are (ik, osc). Got: {controller}"

# set torch device
device      = args.sim_device if args.use_gpu_pipeline else 'cpu'

# configure sim
sim_params                  = gymapi.SimParams()
sim_params.up_axis          = gymapi.UP_AXIS_Z
sim_params.gravity          = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt               = 1.0 / 60.0
sim_params.substeps         = 2
sim_params.use_gpu_pipeline = args.use_gpu_pipeline
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type                    = 1
    sim_params.physx.num_position_iterations        = 8
    sim_params.physx.num_velocity_iterations        = 1
    sim_params.physx.rest_offset                    = 0.0
    sim_params.physx.contact_offset                 = 0.001
    sim_params.physx.friction_offset_threshold      = 0.001
    sim_params.physx.friction_correlation_distance  = 0.0005
    sim_params.physx.num_threads                    = args.num_threads
    sim_params.physx.use_gpu                        = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

"""elif args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.solver_type = 5
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 15
    sim_params.flex.relaxation = 0.75
    sim_params.flex.warm_start = 0.8"""


    

# Set controller parameters
# IK params
damping = 0.05

# OSC params
kp      = 150.
kd      = 2.0 * np.sqrt(kp)
kp_null = 10.
kd_null = 2.0 * np.sqrt(kp_null)

# create sim
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# create viewer
if not args.headless:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise Exception("Failed to create viewer")

asset_root = "../../../../assets"

# create table asset
table_dims                          = gymapi.Vec3(0.6, 1.0, 0.01)
table_asset_options                 = gymapi.AssetOptions()
table_asset_options.fix_base_link   = True
table_asset_options.armature        = 0.01
table_asset                         = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, table_asset_options)
table_pose                          = gymapi.Transform()
table_pose.p                        = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)
print("Creating asset '%s' " % ("table"))

# create box asset
box_size                    = 0.04
box_dim                     = [box_size] * 3
box_asset_options           = gymapi.AssetOptions()
box_asset_options.density   = 1000
box_asset_options.armature  = 0.01
box_asset                   = gym.create_box(sim, box_size, box_size, box_size, box_asset_options)
box_pose                    = gymapi.Transform()
box_pose_np                 = np.array((0,0,0))
print("Creating asset '%s' " % ("box1"))

# create tray asset
tray_dim                            = [0.15, 0.15, 0.005] #small
# tray_dim = [0.2, 0.2, 0.01] #original
tray_color                          = gymapi.Vec3(0.24, 0.35, 0.8)
tray_asset_file                     = "urdf/tray/traybox_smaller.urdf"
tray_asset_options                  = gymapi.AssetOptions()
tray_asset_options.armature         = 0.01
tray_asset_options.density          = 8000
tray_asset_options.override_inertia = True
tray_asset                          = gym.load_asset(sim, asset_root, tray_asset_file, tray_asset_options)
corner                              = table_pose.p - table_dims * 0.5
x           = 0.3 #corner.x  # + table_dims.x * 0.2
y           = 0.1 #corner.y + table_dims.y * 0.8
z           = tray_dim[2]
tray_pose   = gymapi.Transform()
tray_pose.p = gymapi.Vec3(x, y, z)

print("Loading asset '%s' from '%s'" % (tray_asset_file, asset_root))


# Load piper asset
piper_asset_file                        = "urdf/piper_description/urdf/piper_description.urdf"
cube_asset_file                         = "urdf/cube.urdf"
asset_options                           = gymapi.AssetOptions()
asset_options.fix_base_link             = True
asset_options.flip_visual_attachments   = True
asset_options.armature                  = 0.01
# print("Loading asset '%s' from '%s'" % (piper_asset_file, asset_root))
piper_asset = gym.load_asset(sim, asset_root, piper_asset_file, asset_options)

# get joint limits and ranges for piper
piper_dof_props     = gym.get_asset_dof_properties(piper_asset)
piper_lower_limits  = piper_dof_props['lower']
piper_upper_limits  = piper_dof_props['upper']
piper_ranges        = piper_upper_limits - piper_lower_limits
piper_mids          = 0.5 * (piper_upper_limits + piper_lower_limits)
piper_num_dofs      = len(piper_dof_props)

'''
# load franka asset
franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
asset_options = gymapi.AssetOptions()
asset_options.armature = 0.01
asset_options.fix_base_link = True
asset_options.disable_gravity = True
asset_options.flip_visual_attachments = True
franka_asset = gym.load_asset(sim, asset_root, franka_asset_file, asset_options)

# configure franka dofs
franka_dof_props = gym.get_asset_dof_properties(franka_asset)
franka_lower_limits = franka_dof_props["lower"]
franka_upper_limits = franka_dof_props["upper"]
franka_ranges = franka_upper_limits - franka_lower_limits
franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)
'''
# use position drive for all dofs
if controller == "ik":
    piper_dof_props["driveMode"][:6].fill(gymapi.DOF_MODE_POS)
    piper_dof_props["stiffness"][:6].fill(400.0)
    piper_dof_props["damping"][:6].fill(40.0)
    
else: # osc
    piper_dof_props["driveMode"][:6].fill(gymapi.DOF_MODE_EFFORT)
    piper_dof_props["stiffness"][:6].fill(0.0)
    piper_dof_props["damping"][:6].fill(0.0)
    
# grippers
piper_dof_props["driveMode"][6:].fill(gymapi.DOF_MODE_POS)
piper_dof_props["stiffness"][6:].fill(800.0)
piper_dof_props["damping"][6:].fill(40.0)

# default dof states and position targets
piper_num_dofs          = gym.get_asset_dof_count(piper_asset)
default_dof_pos         = np.zeros(piper_num_dofs, dtype=np.float32)
default_dof_pos[:6]     = piper_mids[:6]
# grippers open
default_dof_pos[6:]     = piper_upper_limits[6:]

default_dof_state           = np.zeros(piper_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"]    = default_dof_pos

# send to torch
default_dof_pos_tensor  = to_torch(default_dof_pos, device=device)

# get link index of panda hand, which we will use as end effector
franka_link_dict        = gym.get_asset_rigid_body_dict(piper_asset)
franka_hand_index       = franka_link_dict["piper_hand"]
print("franka hand index", franka_hand_index)
# configure env grid
num_envs    = args.num_envs
num_box     = args.num_box
num_per_row = int(math.sqrt(num_envs))
spacing     = 1.0
env_lower   = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper   = gymapi.Vec3(spacing, spacing, spacing)
print("Creating %d environments" % num_envs)

piper_pose     = gymapi.Transform()
piper_pose.p   = gymapi.Vec3(0, 0, 0)

table_pose      = gymapi.Transform()
table_pose.p    = gymapi.Vec3(0.35, 0.0, 0.5 * table_dims.z )

box_pose        = gymapi.Transform()

camera_handles          = []
camera_props            = gymapi.CameraProperties()
camera_props.width      = 640
camera_props.height     = 480
camera_1_position       = gymapi.Vec3(0.4, 0.5, 0.5)
camera_1_target         = gymapi.Vec3(0, 0, 0)
camera_2_position       = gymapi.Vec3(0.4, - 0.5, 0.6)
camera_2_target         = gymapi.Vec3(0, 0, 0)


envs            = []
tray_handles    = []
piper_handles   = []
box_handles     = []
tray_idxs       = []
box_idxs        = []
hand_idxs       = []
init_pos_list   = []
init_rot_list   = []

# add ground plane
plane_params        = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)
# unfinished_color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
# finished_color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
def random_box_pose():
    box_pose.p.x = table_pose.p.x + np.random.uniform(-0.1, 0.1)
    box_pose.p.y = table_pose.p.y + np.random.uniform(-0.2, 0.2)
    box_pose.p.z = table_dims.z + 0.5 * box_size
    # box_pose_np     = np.array([box_pose.p.x,box_pose.p.y,box_pose.p.z])
    # init_box_pose = np.zeros(1, dtype=gymapi.RigidBodyState.dtype)
    # init_box_pose['pose']['p'][0] =(box_pose.p.x, box_pose.p.y, box_pose.p.z)
    box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
    return box_pose

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add table
    # table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)

    # add tray
    tray_handle     = gym.create_actor(env, tray_asset, tray_pose, "tray", i, 0)
    tray_handles.append(tray_handle)
    gym.set_rigid_body_color(env, tray_handles[i], 0, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)
    # get global index of tray in rigid body state tensor
    tray_idx        = gym.get_actor_rigid_body_index(env, tray_handle, 0, gymapi.DOMAIN_SIM)
    tray_idxs.append(tray_idx)
    
    # add box
    box_handles.append([])
    box_idxs.append([])
    
    for n in range(num_box):
        box_handle          = gym.create_actor(env, box_asset, random_box_pose(), "box_" + str(n), i, 0)
        unfinished_color    = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
        gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, unfinished_color) #color
        box_handles[i].append(box_handle)
        
        # get global index of box in rigid body state tensor
        box_idx = gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
        box_idxs[i].append(box_idx)
    
    

    # add piper
    piper_handle = gym.create_actor(env, piper_asset, piper_pose, "piper", i, 2)
    piper_handles.append(piper_handle)
    
    # set dof properties
    gym.set_actor_dof_properties(env, piper_handle, piper_dof_props)

    # set initial dof states
    gym.set_actor_dof_states(env, piper_handle, default_dof_state, gymapi.STATE_ALL)

    # set initial position targets
    gym.set_actor_dof_position_targets(env, piper_handle, default_dof_pos)

    # get inital hand pose
    hand_handle     = gym.find_actor_rigid_body_handle(env, piper_handle, "piper_hand")
    hand_pose       = gym.get_rigid_transform(env, hand_handle)
    init_pos_list.append([0.1, 0.1, 0.3])
    # init_pos_list.append([tray_pose.p.x, tray_pose.p.y, tray_pose.p.z])
    init_rot_list.append([-0.95, -0.25, 0.0, 0.0])
    # init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
    # init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])
    
    # get global index of hand in rigid body state tensor
    hand_idx        = gym.find_actor_rigid_body_index(env, piper_handle, "piper_hand", gymapi.DOMAIN_SIM)
    hand_idxs.append(hand_idx)
    
    # add camera
    cam_1 = gym.create_camera_sensor(env, camera_props)
    cam_2 = gym.create_camera_sensor(env, camera_props)
    #set the location of camera sensor
    gym.set_camera_location(cam_1, env, camera_1_position, camera_1_target)
    gym.set_camera_location(cam_2, env, camera_2_position, camera_2_target)
    camera_handles.append([])
    camera_handles[i].append(cam_1)
    camera_handles[i].append(cam_2)

unfinished_box_idxs = copy.deepcopy(box_idxs)
finished_box_idxs   = []
# tray_states = gym.get_actor_rigid_body_states(envs[0], tray_handles[0], gymapi.STATE_POS)
# box_states = gym.get_actor_rigid_body_states(envs[0], box_handles[0], gymapi.STATE_POS)
# init_tray_states = tray_states.copy()
# init_box_states = box_states.copy()

# init_tray_pose = np.zeros(1, dtype=gymapi.RigidBodyState.dtype) 

# init_tray_pose = (x, y, z)

# print("tray pose: ", tray_pose.p)

# print("init_pos_list:", hand_pose.p)
# point camera at middle env
cam_pos         = gymapi.Vec3(4, 3, 2)
cam_target      = gymapi.Vec3(-4, -3, 0)
middle_env      = envs[num_envs // 2 + num_per_row // 2]
if not args.headless: gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

# ==== prepare tensors =====
# from now on, we will use the tensor API that can run on CPU or GPU
gym.prepare_sim(sim)

# initial hand position and orientation tensors
init_pos = torch.Tensor(init_pos_list).view(num_envs, 3).to(device)
init_rot = torch.Tensor(init_rot_list).view(num_envs, 4).to(device)

# hand orientation for grasping
down_q = torch.stack(num_envs * [torch.tensor([1.0, 0.0, 0.0, 0.0])]).to(device).view((num_envs, 4))

# box corner coords, used to determine grasping yaw
box_half_size   = 0.5 * box_size
corner_coord    = torch.Tensor([box_half_size, box_half_size, box_half_size])
corners         = torch.stack(num_envs * [corner_coord]).to(device)

# downard axis
down_dir        = torch.Tensor([0, 0, -1]).to(device).view(1, 3)

# get jacobian tensor
# for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
_jacobian       = gym.acquire_jacobian_tensor(sim, "piper")
jacobian        = gymtorch.wrap_tensor(_jacobian)

# jacobian entries corresponding to franka hand
j_eef           = jacobian[:, franka_hand_index - 1, :, :6]

# get mass matrix tensor
_massmatrix     = gym.acquire_mass_matrix_tensor(sim, "piper")
mm              = gymtorch.wrap_tensor(_massmatrix)
mm              = mm[:, :6, :6]          # only need elements corresponding to the franka arm

def find_closest_box_idx(unfinished_box_idxs):
    # unfinished_box_idxs: list of lists, each sublist contains box indices for that env
    boxes_pose = []
    for idxs in unfinished_box_idxs:
        # idxs is a list of box indices for this env
        if len(idxs) == 0:
            # If no unfinished boxes, append a dummy row (will be ignored)
            boxes_pose.append(torch.full((1, 3), float('inf'), device=rb_states.device))
        else:
            boxes_pose.append(rb_states[idxs, :3])
    # boxes_pose: list of [num_unfinished, 3] tensors
    # Pad to max length for stacking
    max_boxes       = max(x.shape[0] for x in boxes_pose)
    padded_boxes    = []
    for x in boxes_pose:
        if x.shape[0] < max_boxes:
            pad     = torch.full((max_boxes - x.shape[0], 3), float('inf'), device=rb_states.device)
            x       = torch.cat([x, pad], dim=0)
        padded_boxes.append(x)
    boxes_pose      = torch.stack(padded_boxes)  # [num_envs, max_boxes, 3]
    norms           = torch.norm(boxes_pose, dim=2)
    min_indices     = torch.argmin(norms, dim=1)
    closest_box_idx = []
    for env_i, idx in enumerate(min_indices):
        if len(unfinished_box_idxs[env_i]) == 0:
            closest_box_idx.append(-1)  # or some invalid index
        else:
            closest_box_idx.append(unfinished_box_idxs[env_i][idx.item()])
    closest_box_idx = torch.tensor(closest_box_idx, device=rb_states.device)
    
    #find the first object to be picked up
    '''boxes_pose          = rb_states[unfinished_box_idxs, :3]
    norms               = torch.norm(boxes_pose, dim=2)
    min_indices         = torch.argmin(norms, dim=1, keepdim=True)
    print("min_indices: ", min_indices)
    closest_box_idx     = min_indices.unsqueeze(-1).expand(-1,-1,rb_states.shape[1])
    print("closest_box_idx: ", closest_box_idx)'''
    # closest_box_pose    = torch.gather(boxes_pose, dim=1, index=closest_box_idx)
    return closest_box_idx  #, min_indices

# get rigid body state tensor
_rb_states      = gym.acquire_rigid_body_state_tensor(sim)
rb_states       = gymtorch.wrap_tensor(_rb_states)
# print("rb_state shape", rb_states.shape)
# print("rb_state shape", rb_states.shape[0])
# print("rb_state shape", rb_states.shape[1])

rb_states_clone = rb_states.clone()
# print("device: ", rb_states_clone.device)
# print("rb_states_clone[tray_idxs, :] =", rb_states_clone[tray_idxs])
init_tray_rot   = rb_states_clone[tray_idxs, 3:7]
init_cube_rot   = rb_states_clone[box_idxs, 3:7]
print("box_ids:", box_idxs)
# print("box_ids:", len(box_idxs))
# print("init_cube_rot", init_cube_rot.shape)
# closest_box_idxs    = find_closest_box_idx(box_idxs)
# print("closest box_ids:", closest_box_idxs)
# print("closest box_ids:", len(closest_box_idxs))
find_new_idx        = torch.full((num_envs,), True, dtype=torch.bool, device=device)
closest_box_idxs    = find_closest_box_idx(unfinished_box_idxs)


# get dof state tensor
_dof_states     = gym.acquire_dof_state_tensor(sim)
dof_states      = gymtorch.wrap_tensor(_dof_states)  #torch.Size([num_envs * 8, 2]) pos & vel
dof_pos         = dof_states[:, 0].view(num_envs, 8, 1) #shape = [num_envs, 8, 1]
# dof_pos_buffer = torch.zeros_like(dof_pos, device=device).squeeze(2).unsqueeze(1)  # want shape = [num_envs, 1, 8]
# dof_pos_buffer = [] * num_envs # a list of tensors of size (num_envs)
dof_pos_buffer  = [torch.empty((0, 8), device=device) for _ in range(num_envs)] # a list of tensors of size (num_envs)

# print("dof_pos", dof_pos.shape)
# print("dof_pos buffer", dof_pos_buffer)
# print("dof_pos buffer", dof_pos_buffer.shape)
dof_vel         = dof_states[:, 1].view(num_envs, 8, 1)
init_dof_states = dof_states.clone().view(num_envs, 8, 2)

# Create a tensor noting whether the hand should return to the initial position
hand_restart    = torch.full([num_envs], False, dtype=torch.bool).to(device)

# Set action tensors
pos_action      = torch.zeros_like(dof_pos).squeeze(-1)
pos_action_buffer = [torch.empty((0, 8), device=device) for _ in range(num_envs)]
effort_action   = torch.zeros_like(pos_action)
t = 0
return_pos      = [0.1, 0.1, 0.3]
return_rot      = [-0.95, -0.25, 0.0, 0.0]
# prev_hand_vel = torch.zeros_like(rb_states[hand_idxs, 7:]).to(device)
prev_hand_pos   = torch.zeros_like(rb_states[hand_idxs, :3]).to(device)
# Create a tensor noting whether the env should restart
envs_restart    = torch.full([num_envs], False, dtype=torch.bool).to(device)
froze_count     = torch.zeros(num_envs).to(device)
frozen_threshold        = 0.1
default_dof_state_np    = np.zeros(piper_num_dofs)
# Convert to tensors
tray_dim_tensor = torch.tensor(tray_dim, device=device)
box_dim_tensor = torch.tensor(box_dim, device=device)
r_min = 0.2
r_max = 0.4
theta = 70
# Add these before the simulation loop
frozen_counter  = 0
success_counter = 0
success_count_each_env = torch.zeros(num_envs, device=device)
# success_count_each_env =  [0.]* num_envs
time_envs = torch.full((num_envs,), 0, dtype=int, device=device)


def one_item_one_tray():
    pass

def random_pos():
    random_tray_states          = init_tray_states.copy()
    random_cube_states          = init_box_states.copy()
    r_1     = np.random.uniform(0.2, 0.35)
    r_2     = np.random.uniform(0.15, 0.35)
    theta_1 = np.random.uniform(- 100/180 * np.pi, 100/180 * np.pi)
    theta_2 = np.random.uniform(- 100/180 * np.pi, 100/180 * np.pi)
    
    random_tray_states[0][0][0]   = (r_1 * np.cos(theta_1), r_1 * np.sin(theta_1),  table_dims.z)
    random_cube_states[0][0][0]   = (r_2 * np.cos(theta_2), r_2 * np.sin(theta_2),  table_dims.z + 0.5 * box_size)
    
    return random_tray_states, random_cube_states

def random_pos_tensor():
    """
        Return: torch.cat((tray_pose, boxes_pose), dim=1)
    """
    radius_tensor   = torch.FloatTensor(num_envs, num_box + 1).uniform_(r_min, r_max).to(device)       # num_envs x num_objs (3)
    theta_tensor    = torch.FloatTensor(num_envs, num_box + 1).uniform_(- theta/180 * np.pi, theta/180 * np.pi).to(device)   
    x_tensor   = radius_tensor * torch.cos(theta_tensor)           # num_envs x 1
    y_tensor   = radius_tensor * torch.sin(theta_tensor)           # num_envs x 1
    z_tensor_tray   = torch.full([num_envs], table_dims.z).to(device)                   # num_envs x 1
    z_tensor_cube   = torch.full([num_envs], table_dims.z + 0.5 * box_size).to(device)   # num_envs x 1
    
    # random_pose     = torch.stack((torch.stack([x_tensor[:, 0], y_tensor[:, 0], z_tensor_tray], dim=1), 
    #                                torch.stack([x_tensor[:, 1], y_tensor[:, 1], z_tensor_cube], dim=1)), dim=1).to(device)  
    tray_pose    = torch.stack([x_tensor[:, 0], y_tensor[:, 0], z_tensor_tray], dim=1).unsqueeze(1)
    boxes_pose  = torch.stack([torch.stack([x_tensor[:, j+1], y_tensor[:, j+1], z_tensor_cube], dim=1) for j in range(num_box)], dim=1)
    
    # print("tray_pose dim", tray_pose.shape)
    # print("boxes_pose dim", boxes_pose.shape)
    # print("tray_pose dim", tray_pose)
    # print("boxes_pose dim", boxes_pose)
    random_pose     = torch.cat((tray_pose, boxes_pose), dim=1)
    
    # print("random pose: ", random_pose)
    # print("random pose shape: ", random_pose.shape)
    
    return random_pose
    
def reset(restart_envs, dof_state):
    # print("Some environments will be restarted:", torch.where(restart_envs)[0])
    restart_indices = torch.where(restart_envs)[0].to(rb_states.device)
    # print("restart_indices", restart_indices)
    # print("restart_indices", restart_indices.shape)
    states_buffer                   = dof_state.clone().view(num_envs, 8, 2)
    states_buffer[restart_indices]  = init_dof_states[restart_indices]
    dof_state                       = states_buffer.view(num_envs * 8,2)
    random_pose                     = random_pos_tensor()
    # print("rb_state" ,rb_states)
    # print("box_idxs", box_idxs)
    
    # print("box_idxs", box_idxs.shape)
    # Get the tray and box indices for the environments to reset
    tray_reset_idxs                 = torch.tensor(tray_idxs, device=rb_states.device)[restart_indices]
    box_reset_idxs                  = torch.tensor(box_idxs, device=rb_states.device)[restart_indices]
    
    # Get the tray and box indices for the environments to reset
    # tray_reset_idxs = torch.tensor([tray_idxs[i] for i in restart_indices.tolist()], device=rb_states.device)
    # Flatten all box indices for the environments to reset
    # box_reset_idxs = torch.tensor([idx for i in restart_indices.tolist() for idx in box_idxs[i]], device=rb_states.device)
    
    
    # print("box_reset_idxs", box_reset_idxs)
    # box_reset_idxs                  = torch.tensor(box_idxs, device=rb_states.device)[restart_indices]
    # print("box_reset_idxs", box_reset_idxs)
    # print("rb_State" ,rb_states.shape)
    
    # print("rb_states[box_reset_idxs, :3] ", rb_states[box_reset_idxs, :3])
    # print("rb_states[box_reset_idxs, :3] shape ", rb_states[box_reset_idxs, :3].shape)
    # print("random_pose[restart_indices, 1:num_box + 1, :3]", random_pose[restart_indices, 1:num_box + 1, :3])
    # print("random_pose[restart_indices, 1:num_box + 1, :3] shape", random_pose[restart_indices, 1:num_box + 1, :3].reshape(-1, 3).shape)
    rb_states[tray_reset_idxs, :3]  = random_pose[restart_indices, 0, :3]
    rb_states[tray_reset_idxs, 3:7] = torch.tensor([0., 0., 0., 1.0], device=device)
    # for n in range(num_box):
    # rb_states[box_reset_idxs, :3]   = random_pose[restart_indices, 1:num_box + 1, :3]
    rb_states[box_reset_idxs, :3]   = random_pose[restart_indices, 1:num_box + 1, :3] #.reshape(-1, 3)
    
    # print("rb_states[box_reset_idxs, :3] ", rb_states[box_reset_idxs, :3])
    
    # print("rb_state shape", rb_states.shape)
    # print("rb_state", rb_states[tray_reset_idxs, 3:7])
    # print("inint_tray", init_tray_rot.shape)
    # print("inint_tray", init_tray_rot[restart_indices])
    '''print("rb_states[box_reset_idxs, 3:7] ", rb_states[box_reset_idxs, 3:7].shape)
    rb_states[tray_reset_idxs, 3:7] = init_tray_rot[0]
    
    rb_states[box_reset_idxs, 3:7]  = init_cube_rot[0]
    print("rb_states[box_reset_idxs, 3:7] ", rb_states[box_reset_idxs, 3:7])'''
    
    # Set orientations
    """rb_states[tray_reset_idxs, 3:7] = init_tray_rot[restart_indices]
    # Flatten the new box orientations
    new_box_orientations = torch.cat([init_cube_rot[i] for i in restart_indices.tolist()], dim=0)
    rb_states[box_reset_idxs, 3:7] = new_box_orientations
    # Zero velocities
    rb_states[tray_reset_idxs, 7:] = 0
    rb_states[box_reset_idxs, 7:] = 0 """
    '''for ind in restart_indices:
        tray_states, cube_states = random_pos()
        print("indices", ind.item())
        i = ind.item()
        gym.set_actor_rigid_body_states(envs[i], tray_handles[i], tray_states, gymapi.STATE_POS)
        gym.set_actor_rigid_body_states(envs[i], box_handles[i], cube_states, gymapi.STATE_POS)'''
    # print("rb_State" ,rb_states.shape)
    # print("rb_State" ,rb_states)
    gym.set_rigid_body_state_tensor(sim, gymtorch.unwrap_tensor(rb_states))
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state))
    
    hand_restart[:] = False


def store_joints_states(env: int, ep: int, joints_data: object, actions: object, success: bool):
    """file = h5py.File("dataset/joint_angles.h5py", 'w')
    group = file.create_group(f"env_{env}_ep_{ep}")
    dataset = file.store_joints_states()"""
    if success:
        assert joints_data.shape == actions.shape, f"Data mismatch! qpos={joints_data.shape}, actions={actions.shape}, "
        with open(f"dataset/success_seed_{args.seed}/env_{env}/states/env_{env}_ep_{ep}.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerows(joints_data.numpy())
        with open(f"dataset/success_seed_{args.seed}/env_{env}/actions/env_{env}_ep_{ep}.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerows(actions.numpy())    
    else:
        with open(f"dataset/failure_seed_{args.seed}/env_{env}/states/env_{env}_ep_{ep}.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerows(joints_data.numpy())
        with open(f"dataset/failure_seed_{args.seed}/env_{env}/actions/env_{env}_ep_{ep}.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerows(actions.numpy())

def create_video_writer(env): 
    """
        create a new pair of video writers in an envronment
        Return: color_writer_1, color_writer_2 
    """
    color_writer_1  = cv2.VideoWriter(f'color_1_env_{env}.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))
    color_writer_2  = cv2.VideoWriter(f'color_2_env_{env}.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))
    return color_writer_1, color_writer_2 
# depth_writer_1    = cv2.VideoWriter('depth_1.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))
# depth_writer_2    = cv2.VideoWriter('depth_2.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))


def image_processing(writers):
    for i in range(num_envs):
        color_image_1   = gym.get_camera_image(sim, envs[i], camera_handles[i][0], gymapi.IMAGE_COLOR)
        color_image_2   = gym.get_camera_image(sim, envs[i], camera_handles[i][1], gymapi.IMAGE_COLOR)
        # depth_image_1   = gym.get_camera_image(sim, envs[0], camera_handles[0][0], gymapi.IMAGE_DEPTH)
        # depth_image_2   = gym.get_camera_image(sim, envs[0], camera_handles[0][1], gymapi.IMAGE_DEPTH)
        # print("depth_image_shape",depth_image_1)
        
        img_np_1 = color_image_1.reshape((camera_props.height, camera_props.width, 4))
        img_np_2 = color_image_2.reshape((camera_props.height, camera_props.width, 4))
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

writers = []
Eps_success     = [0] * num_envs
Eps_failure     = [0] * num_envs

try:
    for env in range(num_envs):
        os.makedirs(f"dataset/success_seed_{args.seed}/env_{env}/states" ,  exist_ok=True)
        os.makedirs(f"dataset/success_seed_{args.seed}/env_{env}/actions" , exist_ok=True)
        os.makedirs(f"dataset/failure_seed_{args.seed}/env_{env}/states" ,  exist_ok=True)
        os.makedirs(f"dataset/failure_seed_{args.seed}/env_{env}/actions" , exist_ok=True)
        os.makedirs(f"videos/success_seed_{args.seed}/env_{env}" , exist_ok=True)
        os.makedirs(f"videos/failure_seed_{args.seed}/env_{env}" , exist_ok=True)
except FileExistsError:
    pass
success_dataset_dir = (os.getenv('ISAACGYM_BASE_PATH') + '/python/examples/RL/predictive_model/dataset/success')
failure_dataset_dir = (os.getenv('ISAACGYM_BASE_PATH') + '/python/examples/RL/predictive_model/dataset/failure')
# sessions = [d for d in os.listdir(success_dataset_dir) if os.path.isdir(os.path.join(success_dataset_dir, d)) and d.startswith('env_')]

for j in range(num_envs):
    color_writer_1, color_writer_2 = create_video_writer(j)
    writers.append([color_writer_1, color_writer_2])


# simulation loop
# while not gym.query_viewer_has_closed(viewer):
while True:
    """
    # randomize the init position
    t += 1
    if t % 200 ==0:
        init_pos_list = []
        # init_rot_list = []
        # return_pos = [np.random.uniform(0.1, 0.15), np.random.uniform(-0.1, 0.1), 0.3]
        for _ in range(num_envs):
            return_pos = [np.random.uniform(0.1, 0.15), np.random.uniform(-0.1, 0.1), 0.3]
            init_pos_list.append(return_pos)
            # init_rot_list.append(return_rot)
        init_pos = torch.Tensor(init_pos_list).view(num_envs, 3).to(device)
        # init_rot = torch.Tensor(init_rot_list).view(num_envs, 4).to(device)
    """
    time_envs[:] += 1
    t+=1
        
    # print("box_idxs", box_idxs)
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # refresh tensors
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)
    
    # print("unfinished idxs", unfinished_box_idxs)
    
    if torch.any(find_new_idx):
        new_box_idxs = find_closest_box_idx(unfinished_box_idxs)
        closest_box_idxs = torch.where(find_new_idx, new_box_idxs, closest_box_idxs)
        
        find_new_idx[:] = False
        
    # closest_box_idxs = find_closest_box_idx(unfinished_box_idxs)
        
    # print(f"current box idxs for each environment are {closest_box_idxs} from total unfinished box idxs {unfinished_box_idxs}")
    # print("min_indices", min_indices)

    # print("closest_box_idxs", closest_box_idxs)
    # print("closest_box_idxs shape", closest_box_idxs.shape)
    # print("rb_states[box_idxs, :]",rb_states[box_idxs, :])
    
    '''boxes_states = torch.gather(rb_states[box_idxs, :], dim=1, index=closest_box_idxs)
    print("box_states", boxes_states)
    print("box_states shape", boxes_states.shape)
    boxes_states = boxes_states.squeeze(1)'''
    
    # dof_pos_buffer = torch.cat((dof_pos_buffer, dof_pos.squeeze(2).unsqueeze(1)), dim=1)
    # dof_pos_buffer = [dof_pos[k].squeeze(2).unsqueeze(1) for k in range(num_envs)]
    
    # print(dof_pos[0].squeeze(1).unsqueeze(0))
    # dof_pos_buffer.append([dof_pos.squeeze(2).unsqueeze(1)])
    
    # print(dof_pos_buffer)
    # print("buffer len =", len(dof_pos_buffer))
    # print("buffer type =", type(dof_pos_buffer))
    # print("buffer shape =", dof_pos_buffer[0].shape)
    boxes_states = rb_states[closest_box_idxs, :]
    
    # box_rot = rb_states[unfinished_box_idxs, 3:7]
    # box_vel = rb_states[unfinished_box_idxs, 7:]
    box_pos     =   boxes_states[:, :3]
    box_rot     =   boxes_states[:, 3:7]
    box_vel     =   boxes_states[:, 7:]
    # print("box_pos:", box_pos)
    # print("box_pos:", box_pos.shape)
    # print("box_rot:", box_rot.shape)
    # print("corners", corners.shape)
    # print("box_vel:", box_vel.shape)
    tray_pos   = rb_states[tray_idxs, :3]
    tray_rot   = rb_states[tray_idxs, 3:7]
    # print("tray_pos:", tray_pos)
    # print("tray_pos:", tray_pos.shape)
    hand_pos = rb_states[hand_idxs, :3]
    print(f"hand pos: {hand_pos}")
    hand_rot = rb_states[hand_idxs, 3:7]
    # print("hand_rot:", hand_pos)
    hand_vel = rb_states[hand_idxs, 7:]
    # print("hand vel", hand_vel)
    # print("diff: ", torch.abs(hand_pos - prev_hand_pos))
    
    to_box = box_pos - hand_pos
    box_dist = torch.norm(to_box, dim=-1).unsqueeze(-1)
    # print("box dist: ", box_dist)
    box_dir = to_box / box_dist
    box_dot = box_dir @ down_dir.view(3, 1)

    # how far the hand should be from box for grasping
    grasp_offset = 0.12 if controller == "ik" else 0.12

    # determine if we're holding the box (grippers are closed and box is near)
    gripper_sep = dof_pos[:, 6] + dof_pos[:, 7]
    
    gripped = (gripper_sep < box_size) & (box_dist < grasp_offset + 0.5 * box_size) #should change to grabbing
    
    
    # if gripped: print("gripper status:", gripped)
    yaw_q = cube_grasping_yaw(box_rot, corners)
    box_yaw_dir = quat_axis(yaw_q, 0)
    hand_yaw_dir = quat_axis(hand_rot, 0)
    yaw_dot = torch.bmm(box_yaw_dir.view(num_envs, 1, 3), hand_yaw_dir.view(num_envs, 3, 1)).squeeze(-1)
    
    # fix the init position to be the tray's position
    init_pos = tray_pos.clone()
    init_pos[:, 2] = 0.25
    
    # determine if we have reached the initial position; if so allow the hand to start moving to the box
    to_init = init_pos - hand_pos
    init_dist = torch.norm(to_init, dim=-1)
    hand_restart = (hand_restart & (init_dist > 0.02)).squeeze(-1)  #hand_restart = True when the hand returns to its init pos
    return_to_start = (hand_restart | gripped.squeeze(-1)).unsqueeze(-1)

    # if hand is above box, descend to grasp offset
    # otherwise, seek a position above the box
    above_box = ((box_dot >= 0.99) & (yaw_dot >= 0.95) & (box_dist < grasp_offset * 3.0)).squeeze(-1)
    # if above_box: print("above box status", above_box)
    grasp_pos = box_pos.clone()
    grasp_pos[:, 2] = torch.where(above_box, box_pos[:, 2] + grasp_offset, box_pos[:, 2] + grasp_offset * 2.5)
    
    
    
    # compute goal position and orientation / Determine whether to go back to the initial pose or continue to approaching the object
    goal_pos = torch.where(return_to_start, init_pos, grasp_pos) #This is where to set the destination for the hand
    # print("return to start", return_to_start)
    goal_rot = torch.where(return_to_start, init_rot, quat_mul(down_q, quat_conjugate(yaw_q)))
    # print("hand pose:", hand_pos)
    # print("goal pose:", goal_pos)
    
    # compute position and orientation error
    pos_err = goal_pos - hand_pos
    orn_err = orientation_error(goal_rot, hand_rot)
    dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
    # print("dpose:", dpose)
    # Deploy control based on type of calculation
    if controller == "ik":
        pos_action[:, :6] = dof_pos.squeeze(-1)[:, :6] + control_ik(dpose)
    else:       # osc  
        effort_action[:, :6] = control_osc(dpose)
    
    # gripper actions depend on distance between hand and box, if the distance of the object is within threshold -> close the gripper
    close_gripper = (box_dist < grasp_offset + 0.03) | gripped 
    # if close_gripper: print("close gripper", close_gripper)
    
    # always open the gripper above a certain height, dropping the box and restarting from the beginning
    is_bounded_by_tray = (torch.abs(box_pos[:, :2] - tray_pos[:, :2]) < (tray_dim_tensor[:2]  -  box_dim_tensor[:2]) / 2)
    hand_restart = hand_restart | (is_bounded_by_tray.all(dim=1) & (box_pos[:, 2] > 0.08)) #| (box_pos[:, 2] > 0.17) #After grabbing the object, the hand will keep moving to its init pos until the object reaches a certain height, then it will drop.   
    # hand_restart = hand_restart | (box_pos[:, 2] > 0.17)
    keep_going = torch.logical_not(hand_restart)
    close_gripper = close_gripper & keep_going.unsqueeze(-1)
    grip_acts = torch.where(close_gripper, torch.Tensor([[0.00, 0.00]] * num_envs).to(device), torch.Tensor([[0.04, -0.04]] * num_envs).to(device))
    pos_action[:, 6:8] = grip_acts
    
    
    # Deploy actions
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))
    gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(effort_action))
    
    
    '''place_dist = torch.norm(box_pos - tray_pos)
    threshold = torch.norm(torch.tensor(tray_dim) - torch.tensor(box_dim))
    print("threshold: ", 0.5 * threshold)
    print("place_dist: ", place_dist)
    is_in_tray = place_dist < 0.5 * threshold'''
    is_frozen = torch.all(torch.abs(hand_pos - prev_hand_pos) < 0.01, dim=1)
    not_frozen = torch.logical_not(is_frozen)
    # is_frozen = torch.all(torch.abs(hand_vel) < frozen_threshold, dim=1)  # shape: (num_envs,)
    froze_count[is_frozen] += 1
    froze_count[not_frozen] = 0
    # froze_count[~is_frozen] = 0 
    frozen_envs = (froze_count > 100) | (time_envs > 1000)
    
    # print("froze_count:", froze_count[0])
    prev_hand_pos = hand_pos
    
    # Compute half extents
    # tray_half = tray_dim_tensor[:2] / 2  # Only x, y
    # box_half =  box_dim_tensor[:2]/ 2
    
    # print("is_bounded_by_tray", is_bounded_by_tray.shape)
    is_inside_tray = (torch.abs(box_pos[:, 2] - tray_pos[:, 2])   < 0.1).unsqueeze(1)
    # print("is_inside_tray", is_inside_tray.shape)
    is_in_tray = torch.cat([is_bounded_by_tray, is_inside_tray], dim=1).all(dim=1)
    # print("isintray shape", is_in_tray.shape)
    
    # Check if box center is within tray bounds (with some margin)
    # is_in_tray = (is_bounded_by_tray & is_inside_tray).all(dim=0)
    # if torch.any(is_in_tray): print("is_in_tray", is_in_tray)
    # print("norm vel", torch.norm(box_vel))
    is_in_tray = (is_in_tray & (torch.norm(box_vel[:, :3]) < 0.15))
    
    num_success = is_in_tray.sum().item()
    num_frozen = (froze_count > 100).sum().item()
    if num_success > 0:
        success_counter += num_success
    if num_frozen > 0:
        frozen_counter += num_frozen
        
    # env_success = is_in_tray.all(dim=1)
    
    if torch.any(is_in_tray): 
        # print("SUCCCESS")
        # print("unfinished_box_idxs before", unfinished_box_idxs)
        # print("isintray", is_in_tray)
        success_idx = torch.where(is_in_tray)[0]
        # print("success_idx", success_idx)

        success_count_each_env[is_in_tray] += 1 
    
        to_be_removed = closest_box_idxs[success_idx] #remove the current closest box from successful environments
        
        
        # unfinished_box_idxs[is_in_tray].remove(to_be_removed)
        for env_i, box_idx in zip(success_idx.tolist(), to_be_removed.tolist()):
            # Remove the box index from the unfinished list for this environment
            if box_idx in unfinished_box_idxs[env_i]:
                unfinished_box_idxs[env_i].remove(box_idx)
                # gym.set_rigid_body_color(envs[env_i], box_handles[env_i][box_idxs[env_i].index(box_idx)], 0, gymapi.MESH_VISUAL_AND_COLLISION, finished_color) #color
                # print(f"The box id {box_idx} in the environment {env_i} is put in the box")
                # print(f"There is/are {len(unfinished_box_idxs[env_i])} left in the environment {env_i}")
        
        find_new_idx[success_idx] = True #to reset the closest box idxs
        # print("unfinished_box_idxs after", unfinished_box_idxs)
        
    # print("frozen_envs", frozen_envs.shape)
    # finished_envs = [i for i, count in enumerate(success_count_each_env) if count >= num_box]
    # print("success count each env", success_count_each_env)
    finished_envs = (success_count_each_env == num_box)
    # print("finished_envs", finished_envs)
    restart_envs = frozen_envs | finished_envs # | (time_envs > 1000)
    
    dof_pos_buffer      = [torch.cat((dof_pos_buffer[k],    dof_pos[k].squeeze(1).unsqueeze(0)),    dim=0) for k in range(num_envs)]
    pos_action_buffer   = [torch.cat((pos_action_buffer[k], pos_action[k].unsqueeze(0)),            dim=0) for k in range(num_envs)]
    
    # update viewer
    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)
    gym.start_access_image_tensors(sim)
    
    if args.record: image_processing(writers)
    # reset the environment if it stops moving for some time.
    if torch.any(restart_envs):
        if torch.any(finished_envs):
            # success_eps = torch.where(is_in_tray)[0]
            
            # i is the index of finished environment
            for i, finished in enumerate(finished_envs): 
                if finished & (len(unfinished_box_idxs[i]) == 0):
                    print(f"Saving successful Ep {Eps_success[i]} of environment {i}, success tracker = {success_count_each_env}")
                    # print(f"There is/are {len(unfinished_box_idxs[i])} left in the environment {i}")
                    shutil.move(f"color_1_env_{i}.avi", f"videos/success_seed_{args.seed}/env_{i}/color_1_env_{i}_ep_{Eps_success[i]}.avi")
                    shutil.move(f"color_2_env_{i}.avi", f"videos/success_seed_{args.seed}/env_{i}/color_2_env_{i}_ep_{Eps_success[i]}.avi")
                    store_joints_states(i, Eps_success[i], dof_pos_buffer[i], pos_action_buffer[i], True)
                    dof_pos_buffer[i]       = torch.empty((0, 8))
                    pos_action_buffer[i]    = torch.empty((0, 8))
                    Eps_success[i] += 1
                    writers[i] = create_video_writer(i)
                    
        if  torch.any(frozen_envs) & (len(unfinished_box_idxs[env_i]) < num_box - 1):
            for i, frozen in enumerate(frozen_envs): 
                if frozen:
                    print(f"Saving failed Ep {Eps_failure[i]} of environment {i}, success tracker = {success_count_each_env}")
                    print(f"There is/are {len(unfinished_box_idxs[i])} left in the environment {i}")
                    shutil.move(f"color_1_env_{i}.avi", f"videos/failure_seed_{args.seed}/env_{i}/color_1_env_{i}_ep_{Eps_failure[i]}.avi")
                    shutil.move(f"color_2_env_{i}.avi", f"videos/failure_seed_{args.seed}/env_{i}/color_2_env_{i}_ep_{Eps_failure[i]}.avi")
                    store_joints_states(i, Eps_failure[i], dof_pos_buffer[i], pos_action_buffer[i], False)
                    dof_pos_buffer[i]       = torch.empty((0, 8))
                    pos_action_buffer[i]    = torch.empty((0, 8))
                    Eps_failure[i] += 1
                    writers[i] = create_video_writer(i)
        elif torch.any(frozen_envs):
            for i, frozen in enumerate(frozen_envs): 
                if frozen:    
                    print(f"discarding an episode of environment {i}")
                    print(f"There is/are {len(unfinished_box_idxs[i])} left in the environment {i}")
                    os.remove(f"color_1_env_{i}.avi")
                    os.remove(f"color_2_env_{i}.avi")
                    dof_pos_buffer[i]       = torch.empty((0, 8))
                    pos_action_buffer[i]    = torch.empty((0, 8))
                    writers[i] = create_video_writer(i)   
             
            
        reset(restart_envs, dof_states)
        froze_count[restart_envs] = 0  
        # change_sign = torch.logical_not(frozen_envs)
        frozen_envs = torch.full_like(frozen_envs, False).to(device)    
        success_count_each_env[restart_envs] = 0 

        for i, do_reset in enumerate(restart_envs):
            if do_reset:
                time_envs[i] = 0
                unfinished_box_idxs[i] = copy.deepcopy(box_idxs[i])
                # color
                '''for n in range(num_box):
                    gym.set_rigid_body_color(envs[i], box_handles[env_i][n], 0, gymapi.MESH_VISUAL_AND_COLLISION, unfinished_color)'''
        
        
    '''# update viewer
    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)
    gym.start_access_image_tensors(sim)
    
    if args.record: image_processing(writers)
    if torch.any(restart_envs):
        if torch.any(finished_envs):
            # success_eps = torch.where(is_in_tray)[0]
            
            # i is the index of finished environment
            for i, finished in enumerate(finished_envs): 
                if finished:
                    shutil.move(f"color_1_env_{i}_ep_{Eps[i]}.avi", f"success/color_1_env_{i}_ep_{Eps[i]}.avi")
                    shutil.move(f"color_2_env_{i}_ep_{Eps[i]}.avi", f"success/color_2_env_{i}_ep_{Eps[i]}.avi")
                    Eps[i] += 1
                    writers[i] = create_video_writer(i,Eps[i])
        if  torch.any(frozen_envs):
            for i, finished in enumerate(frozen_envs): 
                if finished:
                    shutil.move(f"color_1_env_{i}_ep_{Eps[i]}.avi", f"failure/color_1_env_{i}_ep_{Eps[i]}.avi")
                    shutil.move(f"color_2_env_{i}_ep_{Eps[i]}.avi", f"failure/color_2_env_{i}_ep_{Eps[i]}.avi")
                    Eps[i] += 1
                    writers[i] = create_video_writer(i,Eps[i])'''
        
    
    if not args.headless:
        gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)
    gym.end_access_image_tensors(sim)
    
    """if t % 500 == 0:
        print(f"Total frozen resets: {frozen_counter}")
        print(f"Total successes: {success_counter}")"""
    

print(f"Total frozen resets: {frozen_counter}")
print(f"Total successes: {success_counter}")
# cleanup
color_writer_1.release()
color_writer_2.release()
if not args.headless:
    gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
cv2.destroyAllWindows()
