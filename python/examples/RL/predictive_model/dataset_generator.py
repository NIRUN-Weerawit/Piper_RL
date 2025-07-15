
import math
import numpy as np
import torch
import random
import time
import copy
import os
import shutil
import cv2
import csv
import h5py

# def store_joints_states(env: int, ep: int, joints_data: object, success: bool):
#     file = h5py.File("dataset/joint_angles.h5py", 'w')
#     group = file.create_group(f"env_{env}_ep_{ep}")
#     dataset = file.store_joints_states()
#     if success:
#         with open(f"dataset/success_seed_{args.seed}/env_{env}_ep_{ep}.csv", 'w') as f:
#             writer = csv.writer(f)
#             writer.writerows(joints_data.numpy())
#     else:
#         with open(f"dataset/failure_seed_{args.seed}/env_{env}_ep_{ep}.csv", 'w') as f:
#             writer = csv.writer(f)
#             writer.writerows(joints_data.numpy())

# seed = args.seed
# joints_obs_dir = "dataset"
# success_joints_obs_dir     = os.path.join(joints_obs_dir, f"success_seed_{seed}")
# failure_joints_obs_dir     = os.path.join(joints_obs_dir, f"failure_seed_{seed}")
# videos_dir      = "video"
# success_videos_dir     = os.path.join(videos_dir, f"success_seed_{seed}")
# failure_videos_dir     = os.path.join(videos_dir, f"failure_seed_{seed}")

# TODO: make a function to generate images from a video
def video_to_frames(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        # Save frame as image
        # frame_filename = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
        
        frames.append(frame)
        
        # cv2.imwrite(frame_filename, frame)
        
        # print(f"Saved {frame_filename}")
        # frame_count += 1
    cap.release()
    # print(f"Done. Total frames saved: {frame_count}")
    return frames
    


def retrieve_data(ep, episode_joints_dir, episode_actions_dir, episode_cam1_dir, episode_cam2_dir):
    # Read joint positions
    action_data = []
    qpos_data   = []
    with open(episode_joints_dir, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:  
                qpos_data.append([float(val) for val in row])
    with open(episode_actions_dir, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:  
                action_data.append([float(val) for val in row])
                
    assert len(qpos_data) == len(action_data), f"Data mismatch! qpos={len(qpos_data)}, actions={len(action_data)}"
    print(f"Read {len(qpos_data)} joint positions and {len(action_data)} actions data!")

    # Convert video to frames
    cam1_frames = video_to_frames(episode_cam1_dir)  # [T, H, W, C]
    cam2_frames = video_to_frames(episode_cam2_dir)

    # Sanity checks
    assert len(cam1_frames) == len(cam2_frames), \
        f"Frame count mismatch: cam1={len(cam1_frames)}, cam2={len(cam2_frames)}"
    assert len(cam1_frames) == len(action_data), \
        f"Frame count and joint data length mismatch: frames={len(cam1_frames)}, joints={len(action_data)}"

    # Package outputs
    # qpos_data = action_data
    qvel_data = []         # Placeholder if not available
    sim       = True       # Metadata
    compress  = False      # Metadata

    return [ep, action_data, qpos_data, qvel_data, cam1_frames, cam2_frames, sim, compress]


def create_dataset(storage_dir, ep, action_data , qpos_data, qvel_data, cam1_data, cam2_data, sim=True, compress=False):
    with h5py.File(os.path.join(storage_dir, f"episode_{ep}.hdf5"), 'w') as f:
        f.create_dataset('action', data=action_data) # if action is joint angles, then dim=[k x 6], else if action is goal pose, then dim = [k x 7] (3+4)
        
        obs_grp     = f.create_group('observations')
        
        obs_grp.create_dataset('qpos', data=qpos_data) # [k x 6]
        obs_grp.create_dataset('qvel', data=qvel_data) # [k x 6]
        
        img_grp     = f.create_group('images')
        img_grp.create_dataset('cam1', data=cam1_data) # [k x H x W x C]
        img_grp.create_dataset('cam2', data=cam2_data) # [k x H x W x C]
        
        # Optional attributes
        f.attrs['sim'] = sim
        f.attrs['compress'] = compress
    

if __name__ == '__main__':
    joint_obs_dir = 'dataset'
    i = 0
    seed    = 11
    env     = 0
    ep      = 0
    num_envs    =  3
    success = True
    # storage_dir = f"datasets/seed_{seed}"
    storage_dir = f"/media/ucluser/Extreme SSD/datasets_1/seed_{seed}"
    os.makedirs(storage_dir, exist_ok=True)
    
    joints_obs_dir  = "dataset"
    videos_dir      = "videos"
    # joints_obs_dir  = "/media/ucluser/Extreme SSD/dataset"
    # videos_dir      = "/media/ucluser/Extreme SSD/videos"
    suffix = f"success_seed_{seed}" if success else f"failure_seed_{seed}"
    joints_obs_dir = os.path.join(joints_obs_dir, suffix)
    videos_dir     = os.path.join(videos_dir, suffix)
    assert joints_obs_dir
    assert videos_dir
    
    
    
    # videos_obs_sessions = [d for d in os.listdir(videos_dir)  ]
    # assert len(joints_obs_sessions) == len(videos_obs_sessions), \
        # f"Number of episodes mismatch: joints_data={len(joints_obs_sessions)}, videos_data={len(videos_obs_sessions)}"
    for env in range(num_envs):
        # episode_joints_dir = os.path.join(joints_obs_dir, f"env_{env}_ep_{ep}.csv")
        joints_obs_dirr = os.path.join(joints_obs_dir, f"env_{env}")
        videos_dirr     = os.path.join(videos_dir, f"env_{env}")
        states_dir      = os.path.join(joints_obs_dir, f"env_{env}/states")
        joints_obs_sessions = [d for d in os.listdir(states_dir)]
        print(f"size of env_{env}: {len(joints_obs_sessions)}")
        for j in range(len(joints_obs_sessions) - 1):
            # Construct file paths
            episode_joints_dir  = os.path.join(joints_obs_dirr, f"states/env_{env}_ep_{j}.csv")
            episode_actions_dir = os.path.join(joints_obs_dirr, f"actions/env_{env}_ep_{j}.csv")
            episode_cam1_dir    = os.path.join(videos_dirr, f"color_1_env_{env}_ep_{j}.avi")
            episode_cam2_dir    = os.path.join(videos_dirr, f"color_2_env_{env}_ep_{j}.avi")
            
            # episode_joints_dir = os.path.join(joints_obs_dir, joints_obs_sessions[j])
            # episode_cam1_dir   = os.path.join(videos_dir, )
            # episode_cam2_dir   = os.path.join(videos_dir,)
            
            # Ensure files exist
            assert os.path.exists(episode_joints_dir), f"Missing joint file: {episode_joints_dir}"
            assert os.path.exists(episode_cam1_dir),   f"Missing camera 1 file: {episode_cam1_dir}"
            assert os.path.exists(episode_cam2_dir),   f"Missing camera 2 file: {episode_cam2_dir}"
            
            create_dataset(storage_dir, *retrieve_data(i, episode_joints_dir, episode_actions_dir, episode_cam1_dir, episode_cam2_dir))
            i += 1
    