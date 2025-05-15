"""Contains an experiment class for running simulations."""

from RL_Utils.create_env import Gym_env
import datetime
import logging
import time
import os
import numpy as np
import json


class Experiment:

    def __init__(self, args, custom_callables=None):
        """Instantiate the Experiment class.

        Parameters
        ----------
        args : dict
            ... parameters
        custom_callables : dict < str, lambda >
            strings and lambda functions corresponding to some information we
            want to extract from the environment. The lambda will be called at
            each step to extract information from the env and it will be stored
            in a dict keyed by the str.
        """
        self.hidden_size = args['hidden_size']
        
        self.custom_callables = custom_callables or {}

        # Get the env name and a creator for the environment.
        self.gym_instance = Gym_env(args)
        self.gym_instance.create_piper_env()
        
        # self.gym_instance.dist_reward_scale = args['dist_reward_scale']
        # self.gym_instance.rot_reward_scale = args['rot_reward_scale']
        self.sim = self.gym_instance.sim 
        
        # self.action_min = self.gym_instance.piper_lower_limits
        # self.action_max = self.gym_instance.piper_upper_limits
        
        self.action_min = [-1] * 6
        self.action_max = [1] * 6
        
        
        #TODO: define  self.env

        # Create the environment.
        # self.env = create_env()

        logging.info(" Starting experiment at {}".format(str(datetime.datetime.now())))

        logging.info("Initializing environment.")

    def run(self, num_envs, training, testing, Graph, debug_training, debug_testing, warmup, n_episodes, max_episode_len, warmup_step, run):

        import torch
        import torch.nn
        from GRL_Library.common             import replay_buffer
        from RL_Library                     import TD3_agent
        from RL_Utils.Train_and_Test_DDPG   import Training_GRLModels, Testing_GRLModels

        # Initialize GRL models
        # num_envs = num_envs
        N = 33   #gripper_pose : A list of end-effector transformation including position and orientation: [(x, y, z), (x, y, z, w)]
                #SHOULD BE GOAL_POSE INIT?   [goal_position, goal_orientation, end_effector_position, end_effector_orientation, end_effector_velocity]
        A = 6   #joints_angle : A list of joints' angle: [j1, j2, j3, j4, j5, j6, j7, j8]
        
        # action_min = torch.tensor(action_min, device=)
        # action_max = torch.tensor(action_max)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert isinstance(Graph, bool)
        if Graph:
            from RL_Net.Model_Continuous.TD3_net import OUActionNoise, Graph_Actor_Model, Graph_Critic_Model
            actor    = Graph_Actor_Model(N, A, self.action_min, self.action_max, self.hidden_size)        #.to(device)
            critic_1 = Graph_Critic_Model(N, A, self.action_min, self.action_max, self.hidden_size)       #.to(device)
            critic_2 = Graph_Critic_Model(N, A, self.action_min, self.action_max, self.hidden_size)       #.to(device)
        else:
            from RL_Net.Model_Continuous.TD3_net import OUActionNoise, NonGraph_Actor_Model, NonGraph_Critic_Model
            actor    = NonGraph_Actor_Model(N, A, self.action_min, self.action_max, self.hidden_size)     #.to(device)
            critic_1 = NonGraph_Critic_Model(N, A, self.action_min, self.action_max, self.hidden_size)        #.to(device)
            critic_2 = NonGraph_Critic_Model(N, A, self.action_min, self.action_max, self.hidden_size)        #.to(device)

        # Initialize optimizer
        lr_critic   = 0.0005
        lr_actor    = 0.0002
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr_actor)  # 需要定义学习率
        critic_optimizer_1 = torch.optim.Adam(critic_1.parameters(), lr=lr_critic)  # 需要定义学习率
        critic_optimizer_2 = torch.optim.Adam(critic_2.parameters(), lr=lr_critic)  # 需要定义学习率
        # Noisy
        explore_noise = 0.3
        # Replay_buffer
        replay_buffer = replay_buffer.ReplayBuffer(size=10 ** 6)
        
        # Discount factor
        gamma = 0.99

        
        # Initialize GRL agent
        GRL_TD3 = TD3_agent.TD3(
            actor,
            actor_optimizer,
            critic_1,
            critic_optimizer_1,
            critic_2,
            critic_optimizer_2,
            explore_noise,  # noisy
            warmup,  # warmup
            replay_buffer,  # replay buffer
            batch_size=128,  # batch_size
            update_interval=50,  # model update interval (< actor model) 100
            update_interval_actor=100,  # actor model update interval 500
            target_update_interval=400,  # target model update interval 5000
            soft_update_tau=0.001,  # soft update factor
            n_steps=1,  # multi-steps
            gamma=gamma,  # discount factor
            model_name="TD3_model",  # model name]
            action_size = A
        )

        # Training
        
        save_dir = '/home/ucluser/isaacgym/python/examples/RL/RL_TrainedModels/TD3/' + str(run)
        print(f"save_dir= {save_dir}")
        # debug_training = True
        if training:
            Training_GRLModels(actor, GRL_TD3, n_episodes, max_episode_len, save_dir, debug_training, self.gym_instance, warmup, warmup_step)
        
        # Testing
        test_episodes = 10
        load_dir = '/home/ucluser/isaacgym/python/examples/RL/RL_TrainedModels/TD3'
        # debug_testing = False
        if testing:
            Testing_GRLModels(actor, GRL_TD3, test_episodes, max_episode_len, load_dir, debug_training, self.gym_instance)
