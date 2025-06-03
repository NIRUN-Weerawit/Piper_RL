"""Contains an experiment class for running simulations."""

from RL_Utils.create_env import Gym_env
import datetime
import logging
import time
import os
import numpy as np
import json
from RL_Utils.storagemanager import StorageManager 

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
        self.hidden_size            = args['hidden_size']
        self.N                      = args['state_size']
        self.A                      = args['action_size']
        # self.action_scale           = args['action_scale']
        self.warmup                 = args['warmup']
        self.lr_critic              = args['lr_critic']
        self.lr_actor               = args['lr_actor']
        self.explore_noise          = args['explore_noise']
        self.noise_clip             = args['noise_clip']
        self.gamma                  = args['gamma']
        self.batch_size             = args['batch_size']
        self.update_interval        = args['update_interval']
        self.update_interval_actor  = args['update_interval_actor']
        self.target_update_interval = args['target_update_interval']
        self.soft_update_tau        = args['soft_update_tau']
        self.n_steps                = args['n_steps']
        self.test_episodes          = args['test_episodes']
        self.n_episodes             = args['n_episodes']
        self.max_episode_len        = args['max_episode_len']
        self.server                 = args['server']
        self.action_min             = [-1] * 6
        self.action_max             = [1] * 6 
        
        self.custom_callables = custom_callables or {}
        
      

        # Get the env name and a creator for the environment.
        self.gym_instance = Gym_env(args)
        self.gym_instance.create_piper_env()
        
        # self.gym_instance.dist_reward_scale = args['dist_reward_scale']
        # self.gym_instance.rot_reward_scale = args['rot_reward_scale']
        self.sim = self.gym_instance.sim 
        
        # self.action_min = self.gym_instance.piper_lower_limits
        # self.action_max = self.gym_instance.piper_upper_limits

        self.sm = StorageManager("TD3", "", 0, 'cuda:0')
        self.sm.new_session_dir()
        self.sm.store_config(args)
        
        logging.info(" Starting experiment at {}".format(str(datetime.datetime.now())))

        logging.info("Initializing environment.")

    def run(self, num_envs, training, testing, Graph, debug_training, debug_testing):
        import torch
        import torch.nn
        from GRL_Library.common             import replay_buffer
        from RL_Library                     import TD3_agent
        from RL_Utils.Train_and_Test_TD3   import Training_GRLModels, Testing_GRLModels

        # Initialize GRL models
        # num_envs = num_envs
        N = self.N   #gripper_pose : A list of end-effector transformation including position and orientation: [(x, y, z), (x, y, z, w)]
                #SHOULD BE GOAL_POSE INIT?   [goal_position, goal_orientation, end_effector_position, end_effector_orientation, end_effector_velocity]
        A = self.A   #joints_angle : A list of joints' angle: [j1, j2, j3, j4, j5, j6, j7, j8]
        
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


        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=self.lr_actor)  # 
        critic_optimizer_1 = torch.optim.Adam(critic_1.parameters(), lr=self.lr_critic)  # 需要定义学习率
        critic_optimizer_2 = torch.optim.Adam(critic_2.parameters(), lr=self.lr_critic)  # 需要定义学习率
        critic_optimizer_1_scheduler = torch.optim.lr_scheduler.ExponentialLR(critic_optimizer_1, gamma=0.99)  # 需要定义学习率衰减
        critic_optimizer_2_scheduler = torch.optim.lr_scheduler.ExponentialLR(critic_optimizer_2, gamma=0.99)
        actor_optimizer_scheduler    = torch.optim.lr_scheduler.ExponentialLR(actor_optimizer, gamma=0.99)
        # Replay_buffer
        replay_buffer = replay_buffer.ReplayBuffer(size=10 ** 6)
    
        # Initialize GRL agent
        GRL_TD3 = TD3_agent.TD3(
            actor,
            actor_optimizer,
            critic_1,
            critic_optimizer_1,
            critic_2,
            critic_optimizer_2,
            self.explore_noise,
            self.noise_clip,# noisy
            self.warmup,  # warmup
            replay_buffer,  # replay buffer
            self.batch_size,  # batch_size
            self.update_interval,  # model update interval (< actor model) 100
            self.update_interval_actor,  # actor model update interval 500
            self.target_update_interval,  # target model update interval 5000
            self.soft_update_tau,  # soft update factor
            self.n_steps,  # multi-steps
            self.gamma,  # discount factor
            model_name="TD3_model",  # model name]
            action_size = A
        )

        # Training
        self.sm.store_model(GRL_TD3)
        # save_dir = '/home/ucluser/isaacgym/python/examples/RL/RL_TrainedModels/TD3/' + str(run)
        save_dir = self.sm.session_dir
        print(f"save_dir= {save_dir}")
        # debug_training = True
        if training:
            Training_GRLModels(GRL_TD3, self.n_episodes, self.max_episode_len, save_dir, debug_training, self.gym_instance, self.warmup, self.server)
        
        # Testing
        
        load_dir = '/home/ucluser/isaacgym/python/examples/RL/RL_TrainedModels/TD3'
        # debug_testing = False
        if testing:
            Testing_GRLModels(GRL_TD3, self.test_episodes, self.max_episode_len, load_dir, debug_training, self.gym_instance)
