"""Contains an experiment class for running simulations."""

from RL_Utils.create_env_copy import Gym_env
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
        flow_params : dict
            flow-specific parameters
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
        self.n_epochs               = args['n_epochs']
        self.max_episode_len        = args['max_episode_len']
        self.server                 = args['server']
        self.action_min             = [-1] * 6
        self.action_max             = [1] * 6
        self.args                   = args
        self.n                      = args['EP']

        # Get the env name and a creator for the environment.
        self.gym_instance = Gym_env(args)
        self.gym_instance.create_piper_env()
        self.gym_instance.set_seed(11)
        
        # self.gym_instance.dist_reward_scale = args['dist_reward_scale']
        # self.gym_instance.rot_reward_scale = args['rot_reward_scale']
        self.sim = self.gym_instance.sim 
        
        # self.action_min = self.gym_instance.piper_lower_limits
        # self.action_max = self.gym_instance.piper_upper_limits

        self.sm = StorageManager('PPO', "", 0, 'cuda:0')
        
        
        logging.info(" Starting experiment at {}".format(str(datetime.datetime.now())))

        logging.info("Initializing environment.")

    def run(self, num_envs, training, testing, Graph, debug_training, debug_testing):

        # from GRL_Utils.Train_and_Test_PPO import Training_GRLModels, Testing_GRLModels
        import torch
        import torch.nn

        from RL_Library                     import PPO_agent
        from RL_Utils.Train_and_Test_PPO    import Training_GRLModels, Testing_GRLModels

        # Initialize GRL model
        N = self.N
        A = self.A
        F = 11 # Number of features in the graph model
        # action_min = [-3.1415, - 3.40339, -3.1415, -3.92699, -3.92699, -3.92699]  # Min. joints' velocities limits
        # action_max = [3.1415, 3.40339, 3.1415, 3.92699, 3.92699, 3.92699] 
        action_min = [-3.1415, - 3.40339]  # Min. joints' velocities limits
        action_max = [3.1415, 3.4033] 
        assert isinstance(Graph, bool)
        if Graph:
            from RL_Net.Model_Continuous.PPO import Graph_Actor_Model, \
                Graph_Critic_Model 

            GRL_actor = Graph_Actor_Model(F, 1, action_min, action_max)
            GRL_critic = Graph_Critic_Model(F, 1, action_min, action_max)
        else:
            from RL_Net.Model_Continuous.PPO import NonGraph_Actor_Model, \
                NonGraph_Critic_Model
            GRL_actor  = NonGraph_Actor_Model(N,  A, action_min, action_max)
            GRL_critic = NonGraph_Critic_Model(N, A)

        actor_optimizer = torch.optim.Adam(GRL_actor.parameters(), lr=self.lr_actor)
        critic_optimizer = torch.optim.Adam(GRL_critic.parameters(), lr=self.lr_critic)
        critic_optimizer_scheduler = torch.optim.lr_scheduler.ExponentialLR(critic_optimizer, gamma=0.9999)  # 需要定义学习率衰减
        actor_optimizer_scheduler    = torch.optim.lr_scheduler.ExponentialLR(actor_optimizer, gamma=0.9999)
        
        # Discount factor
        gamma = 0.99
        # GAE factor
        GAE_lambda = 0.98
        # Policy clip factor
        policy_clip = 0.2
        
        schedule_update_interval = self.update_interval * 2
        
        # Initialize GRL agent
        GRL_PPO = PPO_agent.PPO(
            GRL_actor,  # actor model
            GRL_critic,  # critic model
            actor_optimizer,
            critic_optimizer,
            actor_optimizer_scheduler,
            critic_optimizer_scheduler,
            schedule_update_interval,  # update interval for actor and critic optimizers
            gamma,  # discount factor
            GAE_lambda,  # GAE factor
            policy_clip,  # policy clip factor
            batch_size=self.batch_size,  # batch_size < update_interval
            n_epochs=self.n_epochs,  # update times for one batch_size
            update_interval=self.update_interval,  # update interval
            model_name="PPO_Model"  # model
        )

        # Training
        n_episodes = 150
        max_episode_len = 2500
        
        
        # save_dir = '../GRL_TrainedModels/PPO/DQN5'
        debug_training = False
        if training:
            self.sm.new_session_dir()
            self.sm.store_config(self.args)
            self.sm.store_model(GRL_PPO)
            # save_dir = '/home/ucluser/isaacgym/python/examples/RL/RL_TrainedModels/TD3/' + str(run)
            save_dir = self.sm.session_dir
            Training_GRLModels(GRL_PPO, self.n_episodes, self.max_episode_len, save_dir, debug_training, self.gym_instance, self.warmup, self.server)

        # Testing
        
        # load_dir = '/home/ucluser/isaacgym/python/examples/RL/RL_TrainedModels/PPO_11'
        debug_testing = False
        if testing:
            test_episodes = 10
            # self.sm.find_latest_session()
            self.sm.find_n_session(self.n)
            load_dir = self.sm.session_dir
            # load_dir = '/home/ucluser/isaacgym/python/examples/RL/RL_TrainedModels/PPO_27'
            Testing_GRLModels(GRL_PPO, test_episodes, self.max_episode_len, load_dir, debug_training, self.gym_instance)
