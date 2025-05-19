# from flow.core.params import VehicleParams, SumoCarFollowingParams, SumoParams, EnvParams, \
#     InitialConfig, NetParams
# from flow.controllers import IDMController, RLController, ContinuousRouter
# from GRL_Envs.FigureEight.FE_specific import AccelEnv  # 定义仿真环境
# from GRL_Envs.FigureEight.FE_network import FigureEightNetwork, ADDITIONAL_NET_PARAMS  # 定义路网文件

# from controller import SpecificMergeRouter, NearestMergeRouter
# from network import HighwayRampsNetwork, ADDITIONAL_NET_PARAMS

'''
    Setup the configurations of the experiment and initiate the experiment
    
'''

# ----------- Configurations -----------#
TRAINING = True
# TRAINING = False

# TESTING = True
TESTING = False

# Graph configuration
# Enable_Graph = True
Enable_Graph = False

DEBUG = True
# DEBUG = False

RENDER = False
#RENDER = True

# Time horizon
HORIZON = 800

num_envs = 1

warmup  = 2000

args = dict(sim_device          = "cuda:0",
            pipeline            = "gpu",      #cpu/gpu
            graphics_device_id  = 0,
            physics_engine     = "physx",     #flex/physx
            num_threads         = 0,          #Number of cores used by PhysX
            subscenes           = 0, 
            slices              = None,
            num_envs            = num_envs,
            dist_reward_scale   = 1.0,
            rot_reward_scale    = 0.1,
            stiffness           = 1000.0,
            damping             = 200.0,
            debug               = True,
            headless            = True,
            debug_interval      = 200,
            dt                  = 0.01,  #time_step duration for executing command   
            warmup              = warmup,
            state_size                  = 21,
            action_size                 = 6,
            hidden_size                 = 256,
            lr_critic                   = 0.0005,
            lr_actor                    = 0.0001,
            explore_noise               = 0.3,
            gamma                       = 0.99,
            batch_size                  = 128,  # batch_size
            update_interval             = 20,  # model update interval (< actor model) 100
            update_interval_actor       = 40,  # actor model update interval 500
            target_update_interval      = 100,  # target model update interval 5000
            soft_update_tau             = 0.001,  # soft update factor
            n_steps                     = 1,
            test_episodes               = 10,
            n_episodes                  = 500, 
            max_episode_len             = 1000, 
            warmup_step                 = 1000,
)
# Boundary of action space
action_min = [-2.618, 0.0,  -2.697, -1.832, -1.22, -3.14, 0.0, -0.04] #Max. joints' limits
action_max = [ 2.618, 3.14,  0.0,    1.832,  1.22,  3.14, 0.04, 0.0 ]  #Min. joints' limits

# ----------- Simulation -----------#
# 1.Simulation setup


# simulation start
from RL_Experiment.Exp_TD3 import Experiment

exp = Experiment(args)
# run the sumo simulation
exp.run(num_envs=num_envs,
        training=TRAINING, testing=TESTING,
        Graph=Enable_Graph, debug_testing= False, debug_training=False)
