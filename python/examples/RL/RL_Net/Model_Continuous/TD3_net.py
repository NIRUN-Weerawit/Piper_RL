import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse


def datatype_transmission(states, device):
    """
        1.This function is used to convert observations in the environment to the
        float32 Tensor data type that pytorch can accept.
        2.Pay attention: Depending on the characteristics of the data structure of
        the observation, the function needs to be changed accordingly.
    """
    features = torch.as_tensor(states[0], dtype=torch.float32, device=device)
    adjacency = torch.as_tensor(states[1], dtype=torch.float32, device=device)
    mask = torch.as_tensor(states[2], dtype=torch.float32, device=device)

    return features, adjacency, mask


# Defining Ornstein-Uhlenbeck noise for stochastic exploration processes
class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OUActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


# ------Graph Actor Model------ #
class Graph_Actor_Model(nn.Module):
    """
        Under development
    """
    def __init__(self, N, F, A, action_min, action_max):
        super(Graph_Actor_Model, self).__init__()
        self.num_agents = N
        self.num_outputs = A
        self.action_min = action_min
        self.action_max = action_max

        # Encoder
        self.encoder_1 = nn.Linear(F, 32)
        self.encoder_2 = nn.Linear(32, 32)

        # GNN
        self.GraphConv = GCNConv(32, 32)
        self.GraphConv_Dense = nn.Linear(32, 32)

        # Policy network
        self.policy_1 = nn.Linear(64, 32)
        self.policy_2 = nn.Linear(32, 32)

        # Actor network
        self.pi = nn.Linear(32, A)

        # GPU configuration
        if torch.cuda.is_available():
            GPU_num = torch.cuda.current_device()
            self.device = torch.device("cuda:{}".format(GPU_num))
        else:
            self.device = "cpu"

        self.to(self.device)

    def forward(self, observation):
        """
            1.The data type here is numpy.ndarray, which needs to be converted to a
            Tensor data type.
            2.Observation is the state observation matrix, including X_in, A_in_Dense
            and RL_indice.
            3.X_in is the node feature matrix, A_in_Dense is the dense adjacency matrix
            (NxN) (original input)
            4.A_in_Sparse is the sparse adjacency matrix COO (2xnum), RL_indice is the
            reinforcement learning index of controlled vehicles.
        """

        X_in, A_in_Dense, RL_indice = datatype_transmission(observation, self.device)

        # Encoder
        X = self.encoder_1(X_in)
        X = F.relu(X)
        X = self.encoder_2(X)
        X = F.relu(X)

        # GCN
        A_in_Sparse, _ = dense_to_sparse(A_in_Dense)
        X_graph = self.GraphConv(X, A_in_Sparse)
        X_graph = F.relu(X_graph)
        X_graph = self.GraphConv_Dense(X_graph)
        X_graph = F.relu(X_graph)

        # Features concatenation
        F_concat = torch.cat((X_graph, X), 1)

        # Policy
        X_policy = self.policy_1(F_concat)
        X_policy = F.relu(X_policy)
        X_policy = self.policy_2(X_policy)
        X_policy = F.relu(X_policy)

        # Pi
        pi = self.pi(X_policy)

        # Action limitation
        # amplitude = 0.5 * (self.action_max - self.action_min)
        # mean = 0.5 * (self.action_max + self.action_min)
        # action = amplitude * torch.tanh(pi) + mean
        action = torch.tanh(pi) 
        
        
        return action
    
# ------Graph Critic Model------ #
class Graph_Critic_Model(nn.Module):
    """
        Under development
    """
    def __init__(self, N, F, A, action_min, action_max):
        super(Graph_Critic_Model, self).__init__()
        self.num_agents = N
        self.num_outputs = A
        self.action_min = action_min
        self.action_max = action_max

        # Encoder
        self.encoder_1 = nn.Linear(F + A, 32)  # Considering action space
        self.encoder_2 = nn.Linear(32, 32)

        # GNN
        self.GraphConv = GCNConv(32, 32)
        self.GraphConv_Dense = nn.Linear(32, 32)

        # Policy network
        self.policy_1 = nn.Linear(64, 32)
        self.policy_2 = nn.Linear(32, 32)

        # Critic network
        self.value = nn.Linear(32, 1)

        # GPU configuration
        if torch.cuda.is_available():
            GPU_num = torch.cuda.current_device()
            self.device = torch.device("cuda:{}".format(GPU_num))
        else:
            self.device = "cpu"

        self.to(self.device)

    def forward(self, observation, action):
        """
            1.The data type here is numpy.ndarray, which needs to be converted to a
            Tensor data type.
            2.Observation is the state observation matrix, including X_in, A_in_Dense
            and RL_indice.
            3.X_in is the node feature matrix, A_in_Dense is the dense adjacency matrix
            (NxN) (original input)
            4.A_in_Sparse is the sparse adjacency matrix COO (2xnum), RL_indice is the
            reinforcement learning index of controlled vehicles.
        """

        X_in, A_in_Dense, RL_indice = datatype_transmission(observation, self.device)

        # Encoder
        X_in = torch.cat((X_in, action), 1)
        X = self.encoder_1(X_in)
        X = F.relu(X)
        X = self.encoder_2(X)
        X = F.relu(X)

        # GCN
        A_in_Sparse, _ = dense_to_sparse(A_in_Dense)
        X_graph = self.GraphConv(X, A_in_Sparse)
        X_graph = F.relu(X_graph)
        X_graph = self.GraphConv_Dense(X_graph)
        X_graph = F.relu(X_graph)

        # Feature concatenation
        F_concat = torch.cat((X_graph, X), 1)

        # Policy network
        X_policy = self.policy_1(F_concat)
        X_policy = F.relu(X_policy)
        X_policy = self.policy_2(X_policy)
        X_policy = F.relu(X_policy)

        # Value calculation
        V = self.value(X_policy)

        return V


# ------NonGraph Actor Model------ #
class NonGraph_Actor_Model(nn.Module):
    """
       1.N is the number of vehicles
       2.F is the feature length of each vehicle
       3.A is the number of selectable actions
       
        Inputs: 
            # - gripper_pose: A list of end-effector transformation including position and orientation: [(x, y, z), (x, y, z, w)]
            - goal_position: A goal position of end-effector ('p',[x,y,z]) 3
            - goal_orientation: A goal posture of the end-effector ('r', [(x,y,z,w)]) 4
            # - joint_angles:  A list of all joint angles 8
            - end_effector_position: A current position of the end-effector 3
            - end_effector_orientation: A current posture of the end-effector 4
        
        Outputs:
            - joints_angle: A list of joints' angle: [j1, j2, j3, j4, j5, j6, j7, j8]

        Parameters:
            - N is the length of gripper_pose
   """
    def __init__(self, N, A, action_max, action_min, hidden_size):
        super(NonGraph_Actor_Model, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # print("ACTOR device:", self.device)
        
        self.len_states  = N
        self.num_outputs = A
        # hidden_size      = hidden_size
        
        #----Numpy version of limits-----#
        self.action_max  = action_max
        self.action_min  = action_min
        
        #----Tensor version of limits-----#
        self.action_max_tensor  = torch.tensor(action_max, dtype=torch.float32)
        self.action_min_tensor  = torch.tensor(action_min, dtype=torch.float32)
        # Encoder
        self.encoder_1 = nn.Linear(N, 32)
        self.encoder_2 = nn.Linear(32, 32)
        
        self.policy_1 = nn.Linear(32,   800)
        self.policy_2 = nn.Linear(800,  1200)
        self.policy_3 = nn.Linear(1200, 600)
        
        '''self.policy_1 = nn.Linear(32,                    hidden_size)
        self.policy_2 = nn.Linear(hidden_size,          hidden_size * 2)
        self.policy_3 = nn.Linear(hidden_size * 2,      hidden_size)'''
        
        # self.policy_3 = nn.Linear(hidden_size * 2,  hidden_size * 2)
        # self.policy_4 = nn.Linear(hidden_size * 2,  hidden_size)
        # self.policy_5 = nn.Linear(hidden_size,      int(hidden_size / 2))
        # self.policy_6 = nn.Linear(int(hidden_size / 2),  int(hidden_size / 4))
        
        # Actor network
        self.pi = nn.Linear(600, A)
        # self.pi = nn.Linear(int(hidden_size / 4), A)
        self.to(self.device)

    def forward(self, observation):
        # observation = observation.unsqueeze(0)
        
        # Policy
        X_in = F.relu(self.encoder_1(observation))
        X_in = F.relu(self.encoder_2(X_in))        
        
        X_policy_1 = F.relu(self.policy_1(X_in))
        X_policy_2 = F.relu(self.policy_2(X_policy_1))
        X_policy_3 = F.relu(self.policy_3(X_policy_2))
        # X_policy = F.relu(self.policy_4(X_policy))
        # X_policy = F.relu(self.policy_5(X_policy))
        # X_policy = F.relu(self.policy_6(X_policy))
        
        # Pi
        pi = self.pi(X_policy_3)
        action = torch.tanh(pi)
        # print(f"action: {action}")
        
        
        return action       #action.detach().cpu().numpy().astype(np.float32)     #action.squeeze(0)
    
    
        # print("action")
        # print(',    '.join(f'{q:.2f}' for q in action))
        # Action limitation
        # amplitude   = 0.5 * (self.action_max_tensor - self.action_min_tensor)
        # mean        = 0.5 * (self.action_max_tensor + self.action_min_tensor)
        # action      = amplitude * torch.tanh(pi) + mean
        
        
        #----Declare buffer of limits-----#
        # self.register_buffer('buffer_action_max', self.action_max)
        # self.register_buffer('buffer_action_min', self.action_min)
# amplitude  = torch.tensor(0.5*(self.action_max - self.action_min),  dtype=torch.float32, device=pi.device)
        # mean = torch.tensor(0.5*(self.action_max + self.action_min),        dtype=torch.float32, device=pi.device)
        
        # X_in, _, RL_indice = datatype_transmission(observation, self.device)
        # if isinstance(observation, np.ndarray):
            # from NumPy to torch Tensor on the correct device
            # observation = torch.from_numpy(observation).to(self.device)
        # print(f"type of observation: {type(observation)}, device: {observation.get_device()}")
        # print(f"device of policy 1: {self.policy_1.get_device()} ")
        # X_in = observation      # now a Tensor
        # print("X_in", X_in)
        # X_in = torch.from_numpy(observation)    #.to(self.device)
        # print(f"type of X_in: {type(X_in)}, device: {X_in.get_device()}")
        
        
# ------NonGraph Critic Model------ #
class NonGraph_Critic_Model(nn.Module):

    def __init__(self, N, A, action_min, action_max, hidden_size):
        super(NonGraph_Critic_Model, self).__init__() 
        self.len_states     = N
        self.num_outputs    = A
        # State Encoder
        self.encoder_1 = nn.Linear(N, 32)
        self.encoder_2 = nn.Linear(32, 32)
        # Policy network
        self.TFC_A = nn.Linear(A,                   600)
        self.FC_N  = nn.Linear(32,                  800)
        self.TFC_N = nn.Linear(800,     600)
        self.CFC   = nn.Linear(1200,     600)
        
        '''self.TFC_A = nn.Linear(A,                   hidden_size)
        self.FC_N  = nn.Linear(32,                  hidden_size * 2)
        self.TFC_N = nn.Linear(hidden_size * 2,     hidden_size)
        self.CFC   = nn.Linear(hidden_size * 2,     hidden_size)'''
        
        '''self.policy_1 = nn.Linear(N,                hidden_size)
        self.policy_2 = nn.Linear(hidden_size + A,  hidden_size)
        self.policy_3 = nn.Linear(hidden_size,      hidden_size)'''

        # Critic network
        self.value = nn.Linear(600, 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print("device:", self.device)
        self.to(self.device)
        
    def forward(self, observation, action):
        # Policy
        '''X_in = F.relu(self.policy_1(observation))
        # print(f"X_in = {X_in.shape}")

        X = torch.cat((X_in, action), dim=1)
        X_policy = F.relu(self.policy_2(X))
        X_policy = F.relu(self.policy_3(X_policy))'''

        A_in = self.TFC_A(action)
        N_encoded = F.relu(self.encoder_1(observation))
        N_encoded = F.relu(self.encoder_2(N_encoded))
        
        N_in = F.relu(self.FC_N(N_encoded))
        N_in = self.TFC_N(N_in)
        
        X = torch.cat((A_in, N_in), dim=1)
        X_policy = F.relu(self.CFC(X))

        # Value
        V = self.value(X_policy)

        return V
