import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse

N = 6  # Number of joints/nodes
adjacency = np.eye(N)  # Start with self-connections

for i in range(N - 1):
    adjacency[i, i + 1] = 1  # Connect i to i+1
    adjacency[i + 1, i] = 1  # Connect i+1 to i

def datatype_transmission(states, device):
    """
        1. This function is used to convert observations in the environment to the
        float32 Tensor data type that pytorch can accept.
        2. Pay attention: Depending on the characteristics of the data structure of
        the observation, the function needs to be changed accordingly.
        
        - Returns:
            - features: Node feature matrix (NxF)
            
            - adjacency: Dense adjacency matrix (NxN)
            
            - mask: Mask for controlled vehicles (Nx1)
    """

    Features = torch.as_tensor(states, dtype=torch.float32, device=device)
    Adjacency = torch.as_tensor(adjacency, dtype=torch.float32, device=device)
    Mask = torch.as_tensor(np.ones((6, 1)), dtype=torch.float32, device=device)

    return Features, Adjacency, Mask


# ------Graph Actor Model------ #
class Graph_Actor_Model(nn.Module):
    """
        1.N is the number of vehicles
        2.F is the feature length of each vehicle
        3.A is the number of selectable actions
    """
    def __init__(self, F, A, action_min, action_max):
        super(Graph_Actor_Model, self).__init__()
        self.num_outputs = A
        self.action_min = torch.tensor(action_min, dtype=torch.float32)
        self.action_max = torch.tensor(action_max, dtype=torch.float32)
        

        # Encoder
        self.encoder_1 = nn.Linear(F, 64)
        self.encoder_2 = nn.Linear(64, 64)

        # GNN
        self.GraphConv_1 = GCNConv(64, 64)
        self.GraphConv_2 = GCNConv(64, 64)
        self.GraphConv_Dense = nn.Linear(64, 64)

        # Policy network
        self.policy_1 = nn.Linear(128, 512)
        self.policy_2 = nn.Linear(512, 512)
        self.policy_3 = nn.Linear(512, 128)  
        self.policy_4 = nn.Linear(128, 32) 

        # Actor network
        self.mu = nn.Linear(32, A)
        self.sigma = nn.Linear(32, A)

        # GPU configuration
        if torch.cuda.is_available():
            GPU_num = torch.cuda.current_device()
            self.device = torch.device("cuda:{}".format(GPU_num))
        else:
            self.device = "cpu"

        self.to(self.device)
        self.action_min = self.action_min.to(self.device)
        self.action_max = self.action_max.to(self.device)

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
        # print("X_in.shape:", X_in.shape)
        # print("A_in_Dense.shape:", A_in_Dense.shape)

        # Encoder
        X = self.encoder_1(X_in)
        X = F.relu(X)
        X = self.encoder_2(X)
        X = F.relu(X)

        # GCN
        A_in_Sparse, _ = dense_to_sparse(A_in_Dense)  # 将observation的邻接矩阵转换成稀疏矩阵
        X_graph = self.GraphConv(X, A_in_Sparse)
        X_graph = F.relu(X_graph)
        X_graph = self.GraphConv_Dense(X_graph)
        X_graph = F.relu(X_graph)
        # print("X shape:", X.shape)
        # print("X_graph shape:", X_graph.shape)
       

        # Feature concatenation
        F_concat = torch.cat((X_graph, X), dim=1)
        # print("F_concat shape:", F_concat.shape)
        # Policy
        X_policy = self.policy_1(F_concat)
        X_policy = F.relu(X_policy)
        X_policy = self.policy_2(X_policy)
        X_policy = F.relu(X_policy)

        # mu and sigma
        pi_mu = self.mu(X_policy)
        pi_sigma = self.sigma(X_policy)

        # Action and log value
        pi_sigma = torch.exp(pi_sigma)
        action_probabilities = torch.distributions.Normal(pi_mu, pi_sigma)
        action = action_probabilities.sample()
        log_probs = action_probabilities.log_prob(action)
        # Action limitation
        # print("action.shape:", action.shape)
        action = torch.clamp(action, min=self.action_min.view(-1,1), max=self.action_max.view(-1,1))
        # print("action.shape:", action.shape)
        return action, log_probs, action_probabilities


# ------Graph Critic Model------ #
class Graph_Critic_Model(nn.Module):
    """
        1.N is the number of vehicles
        2.F is the feature length of each vehicle
        3.A is the number of selectable actions
    """
    def __init__(self, F, A, action_min, action_max):
        super(Graph_Critic_Model, self).__init__()
        self.num_outputs = A
        self.action_min = action_min
        self.action_max = action_max

        # Encoder
        self.encoder_1 = nn.Linear(F, 64)
        self.encoder_2 = nn.Linear(64, 256)

        # GNN
        self.GraphConv_1 = GCNConv(64, 64)
        self.GraphConv_2 = GCNConv(64, 64)
        self.GraphConv_Dense = nn.Linear(64, 64)

        # Policy network
        self.policy_1 = nn.Linear(128, 800)
        self.policy_2 = nn.Linear(800, 600)
        self.policy_3 = nn.Linear(600, 400)  
        self.policy_4 = nn.Linear(400, 200) 
        
        """self.policy_1 = nn.Linear(512, 512)
        self.policy_2 = nn.Linear(512, 64)"""

        # Critic network
        self.value = nn.Linear(200, 1)

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

        # Feature concatenation
        F_concat = torch.cat((X_graph, X), 1)

        # Policy
        X_policy = self.policy_1(F_concat)
        X_policy = F.relu(X_policy)
        X_policy = self.policy_2(X_policy)
        X_policy = F.relu(X_policy)
        X_policy = self.policy_3(X_policy)
        X_policy = F.relu(X_policy)
        X_policy = self.policy_4(X_policy)
        X_policy = F.relu(X_policy)

        # Mask
        # print("RL_indice:", RL_indice)
        # mask = torch.reshape(RL_indice, (self.num_outputs, 1))
        mask = RL_indice
        # Value
        value = self.value(X_policy)
        value = torch.mul(value, mask)

        return value


# ------NonGraph Actor Model------ #
class NonGraph_Actor_Model(nn.Module):
    """
        1.N is the number of vehicles
        2.F is the feature length of each vehicle
        3.A is the number of selectable actions
    """
    def __init__(self, S, A):
        super(NonGraph_Actor_Model, self).__init__()
        self.len_states = S
        self.num_outputs = A
        # self.action_min = action_min
        # self.action_max = action_max
        
        # Encoder
        self.encoder_1 = nn.Linear(S, 64)
        self.encoder_2 = nn.Linear(64, 1024)
        
        # Policy network
        self.policy_1 = nn.Linear(1024, 1024)  #800-600 was used before.
        self.policy_2 = nn.Linear(1024, 512)  #128-64 is bad
        self.policy_3 = nn.Linear(512, 128)   #64-128-1024-512-128
        
        # Actor network
        self.mu = nn.Linear(128, A)
        self.sigma = nn.Linear(128, A)

        # GPU configuration
        if torch.cuda.is_available():
            GPU_num = torch.cuda.current_device()
            self.device = torch.device("cuda:{}".format(GPU_num))
        else:
            self.device = "cpu"

        self.to(self.device)

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, observation):
        
        #Encoding
        X_in = F.relu(self.encoder_1(observation))
        X_in = F.relu(self.encoder_2(X_in))  
        
        # Policy
        X_policy = self.policy_1(X_in)
        X_policy = F.relu(X_policy)
        X_policy = self.policy_2(X_policy)
        X_policy = F.relu(X_policy)
        X_policy = self.policy_3(X_policy)
        X_policy = F.relu(X_policy)

        # mu and sigma
        pi_mu = self.mu(X_policy)
        pi_sigma = self.sigma(X_policy)

        # Action and log value
        pi_sigma = torch.exp(pi_sigma)
        action_probabilities = torch.distributions.Normal(pi_mu, pi_sigma)
        action = action_probabilities.sample()
        log_probs = action_probabilities.log_prob(action).sum()
        
        # Action limitation
        # action = torch.clamp(action, min=self.action_min, max=self.action_max)

        return action, log_probs, action_probabilities


# ------NonGraph Critic Model------ #
class NonGraph_Critic_Model(nn.Module):
    """
        1.N is the number of vehicles
        2.F is the feature length of each vehicle
        3.A is the number of selectable actions
    """
    def __init__(self, S, A):
        super(NonGraph_Critic_Model, self).__init__()
        self.len_states = S
        self.num_outputs = A
        # self.action_min = action_min
        # self.action_max = action_max
        
        # State Encoder
        self.encoder_1 = nn.Linear(S, 64)
        self.encoder_2 = nn.Linear(64, 128)

        # Policy network
        self.policy_1 = nn.Linear(128, 512)  #800-600-400 was used before and it was good.
        self.policy_2 = nn.Linear(512, 1024) #256-512-512-256
        self.policy_3 = nn.Linear(1024, 512) #128-256-128-64 is bad
        self.policy_4 = nn.Linear(512, 64)

        # Critic network
        self.value = nn.Linear(64, 1)

        # GPU configuration
        if torch.cuda.is_available():
            GPU_num = torch.cuda.current_device()
            self.device = torch.device("cuda:{}".format(GPU_num))
        else:
            self.device = "cpu"

        self.to(self.device)

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, observation):
        """
            1.The data type here is numpy.ndarray, which needs to be converted to a
            Tensor data type.
            2.Observation is the state observation matrix, including X_in, and RL_indice.
            3.X_in is the node feature matrix, RL_indice is the reinforcement learning
            index of controlled vehicles.
        """
        N_encoded = F.relu(self.encoder_1(observation))
        N_encoded = F.relu(self.encoder_2(N_encoded))
        
        # Policy
        X_policy = self.policy_1(N_encoded)
        X_policy = F.relu(X_policy)
        X_policy = self.policy_2(X_policy)
        X_policy = F.relu(X_policy)
        X_policy = self.policy_3(X_policy)
        X_policy = F.relu(X_policy)
        X_policy = self.policy_4(X_policy)
        X_policy = F.relu(X_policy)
        
        #Mask
        Mask = torch.as_tensor(np.ones((6, 1)), dtype=torch.float32, device=self.device)
        mask = torch.reshape(Mask, (self.num_outputs, 1))

        # Value
        value = self.value(X_policy)
        value = torch.mul(value, mask)
        
        return value
