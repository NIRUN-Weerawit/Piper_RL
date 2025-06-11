"""
    This function is used to define the PPO agent
"""

import torch
import numpy as np
import torch.autograd as autograd
import torch.nn.functional as F
import collections

# CUDA configuration
USE_CUDA = torch.cuda.is_available()
# Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() \
#     if USE_CUDA else autograd.Variable(*args, **kwargs)


class PPOMemory(object):
    """
        Define PPOMemory class as replay buffer

        Parameter description:
        --------
        state: current state
    """

    def __init__(self, batch_size):
        self.states = []
        self.probs = []  # Action probability
        self.vals = []  # Value
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batch(self):
        """
           <batch sampling function>
           Used to implement empirical sampling of PPOMemory
        """
        '''n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        # print("Batch start indices:", batch_start)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]
        # print("Batches:", batches)
        # print("Number of batches:", len(batches))

        return self.states, \
               self.actions, \
               self.probs, \
               self.vals, \
               np.asarray(self.rewards), \
               np.asarray(self.dones), \
               batches'''

            # Convert lists to tensors
        states      = torch.stack(self.states)
        actions     = torch.stack(self.actions)
        probs       = torch.stack(self.probs)
        vals        = torch.stack(self.vals)
        rewards     = torch.tensor(self.rewards)
        dones       = torch.tensor(self.dones)

        # Generate random permutation of indices
        n_states = states.size(0)
        indices = torch.randperm(n_states)

        # Apply permutation to shuffle data
        states = states[indices]
        actions = actions[indices]
        probs = probs[indices]
        vals = vals[indices]
        rewards = rewards[indices]
        dones = dones[indices]

        # Create batches
        batch_start = torch.arange(0, n_states, self.batch_size)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return states, actions, probs, vals, rewards, dones, batches
        
    def store_memory(self, state, action, probs, vals, reward, done):
        """
           <data storage function>
           Used to store the data of the agent interaction process

           Parameters:
           --------
           state: current state
           action: current action
           probs: action probability
           vals: value of the action
           reward: the reward for performing the action
           done: whether the current round is completed or not
        """
        # state = state.cpu().detach().numpy() if USE_CUDA else state.detach().numpy()
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        """
           <data clear function>
           Used to clear the interaction data already stored and free memory
        """
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class PPO(object):
    """
        Define the PPO class (Proximal Policy Optimization)

        Parameter description:
        --------
        actor_model: actor network
        actor_optimizer: actor optimizer
        critic_model: value network
        critic_optimizer: critic optimizer
        gamma: discount factor
        GAE_lambda: GAE (generalized advantage estimator) coefficient
        policy_clip: policy clipping coefficient
        batch_size: sample size
        n_epochs: number of updates per batch
        update_interval: model update step interval
        model_name: model name (used to save and read)
    """

    def __init__(self,
                 actor_model,
                 critic_model,
                 actor_optimizer,
                 critic_optimizer,
                 actor_optimizer_scheduler,
                 critic_optimizer_scheduler,
                 schedule_update_interval,
                 gamma,
                 GAE_lambda,
                 policy_clip,
                 batch_size,
                 n_epochs,
                 update_interval,
                 model_name):

        self.actor_model                = actor_model
        self.critic_model               = critic_model
        self.actor_optimizer            = actor_optimizer
        self.critic_optimizer           = critic_optimizer
        self.actor_optimizer_scheduler  = actor_optimizer_scheduler
        self.critic_optimizer_scheduler = critic_optimizer_scheduler
        self.schedule_update_interval   = schedule_update_interval  # Update interval for the scheduler
        self.gamma                      = gamma
        self.GAE_lambda                 = GAE_lambda
        self.policy_clip                = policy_clip
        self.batch_size                 = batch_size
        self.n_epochs                   = n_epochs
        self.update_interval            = update_interval
        self.model_name                 = model_name
        self.time_step                  = 0  # Time step counter

        # GPU configuration
        if USE_CUDA:
            GPU_num = torch.cuda.current_device()
            self.device = torch.device("cuda:{}".format(GPU_num))
        else:
            self.device = "cpu"

        # Replay buffer
        self.memory = PPOMemory(self.batch_size)

        # Record loss
        self.loss_record_actor = collections.deque(maxlen=100)
        self.loss_record_critic = collections.deque(maxlen=100)

    def store_transition(self, state, action, probs, vals, reward, done):
        """
           <Experience storage function>
           Used to store the experience data during the agent learning process

           Parameters:
           --------
           state: the state of the current moment
           action: current moment action
           probs: probability of current action
           vals: the value of the current action
           reward: the reward obtained after performing the current action
           done: whether to terminate or not
        """
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def choose_action(self, observation):
        """
          <Action selection function>
          Generate agent's action based on environment observation

          Parameters:
          --------
          observation: the environment observation of the smart body
        """
        # print("observation shape:", observation.shape)
        action, probs, _ = self.actor_model(observation)
        value = self.critic_model(observation)

        action = torch.squeeze(action)
        probs = torch.squeeze(probs)
        value = torch.squeeze(value)

        return action, probs, value

    def learn(self):
        """
           <policy update function>
           Used to implement the agent's learning process
        """
        # ------Training according to the specific value of n_epochs------ #
        self.time_step += 1
        
        if self.time_step % self.schedule_update_interval == 0:
            self.actor_optimizer_scheduler.step()
            self.critic_optimizer_scheduler.step()

        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, \
            reward_arr, dones_arr, batches = \
                self.memory.generate_batch()

            values = vals_arr

            # ------Training for each epochs------ #
            # advantage
            advantage = torch.zeros(len(reward_arr), len(action_arr[1])).to(self.device)
            # advantage = torch.zeros(len(reward_arr)).to(self.device)
            # print("advantage shape:", advantage.shape)
            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] *
                                       (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.GAE_lambda
                # advantage[t] = a_t
                advantage[t, :] = a_t
            # print("advantage shape:", advantage.shape)
            # advantage = torch.stack(advantage)

            # values
            # values = torch.stack(values)

            # Training for the collected samples
            # Note: do loss.backward() in a loop, to avoid gradient compounding
            

            # Train for each index in the batch
            # Calculate actor_loss and update the actor network
            for batch in batches:
                # Initialize the loss matrix
                actor_loss_matrix = []
                critic_loss_matrix = []
                
                for i in batch:
                    old_probs = old_prob_arr[i].detach()
                    actions = action_arr[i].detach()

                    _, _, dist = self.actor_model(state_arr[i])

                    new_probs = dist.log_prob(actions)
                    prob_ratio = new_probs.exp() / old_probs.exp()
                    weighted_probs = advantage[i].detach() * prob_ratio  # PPO1
                    # ------PPO2------#
                    weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,
                                                     1 + self.policy_clip) * advantage[i].detach()
                    # ----------------#
                    actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                    actor_loss_matrix.append(actor_loss)

                actor_loss_matrix = torch.stack(actor_loss_matrix)
                
                """old_probs = old_prob_arr.detach()
                actions = action_arr.detach()
                print("state_arr shape:", state_arr.shape)
                _, _, dist = self.actor_model(state_arr)

                new_probs = dist.log_prob(actions).sum()
                # print("new_probs shape:", new_probs.shape)
                prob_ratio = new_probs.exp() / old_probs.exp()
                # print("prob_ratio shape:", prob_ratio.shape)
                weighted_probs = advantage.detach() * prob_ratio  # PPO1
                # weighted_probs = torch. 
                # ------PPO2------#
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,
                                                    1 + self.policy_clip) * advantage.detach()
                # ----------------#
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()"""

                # actor_loss_matrix.append(actor_loss)

                # actor_loss_matrix = torch.stack(actor_loss_matrix)
                actor_loss_mean = torch.mean(actor_loss_matrix)

                self.actor_optimizer.zero_grad()
                actor_loss_mean.backward()
                self.actor_optimizer.step()

                """# Calculate critic_loss and update the critic network
                critic_value = self.critic_model(state_arr)
                # print("critic_value shape:", critic_value.shape)
                critic_value = torch.squeeze(critic_value)
                
                # print("advantage shape:", advantage.shape)
                # print("values shape:", values.shape)
                returns = advantage + values
                # print("returns shape:", returns.shape)
                returns = returns.detach()
                # returns = torch.squeeze(returns)
                
                # critic_value = self.critic_model(state_arr[i])
                # critic_value = torch.squeeze(critic_value)

                # Ensure both are 1D tensors of the same shape
                # returns = returns.view(-1)
                # critic_value = critic_value.view(-1)
                
                critic_loss = F.smooth_l1_loss(returns, critic_value)

                # critic_loss_matrix.append(critic_loss)"""

                for i in batch:
                    critic_value = self.critic_model(state_arr[i])
                    critic_value = torch.squeeze(critic_value)

                    returns = advantage[i] + values[i]
                    returns = returns.detach()
                    critic_loss = F.smooth_l1_loss(returns, critic_value)

                    critic_loss_matrix.append(critic_loss)
                
                critic_loss_matrix = torch.stack(critic_loss_matrix)
    
                critic_loss_mean = 0.5 * torch.mean(critic_loss)



                self.critic_optimizer.zero_grad()
                critic_loss_mean.backward()
                self.critic_optimizer.step()
                
                

                
                # self.loss_record.append(float((actor_loss_mean + critic_loss_mean).detach().cpu().numpy()))
                self.loss_record_critic.append(float((critic_loss_mean).detach().cpu().numpy()))
                self.loss_record_actor.append(float((actor_loss_mean).detach().cpu().numpy()))
        
        self.memory.clear_memory()

    def get_statistics(self):
        """
           <training data acquisition function>
           Used to get the relevant data during training
        """
        # loss_statistics = np.mean(self.loss_record) if self.loss_record else np.nan
        loss_critic = np.mean(self.loss_record_critic) if self.loss_record_critic else np.nan
        loss_actor = np.mean(self.loss_record_actor) if self.loss_record_actor else np.nan
        return [loss_actor, loss_critic]

    def save_model(self, save_path):
        """
           <Model saving function>
           Used to save the trained model
        """
        save_path_actor = save_path + "/" + self.model_name + "_actor" + ".pt"
        save_path_critic = save_path + "/" + self.model_name + "_critic" + ".pt"
        torch.save(self.actor_model, save_path_actor)
        torch.save(self.critic_model, save_path_critic)

    def load_model(self, load_path):
        """
           <Model reading function>
           Used to read the trained model
        """
        load_path_actor = load_path + "/" + self.model_name + "_actor" + ".pt"
        load_path_critic = load_path + "/" + self.model_name + "_critic" + ".pt"
        self.actor_model = torch.load(load_path_actor)
        self.critic_model = torch.load(load_path_critic)





        '''for batch in batches:
                # Initialize the loss matrix
                actor_loss_matrix = []
                critic_loss_matrix = []

                # Train for each index in the batch
                # Calculate actor_loss and update the actor network
                for i in batch:
                    old_probs = old_prob_arr[i].detach()
                    actions = action_arr[i].detach()

                    _, _, dist = self.actor_model(state_arr[i])

                    new_probs = dist.log_prob(actions)
                    prob_ratio = new_probs.exp() / old_probs.exp()
                    weighted_probs = advantage[i].detach() * prob_ratio  # PPO1
                    # ------PPO2------#
                    weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,
                                                     1 + self.policy_clip) * advantage[i].detach()
                    # ----------------#
                    actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                    actor_loss_matrix.append(actor_loss)

                actor_loss_matrix = torch.stack(actor_loss_matrix)
                actor_loss_mean = torch.mean(actor_loss_matrix)

                self.actor_optimizer.zero_grad()
                actor_loss_mean.backward()
                self.actor_optimizer.step()

                # Calculate critic_loss and update the critic network
                for i in batch:
                    critic_value = self.critic_model(state_arr[i])
                    critic_value = torch.squeeze(critic_value)

                    returns = advantage[i] + values[i]
                    returns = returns.detach()
                    critic_value = self.critic_model(state_arr[i])
                    critic_value = torch.squeeze(critic_value)

                    # Ensure both are 1D tensors of the same shape
                    returns = returns.view(-1)
                    critic_value = critic_value.view(-1)

                    critic_loss = F.smooth_l1_loss(returns, critic_value)

                    critic_loss_matrix.append(critic_loss)

                critic_loss_matrix = torch.stack(critic_loss_matrix)
                critic_loss_mean = 0.5 * torch.mean(critic_loss_matrix)

                self.critic_optimizer.zero_grad()
                critic_loss_mean.backward()
                self.critic_optimizer.step()

                # self.loss_record.append(float((actor_loss_mean + critic_loss_mean).detach().cpu().numpy()))
                self.loss_record_critic.append(float((critic_loss_mean).detach().cpu().numpy()))
                self.loss_record_actor.append(float((actor_loss_mean).detach().cpu().numpy()))
        '''