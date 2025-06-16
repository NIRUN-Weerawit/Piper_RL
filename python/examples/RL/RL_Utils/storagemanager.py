from collections import deque
import json
import os
import io
import pickle
import socket
from sympy import expand_log
import torch

class StorageManager:
    def __init__(self, name, load_session, load_episode, device):
        self.machine_dir = (os.getenv('ISAACGYM_BASE_PATH') + '/python/examples/RL/RL_TrainedModels')
        self.name = name
        self.session = load_session
        self.load_episode = load_episode
        self.session_dir = os.path.join(self.machine_dir, self.session)
        self.map_location = device

    def new_session_dir(self):
        i = 0
        session_dir = os.path.join(self.machine_dir, f"{self.name}_{i}")
        while(os.path.exists(session_dir)):
            i += 1
            session_dir = os.path.join(self.machine_dir, f"{self.name}_{i}")
        self.session = f"{self.name}_{i}"
        print(f"making new model dir: {session_dir}")
        os.makedirs(session_dir)
        self.session = self.session
        self.session_dir = session_dir
        
    def find_latest_session(self):
        if not os.path.exists(self.machine_dir):
            print(f"Machine directory does not exist: {self.machine_dir}")
            return None
        
        sessions = [d for d in os.listdir(self.machine_dir) if os.path.isdir(os.path.join(self.machine_dir, d)) and d.startswith(self.name)]
        if not sessions:
            print(f"No sessions found for {self.name} in {self.machine_dir}")
            return None
        
        latest_session = max(sessions, key=lambda x: int(x.split('_')[-1]))
        self.session = latest_session
        self.session_dir = os.path.join(self.machine_dir, latest_session)
        print(f"Latest session found: {self.session}")
        # return self.session

    def find_n_session(self, n):
        if not os.path.exists(self.machine_dir):
            print(f"Machine directory does not exist: {self.machine_dir}")
            return None
        
        sessions = [d for d in os.listdir(self.machine_dir) if os.path.isdir(os.path.join(self.machine_dir, d)) and d.startswith(self.name)]
        if not sessions:
            print(f"No sessions found for {self.name} in {self.machine_dir}")
            return None
        
        if n >= len(sessions):
            print(f"Requested session {n} exceeds available sessions.")
            return None
        # print(f"Available sessions: {sessions}")
        self.session = 'PPO_' + str(n)
        self.session_dir = os.path.join(self.machine_dir, self.session)
        print(f"Session {n} found: {self.session}")
        # return self.session
    
    def delete_file(path):
        if os.path.exists(path):
            os.remove(path)

    # ------------------------------- SAVING -------------------------------

    def network_save_weights(self, network, model_dir, episode):
        filepath = os.path.join(model_dir, str(network.name) + '_episode'+str(episode)+'.pt')
        print(f"saving {network.name} model for episode: {episode}")
        torch.save(network.state_dict(), filepath)

    def save_session(self, episode, networks, pickle_data, replay_buffer):
        print(f"saving data for episode: {episode}, location: {self.session_dir}")
        for network in networks:
            self.network_save_weights(network, self.session_dir, episode)

        # Store graph data
        with open(os.path.join(self.session_dir, '_episode'+str(episode)+'.pkl'), 'wb') as f:
            pickle.dump(pickle_data, f, pickle.HIGHEST_PROTOCOL)

        # Store latest buffer (can become very large, multiple gigabytes)
        with open(os.path.join(self.session_dir, '_latest_buffer.pkl'), 'wb') as f:
            pickle.dump(replay_buffer, f, pickle.HIGHEST_PROTOCOL)

        # Delete previous iterations (except every 1000th episode)
        if (episode % 1000 == 0):
            for i in range(episode, episode - 1000, 100):
                for network in networks:
                    self.delete_file(os.path.join(self.session_dir, network.name + '_episode'+str(i)+'.pt'))
                self.delete_file(os.path.join(self.session_dir,'_episode'+str(i)+'.pkl'))
    #Have problem with pickle unpickleable
    # def store_model(self, model):
    #     with open(os.path.join(self.session_dir, '_agent.pkl'), 'wb') as f:
    #         pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

    def store_model(self, model):
        model_path = os.path.join(self.session_dir, '_agent.pth')
        if self.name == 'TD3':
            torch.save({
                'actor': model.actor.state_dict(),
                'critic_1': model.critic_1.state_dict(),
                'critic_2': model.critic_2.state_dict(),
                'actor_optimizer': model.actor_optimizer.state_dict(),
                'critic_1_optimizer': model.critic_optimizer_1.state_dict(),
                'critic_2_optimizer': model.critic_optimizer_2.state_dict(),
            }, model_path)
        elif self.name == 'PPO':
            # Assuming PPO model has actor_model and critic_model attributes
            if hasattr(model, 'actor_model') and hasattr(model, 'critic_model'):
                torch.save({
                    'actor': model.actor_model.state_dict(),
                    'critic_1': model.critic_model.state_dict(),

                }, model_path)
            else:
                print("Model does not have the expected attributes for saving.")
       
        
    def store_config(self, config):
        config_path = os.path.join(self.session_dir, 'config.txt')
        # config_path_1 = os.path.join(self.session_dir, 'config_.txt')
        # with open(config_path_1, 'wb') as f:
        #     f.write(pickle.dumps(config))
        with open(config_path, 'w') as file:
            file.write(json.dumps(config, indent= 4))
            
        # torch.save(config, config_path)
    # ------------------------------- LOADING -------------------------------

    def network_load_weights(self, network, model_dir, episode):
        filepath = os.path.join(model_dir, str(network.name) + '_episode'+str(episode)+'.pt')
        print(f"loading: {network.name} model from file: {filepath}")
        network.load_state_dict(torch.load(filepath, self.map_location))

    def load_graphdata(self):
        with open(os.path.join(self.session_dir, '_episode' + str(self.load_episode)+'.pkl'), 'rb') as f:
            return pickle.load(f)

    def load_replay_buffer(self, size, buffer_path):
        buffer_path = os.path.join(self.machine_dir, buffer_path)
        if (os.path.exists(buffer_path)):
            with open(buffer_path, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"buffer does not exist: {buffer_path}")
            return deque(maxlen=size)

    def load_model(self):
        model_path = os.path.join(self.session_dir, '_agent.pkl')
        try :
            with open(model_path, 'rb') as f:
                return CpuUnpickler(f, self.map_location).load()
        except FileNotFoundError:
            quit(f"The specified model: {model_path} was not found. Check whether you specified the correct model name")

    def load_weights(self, networks, exclude_layer=None):
        
        layers_to_load = networks[:-1] if exclude_layer else networks
        
        for network in layers_to_load:
            print(f"Loading weights for layer: {network.name}")
            self.network_load_weights(network, self.session_dir, self.load_episode)
        
        if exclude_layer:
            print(f"Skipping loading weights for the last layer: {networks[-1].name}")    
        # for network in networks:
        #     if network.name in exclude_layers:
        #         print(f"Skipping loading weights for layer: {network.name}")
        #         continue
        #     self.network_load_weights(network, self.session_dir, self.stage, self.load_episode)

class CpuUnpickler(pickle.Unpickler):
    def __init__(self, file, map_location):
        self.map_location = map_location
        super(CpuUnpickler, self).__init__(file)
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=self.map_location)
        else:
            return super().find_class(module, name)