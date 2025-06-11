


def step(self, rl_actions):
    self.time_counter += 1
    self.time_ep += 1

    # advance the simulation in the simulator by one step
    self.step_physics()
    
    self.refresh()
    
    #take actions given by RL agent
    # self.apply_rl_actions(rl_actions)
    if rl_actions is None:
        # self.apply_rl_actions_force(np.zeros(6))
        self.apply_rl_actions_velocity(np.zeros(6))
        # print("NONE ACTION IS ACTIVATED")
        self.update()
        
        self.render()
        states, state_tensor = self.get_states()
        # print(f"states: {states}")
        return state_tensor, 0, False
    else:    
        # self.apply_rl_actions_force(rl_actions)
        self.apply_rl_actions_velocity(rl_actions)
    
    
    # render a frame
    self.render()
    
    self.update()
    # actions = np.array(rl_actions) * 0.1
    
    # print("action=" ,actions)
    # print("velocity=", self.piper_dof_states['vel'])
    # diff = []
    # for j in range(len(actions)):
        # diff[j] = actions[j] - self.piper_dof_states['vel'][j]
    # diff = [actions[j] - self.piper_dof_states['vel'][j] for j in range(len(actions))]
    # print("diff=", diff)
    # print("mean=", np.array(diff).mean())
    
    
    
    states_np, states_tensor = self.get_states()
    # print("states")
    # print(', '.join(f'{q:.2f}' for q in states_np))
    # print(f"type = {type(states_tensor)}")

    # test if the environment should terminate due to a collision or the
    # time horizon being met
    # done = (time_counter >= self.num_envs * (1_steps + env_params.horizon) ) #or crash)
    # done = (self.time_counter >= horizon )
    #TODO: Need to set the proper done conditions. Like grab it successfully or crash (HOW??)
    
    # compute the info for each agent
    infos = {}

    # compute the reward
    # rl_clipped = clip_actions(rl_actions)   #TODO: Make sure whether clip_actions() function is necessary
    reward, result = self.compute_reward(states_np, rl_actions)  #TODO: fix this when crash is availalbe fail=crash 

    return states_tensor, reward, result  # infos


def update(self):
    '''Update the state variables
    
    Definitions
    ----------
    piper_dof_states: 
        np.ndarray[RigidBodyState] [('pose',[('p',[x,y,z]),('r', [(x,y,z,w)])]), ('vel', [('linear',[x,y,z]),('angular', [x,y,z])])]
    cube_states:   
        np.ndarray[RigidBodyState] [('pose',[('p',[x,y,z]),('r', [(x,y,z,w)])])
    '''
    for i in range(self.num_envs):
        # self.cube_states        = self.gym.get_actor_rigid_body_states(self.envs[i],    self.cube_handles[i],   gymapi.STATE_POS) 
        self.piper_dof_states       = self.gym.get_actor_dof_states(self.envs[i],           self.piper_handles[i],  gymapi.STATE_ALL)  
        #gymapi.DofState  ([('pos', '<f4'), ('vel', '<f4')]):   piper_dof_state['pos'][i]
        self.piper_body_states      = self.gym.get_actor_rigid_body_states( self.envs[i],   self.piper_handles[i],  gymapi.STATE_ALL)  
        
        self.piper_forces           = self.gym.get_actor_dof_forces(self.envs[i], self.piper_handles[i])
        
        self.piper_velocity_target  = self.gym.get_actor_dof_velocity_targets(self.envs[i], self.piper_handles[i])
        
        # print(f"forces: {self.piper_forces}")
        
def refresh(self):
    # self.gym.refresh_rigid_body_state_tensor(self.sim)
    self.gym.refresh_dof_state_tensor(self.sim)
    # self.gym.refresh_jacobian_tensors(self.sim)
    # self.gym.refresh_mass_matrix_tensors(self.sim)
