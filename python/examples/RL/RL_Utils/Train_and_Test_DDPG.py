# This python file includes the training and testing functions for the GRL model
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time

def Training_GRLModels(GRL_Net, GRL_model, n_episodes, max_episode_len, save_dir, debug, gym_instance, warmup, warmup_step):
    """
        This function is a training function for the GRL model

        Parameters:
        --------
        GRL_Net: the neural network used in the GRL model
        GRL_model: the GRL model to be trained
        env: the simulation environment registered to gym
        n_episodes: number of training rounds
        max_episode_len: the maximum number of steps to train in a single step
        save_dir: path to save the model
        debug: debug-related model parameters
    """
    # The following is the model training process
    Rewards         = []  # Initialize Reward Matrix for data preservation
    Loss_Actor      = []  # Initialize the Loss matrix for data preservation
    Loss_Critic     = []  # Initialize the Loss matrix for data preservation
    Episode_Steps   = []  # Initialize the Steps matrix to hold the number of steps taken at task completion for each episode
    Average_Q       = []  # Initialize the Average Q matrix to hold the average Q value for each episode
    
    writer = SummaryWriter('logs_train')
    
    print("#------------------------------------#")
    print("#----------Training Begins-----------#")
    print("#------------------------------------#")

    gym = gym_instance.gym
    sim = gym_instance.sim
    dt  = gym_instance.dt
    gym_instance.step_physics() 

    t_warmup        = 0
    R_warmup        = 0
    t               = 0
    R               = 0
    
    obs, _, _       = gym_instance.step(None)  #obs is of np.array/ tensor
    previous_action = [0.0] * 6
    combined_action = [0.0] * 6
    success = 0
    fail    = 0
    action = [0.0] * 6
    t_0 = gym.get_sim_time(sim)
    done = False
    # obs = env.reset() #initial state
    # obs = [goal_position (3), goal_orientation (4), end_effector_position (3), end_effector_orientation (4)]
    print("#-----------------START WARMING UP-----------------#")
    time_0 = time.time()
    while t_warmup < warmup: 
        gym_instance.step_physics()
        t_now = gym.get_sim_time(sim)
        if t_now - t_0  >= dt:
            '''if t_warmup % 100 == 0:
                # actions = np.array(action) * 0.5
                gym_instance.update()
                # diff = [actions[j] - gym_instance.piper_dof_states['vel'][j] for j in range(len(actions))]
                diff = [combined_action[j]  * 0.02 - gym_instance.piper_dof_states['vel'][j] for j in range(len(combined_action))]
                # print("combined_action= ", combined_action)
                print("diff=", diff)
                print("most diff=", np.array(diff).argmax())
                print("mean=", np.array(diff).mean())'''

            
            action = GRL_model.choose_action_random()  # agent interacts with the environment, action is of tensor(size=8)
            
            combined_action = np.add(action, previous_action)
            
            
            if t_warmup % 200 == 0: print(f"combined_action = {combined_action * 0.02}")
            # if t_warmup % 1 == 0: print(f"combined_action = {action}")
            
            # print("time=",t_warmup, "action=", action)
            # print("combined action", combined_action)
            # print(f"action.shape = {len(action)}, previous_action.shape = {len(previous_action)} ")
            obs_next, reward, done = gym_instance.step(combined_action)   #obs_next, reward, done, info 
            
            t_warmup += 1
            if t_warmup % warmup_step == 0: reward -= 200
            R_warmup += reward
            
            # ------Storage of interaction results in the experience replay pool------ #
            GRL_model.store_transition(obs, action, reward, obs_next, done)
            
            # ------Policy update------ #
            if GRL_model.get_length() >= GRL_model.batch_size:
                GRL_model.learn()
            # print(f"timestep TD3: {GRL_model.time_counter}")

            # ------Observation update------ #
            obs = obs_next
            
            #-------Store previous time------#
            t_0 = t_now
            previous_action = combined_action.copy()
            
            if t_warmup % warmup_step == 0:
                print("#-----------------FAILURE-----------------#")
                fail += 1
                gym_instance.reset() 
                # combined_action = [0.0] * 6
                previous_action = [0.0] * 6
                print(f"#----STAT: fail= {fail}, success= {success}, total= {fail+success}, time={time.time() - time_0}")
                time_0 = time.time()
                
        else:
            gym_instance.render()
        # print(f"t= {t}")
        # print(f"time now= {t_now}")
        # print(f"time diff= {t_now - t_0}")
            
        if done: 
            success += 1
            print("#-----------------SUCCESS-----------------#")
            gym_instance.reset() 
            # combined_action = [0.0] * 6
            previous_action = [0.0] * 6
            done = False
            print(f"#----STAT: fail= {fail}, success= {success}, total= {fail+success}, time={time.time() - time_0}")
            time_0 = time.time()
        
            
    print("#-----------------FINISHED WARMING UP, EPISODE 1 STARTS-----------------#")
    gym_instance.reset()   
    
    for i in range(1, n_episodes + 1):
        # Print parameters in the network in real time if debugging is required
        print(f"#-----------------STARTING EPISODE {i}-----------------#")    

        if debug:
            print("#------------------------------------#")
            for parameters in GRL_Net.parameters():
                print("param:", parameters)
            print("#------------------------------------#")
            
        gym_instance.step_physics()  
        # print("phys. stepped")
        obs, _, _ = gym_instance.step(None)  #obs is of np.array
        previous_action = [0.0] * 6
        # combined_action = [0.0] * 6
        # action          = [0.0] * 6
        # print(f"obs (t = {t}) = , type={type(obs)}")
        # print("Initial obs" , ',    '.join(f'{q:.2f}' for q in obs))
        # print("None action executed")
        t_0 = gym.get_sim_time(sim)
        done = False
        # obs = env.reset() #initial state
        # obs = [goal_position (3), goal_orientation (4), end_effector_position (3), end_effector_orientation (4)]
        R = 0
        t = 0
        time_0 = time.time()
        time_ep = time.time()
        while True:
            # ------Action generation------ #
            gym_instance.step_physics()        
            t_now = gym.get_sim_time(sim)
            if t_now - t_0  >= dt:
                # print("time per step = ", t_now - t_0)
                # if t % 10 == 0:
                    # print(f"obs (t = {t}) = , type={type(obs)}")
                    # print(',    '.join(f'{q:.2f}' for q in obs))
                
                # actions = np.array(action) * 0.5
                
                # diff = [actions[j] - gym_instance.piper_dof_states['vel'][j] for j in range(len(actions))]
                '''if t % 100 == 0:
                    gym_instance.update()
                    diff = [combined_action[j]* 0.02 - gym_instance.piper_dof_states['vel'][j] for j in range(len(combined_action))]
                    # print("combined_action= ", combined_action)
                    print("diff=", diff)
                    print("most diff=", np.array(diff).argmax())
                    print("mean=", np.array(diff).mean())'''
                
                action = GRL_model.choose_action(obs)  # agent interacts with the environment, action is of tensor(size=8)
                combined_action = np.add(action, previous_action)
                
                if t % 200 == 0: 
                    print(f"time_step= {t}, combined_action = {combined_action * 0.02}")
                    print(f"action = {action}")
                    print("time 200 steps: ", time.time() - time_0)
                    time_0 = time.time()
                # print("action", combined_action)
                # print(',    '.join(f'{q:.2f}' for q in action))
                obs_next, reward, done = gym_instance.step(combined_action)   #obs_next, reward, done, info 
                # G1, A1
                
                R += reward
                t += 1
                if t >= max_episode_len: reward -= 200
                # ------Storage of interaction results in the experience replay pool------ #
                GRL_model.store_transition(obs, action, reward, obs_next, done)
                
                # ------Policy update------ #
                GRL_model.learn()
                # print(f"timestep TD3: {GRL_model.time_counter}")

                # ------Observation update------ #
                obs             = obs_next
                previous_action = combined_action.copy()
                
                #-------Store previous time------#
                t_0 = t_now
            else:    
                gym_instance.render()
            # print(f"t= {t}")
            # print(f"time now= {t_now}")
            # print(f"time diff= {t_now - t_0}")
            
            if t >= max_episode_len or done:
                combined_action = [0.0] * 6
                previous_action = [0.0] * 6
                break
            # reset = t == max_episode_len
            # if reset or done:
            #     break
        print("Total time ep.", i ," = ", time.time() - time_ep)
        # ------ records training data ------ #
        # Get the training data
        loss_actor, loss_critic = GRL_model.get_statistics()
        # Record the training data
        Rewards.append(R)
        Episode_Steps.append(t)
        Loss_Actor.append(loss_actor)
        Loss_Critic.append(loss_critic)
        # Average_Q.append(avg_q)
        
        if done:
            success += 1
            print('#-----SUCCESS! EPISODE:', i, 'Finished,  Reward:', R, '  Loss_actor:', loss_actor, '  Loss_critic:', loss_critic, '----------#') 
        else:
            fail    += 1
            print('#-----FAILED! EPISODE:', i,  'Finished,  Reward:', R, '  Loss_actor:', loss_actor, '  Loss_critic:', loss_critic, '----------#') 
        writer.add_scalar('Reward/episode', R, i)
        writer.add_scalar('Loss_Actor/episode', loss_actor, i)
        writer.add_scalar('Loss_Critic/episode', loss_critic, i)
        # writer.add_scalar('Avg_Q/episode', avg_q, i)
        
        # print("EP finished, obs=", obs)
        gym_instance.reset()
        # print("After reset, obs=", obs)
        # gym_instance.render()
        
        print(f"#----STAT: fail= {fail}, success= {success}, total= {fail+success}")
        
        if i % 50 == 0:
            # Save model
            GRL_model.save_model(save_dir)
            # Save other data
            np.save(save_dir + "/Rewards_" + str(i), Rewards)
            np.save(save_dir + "/Episode_Steps_",  str(i), Episode_Steps)
            np.save(save_dir + "/Loss_Actor_" +  str(i), Loss_Actor)
            np.save(save_dir + "/Loss_Critic_" +  str(i), Loss_Critic)
            np.save(save_dir + "/Average_Q_" +  str(i), Average_Q)
            
    
    print('#-----------------TRAINING FINISHED-----------------#')

    # Save model
    GRL_model.save_model(save_dir)
    # Save other data
    np.save(save_dir + "/Rewards", Rewards)
    np.save(save_dir + "/Episode_Steps", Episode_Steps)
    np.save(save_dir + "/Loss_Actor_" +  str(i), Loss_Actor)
    np.save(save_dir + "/Loss_Critic_" +  str(i), Loss_Critic)
    np.save(save_dir + "/Average_Q", Average_Q)
    
    gym_instance.stop_simulation()


def Testing_GRLModels(GRL_Net, GRL_model, n_episodes, max_episode_len, load_dir, debug, gym_instance):
    """
        This function is a test function for a trained GRL model

        Parameters:
        --------
        GRL_Net: the neural network used in the GRL model
        GRL_model: the GRL model to be tested
        env: the simulation environment registered to gym
        test_episodes: the number of rounds to be tested
        load_dir: path to read the model
        debug: debug-related model parameters
    """
    # The following is the model testing process
    Rewards = []
    writer = SummaryWriter('logs_test')
    gym = gym_instance.gym
    sim = gym_instance.sim
    dt  = gym_instance.dt
    GRL_model.load_model(load_dir)

    print("#-------------------------------------#")
    print("#-----------Testing Begins------------#")
    print("#-------------------------------------#")
    
    
    for i in range(1, n_episodes + 1):
        obs, _, _ = gym_instance.step(None)  #obs is of np.array
        t_0 = gym.get_sim_time(sim)
        done = False
        # obs = env.reset() #initial state
        # obs = [goal_position (3), goal_orientation (4), end_effector_position (3), end_effector_orientation (4)]
        R = 0
        t = 0
        while True:
            # ------Action generation------ #
            gym_instance.step_physics()
            t_now = gym.get_sim_time(sim)
            if t_now - t_0  >= dt:
                # action = GRL_model.choose_action(obs)  # agent interacts with the environment, action is of tensor(size=8)
                action = GRL_model.test_action(obs)  # agent interacts with the environment, action is of tensor(size=8)
                # print(action)
                obs, reward, done = gym_instance.step(action)   #obs_next, reward, done, info 

                R += reward
                t += 1
                
                #-------Store previous time------#
                t_0 = t_now
            gym_instance.render()
            reset = t == max_episode_len
            if reset or done:
                break
        Rewards.append(R)
        if done:
            print('#-----SUCCESS! EPISODE:', i, 'Finished,  Reward:', R, '----------#') 
        else:
            print('#-----FAILED! EPISODE:', i,  'Finished,  Reward:', R, '----------#') 
        writer.add_scalar('Reward per episode', R, i)
        gym_instance.reset()
        
    # gym_instance.reset()
    print('#-----------------TESTING FINISHED-----------------#')


    # Test data storage
    np.save(load_dir + "/Test_Rewards", Rewards)
