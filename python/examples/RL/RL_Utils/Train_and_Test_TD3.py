# This python file includes the training and testing functions for the GRL model
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time

def Training_GRLModels(GRL_model, n_episodes, max_episode_len, save_dir, debug, gym_instance, warmup, server):
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
    
    writer = SummaryWriter(os.path.join(save_dir,'logs_train'))
    
    print("#------------------------------------#")
    print("#----------Training Begins-----------#")
    print("#------------------------------------#")

    gym = gym_instance.gym
    sim = gym_instance.sim
    dt  = gym_instance.dt
    gym_instance.step_physics() 
    action_scale = gym_instance.action_scale

    t_warmup        = 0
    R_warmup        = 0
    t               = 0
    R               = 0
    total_t         = 0
    i               = 0
    
    # obs, _, _       = gym_instance.step(None)  #obs is of np.array/ tensor
    # previous_action = [0.0] * 6
    # combined_action = [0.0] * 6
    success_count = 0
    fail    = 0
    # action = [0.0] * 6
    # t_0 = gym.get_sim_time(sim)
    # done = False
    print("#-----------------START WARMING UP-----------------#")
    
    while i <= n_episodes :
        if i == 0:
            print(f"#-----------------WARMUP STAGE-----------------#")    
        else:
            print(f"#-----------------STARTING EPISODE {i}-----------------#")  
        
        # gym_instance.step_physics()  
        # obs, _, _ = gym_instance.step(None)  #obs is of np.array
        obs = gym_instance.init_episode()
        previous_action = [0.0] * 6

        t_0 = gym.get_sim_time(sim)
        done    = False
        success = False
        R = 0
        t = 0
        # time_0 = time.time()
        time_ep = time.time()
        while not done and not success:
            gym_instance.step_physics()  
            t_now = gym.get_sim_time(sim)
            
            if t_now - t_0  >= dt:
                # loop_start =  gym.get_sim_time(sim)
                # ------Action generation------ #
                # gym_instance.step_physics()        
                # t_now = gym.get_sim_time(sim)
                
                if total_t <= warmup:         
                    action = GRL_model.choose_action_random() 
                else:
                    action = GRL_model.choose_action(obs)    
                    # print("action = ",action)
                    # print("obs = ", obs)
                    
                # combined_action = np.add(action, previous_action)
                
                if t % gym_instance.debug_interval == 0: 
                    print(f"time_step= {t}")
                    # print("combined_action = ", ',    '.join(f'{q * action_scale:.2f}' for q in combined_action))
                    print("action = ", ',    '.join(f'{q:.2f}' for q in action))
                    # print(',    '.join(f'{q:.2f}' for q in action))
                    # print("time 200 steps: ", time.time() - time_0)
                    # time_0 = time.time()

                obs_next, reward, success = gym_instance.step(action)
                

                if t >= max_episode_len:
                    done        = True 
                    reward      -= 500
                    # print("fail!!")
                
                R  += reward
                writer.add_scalar('Reward per time step', reward, t)
                # ------Storage of interaction results in the experience replay pool------ #
                GRL_model.store_transition(obs, action, reward, obs_next, done)
                
                # ------Policy update------ #
                if GRL_model.get_length() >= GRL_model.batch_size:
                    GRL_model.learn()
                # print(f"timestep TD3: {GRL_model.time_counter}")

                # ------Observation update------ #
                obs             = obs_next
                # previous_action = combined_action.copy()
                
                #-------Store previous time------#
                # t_0 = t_now
                
                # gym_instance.render()
                t       += 1
                total_t += 1
                t_0 = t_now
                # print("t = ", t)
                # elapsed =  gym.get_sim_time(sim) - loop_start
                # if elapsed < dt:
            #     time.sleep(dt - elapsed)
                
            else: 
                # if not server:   
                gym_instance.render() #This is for rendering the simulation, if you dont want to render, comment this line out
                

        previous_action = [0.0] * 6
        print("Total time ep.", i ," = ", time.time() - time_ep)
        time_ep = time.time()
                
        if i > 0:
            # ------ records training data ------ #
            # Get the training data
            loss_actor, loss_critic = GRL_model.get_statistics()
            
            # Record the training data
            Rewards.append(R)
            Episode_Steps.append(t)
            Loss_Actor.append(loss_actor)
            Loss_Critic.append(loss_critic)
            # Average_Q.append(avg_q)
            
            writer.add_scalar('Reward/episode', R, i)
            writer.add_scalar('Loss_Actor/episode', loss_actor, i)
            writer.add_scalar('Loss_Critic/episode', loss_critic, i)
            # writer.add_scalar('Avg_Q/episode', avg_q, i)
        
            if success:
                success_count += 1
                print('#-----SUCCESS! EPISODE:', i, 'Finished,  Reward:', R, '  Loss_actor:', loss_actor, '  Loss_critic:', loss_critic, '----------#') 
            else:
                fail    += 1
                print('#-----FAILED! EPISODE:', i,  'Finished,  Reward:', R, '  Loss_actor:', loss_actor, '  Loss_critic:', loss_critic, '----------#') 
            
            print(f"#----STAT: fail= {fail}, success= {success_count}, total= {fail+success_count}")
        else:
            if success:
                success_count += 1
                print('#-----SUCCESS!  Reward:', R ,'----------#') 
            else:
                fail    += 1
                print('#-----FAILED!  Reward:', R, '----------#') 
            print(f"#----STAT: fail= {fail}, success= {success_count}, total= {fail+success_count}")
        gym_instance.reset()
        # gym_instance.render()
        
        
        
        if i % 100 == 0 and i != 0:
            # Save model
            GRL_model.save_model(save_dir)
            # Save other data
            np.save(save_dir + "/Rewards_" + str(i), Rewards)
            np.save(save_dir + "/Episode_Steps_",  str(i), Episode_Steps)
            np.save(save_dir + "/Loss_Actor_" +  str(i), Loss_Actor)
            np.save(save_dir + "/Loss_Critic_" +  str(i), Loss_Critic)
            np.save(save_dir + "/Average_Q_" +  str(i), Average_Q)
        
        i = 0 if total_t < warmup else i+1
            
    
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
