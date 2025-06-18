# This python file includes the training and testing functions for the GRL model
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time

def Training_GRLModels(GRL_model, n_episodes, max_episode_len, save_dir, debug,  gym_instance, warmup, server):
    """
        This function is a training function for the GRL model

        Parameter description:
        --------
        GRL_model: the GRL model to be trained
        env: the simulation environment registered to gym
        n_episodes: the number of training rounds
        max_episode_len: the maximum number of steps to train in a single step
        save_dir: path to save the model
        warmup: model free exploration steps (randomly selected actions)
        debug: model parameters related to debugging
    """
    # The following is the model training process
    Rewards         = []  # Initialize the reward matrix for data saving
    Loss_actor      = []  # Initialize the Loss matrix for data storage
    Loss_critic     = []  # Initialize the Loss matrix for data storage
    Loss_total      = []
    Episode_Steps   = []  # Initialize the step matrix to hold the step length at task completion for each episode
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
    R_1             = 0
    i               = 1
    success_count   = 0
    success_count_1 = 0
    fail            = 0
    total_steps     = 0
    
    while i <= n_episodes :

        obs = gym_instance.init_episode()
        # previous_action = [0.0] * 6

        t_0 = gym.get_sim_time(sim)
        done    = False
        success = False
        R = 0
        t = 0
        
        while not done and not success:
            
            gym_instance.step_physics()  
            t_now = gym.get_sim_time(sim)
            
            action, prob, val = GRL_model.choose_action(obs)
            # if t % 50 == 0: print("obs", len(obs))
            detached_action = action.detach().cpu().data.numpy().astype(np.float32)
            
            obs_next, reward, success = gym_instance.step(detached_action)
            
            if t % gym_instance.debug_interval == 0: 
                    print(f"time_step= {t}")
                    # print("combined_action = ", ',    '.join(f'{q * action_scale:.2f}' for q in combined_action))
                    print("action = ", ',    '.join(f'{q:.2f}' for q in detached_action))
            if t >= max_episode_len:
                    done        = True 
                    # reward      -= 100
                    # print("fail!!")
                    
            R += reward
            
            t += 1
            total_steps += 1
            writer.add_scalar('Reward in a episode', reward, t)
            # ------Storing interaction results in PPOMemory------ #
            GRL_model.store_transition(obs, action, prob, val, reward, done)

            # ------Policy update------ #
            if total_steps % GRL_model.update_interval == 0:
                GRL_model.learn()

            # ------Observation update------ #
            obs = obs_next
            
            gym_instance.render()

        # ------ Records training data ------ #
        # Get the training data
        # loss_actor, loss_critic = GRL_model.get_statistics()
        loss_total = GRL_model.get_statistics()
        # print("loss_total", loss_total)
        # Recording training data
        Rewards.append(R)
        Episode_Steps.append(t)
        # Loss_actor.append(loss_actor)
        # Loss_critic.append(loss_critic)
        Loss_total.append(loss_total)
        # Average_Q.append(avg_q)
        
        writer.add_scalar('Reward/episode', R, i)
        writer.add_scalar('Loss_total/episode', loss_total, i)
        """writer.add_scalars('Losses per episode', {'Loss_actor': loss_actor,
                                                  'Loss_critic': loss_critic}, i)"""
        # writer.add_scalar('Loss_Critic/episode', loss_critic, i)
        
    
        if i % gym_instance.debug_interval == 0:
            # print('Training Episode:', i, 'Reward:', R, '  Loss_actor:', loss_actor, '  Loss_critic:', loss_critic, '----------#')
            print('Training Episode:', i, 'Reward:', R, '  Loss_total: ', loss_total, '----------#')
        if success:
            success_count += 1
            success_count_1 += 1
            # print('#-----SUCCESS! EPISODE:', i, 'Finished,  Reward:', R, '  Loss_actor:', loss_actor, '  Loss_critic:', loss_critic, '----------#') 
            print('#-----SUCCESS! EPISODE:', i, 'Finished,  Reward:', R, '  Loss_total:', loss_total, '----------#') 
        else:
            fail    += 1
            # print('#-----FAILED! EPISODE:', i,  'Finished,  Reward:', R, '  Loss_actor:', loss_actor, '  Loss_critic:', loss_critic, '----------#') 
            print('#-----FAILED! EPISODE:', i,  'Finished,  Reward:', R, '  Loss_total:', loss_total, '----------#') 
        
        if i != 0:
            writer.add_scalar('success rate/episode', success_count / i , i)
            print(f"#----STAT: success= {success_count}, total= {fail+success_count}, success_rate = {success_count/i}")
        else:
            writer.add_scalar('success rate/episode', 0 , i)
            print(f"#----STAT: success= {success_count}, total= {fail+success_count}")
        
        R_1 += R
        if i % 10 == 0:
            success_avg = success_count_1 / 10
            R_avg       = R_1 / 10
            writer.add_scalar('success/episode', success_avg, i)
            writer.add_scalar('avg_reward/episode', R_avg, i)
            R_1 = 0
            success_count_1 = 0
            
           
        if i % 100 == 0 and i != 0:
            # Save model
            GRL_model.save_model(save_dir)
            # Save other data
            np.save(save_dir + "/Rewards_" + str(i), Rewards)
            np.save(save_dir + "/Episode_Steps_" + str(i), Episode_Steps)
            # np.save(save_dir + "/Loss_Actor_" +  str(i), Loss_actor)
            # np.save(save_dir + "/Loss_Critic_" +  str(i), Loss_critic)
            np.save(save_dir + "/Loss_Total_" +  str(i), loss_total)
        gym_instance.reset()
        i += 1

    gym_instance.stop_simulation()
    print('Training Finished. Saved at ', save_dir)
    
    
def Testing_GRLModels(GRL_model, n_episodes, max_episode_len, load_dir, debug,  gym_instance):
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
    # Here is how the model is tested
    Rewards = [] # Initialize the reward matrix for data storage
    
    writer = SummaryWriter(os.path.join(load_dir,'logs_train'))
    gym = gym_instance.gym
    sim = gym_instance.sim
    dt  = gym_instance.dt
    GRL_model.load_model(load_dir)
    gym_instance.step_physics() 

    print("#-------------------------------------#")
    print("#-----------Testing Begins------------#")
    print("#-------------------------------------#")
    t_warmup        = 0
    R_warmup        = 0
    t               = 0
    R               = 0
    i               = 0
    success_count   = 0
    fail            = 0
    total_steps     = 0
    
    while i <= n_episodes :

        obs = gym_instance.init_episode()
        # previous_action = [0.0] * 6

        t_0 = gym.get_sim_time(sim)
        done    = False
        success = False
        R = 0
        t = 0
        
        while not done and not success:
            
            gym_instance.step_physics()  
            t_now = gym.get_sim_time(sim)
            
            action, prob, val = GRL_model.choose_action(obs)
            detached_action = action.detach().cpu().data.numpy().astype(np.float32)
            obs_next, reward, success = gym_instance.step(detached_action)
            
            if t % gym_instance.debug_interval == 0: 
                    print(f"time_step= {t}")
                    # print("combined_action = ", ',    '.join(f'{q * action_scale:.2f}' for q in combined_action))
                    print("action = ", ',    '.join(f'{q:.2f}' for q in detached_action))
            if t >= max_episode_len:
                    done        = True 
                    # reward      -= 50
                    # print("fail!!")
                    
            R += reward
            t += 1
            total_steps += 1


            # ------Observation update------ #
            obs = obs_next
            
            gym_instance.render()
            
        Rewards.append(R)
        gym_instance.reset()
        # print('Evaluation Episode:', i, 'Reward:', R)
        
        if success:
                success_count += 1
                print('#------------------SUCCESS! -------------------#')
                print('#----------EPISODE:', i, 'Finished,  Reward:', R, '----------#') 
        else:
                fail    += 1
                print('#------------------FAILED! --------------------#')
                print('#----------EPISODE:', i,  'Finished,  Reward:', R, '----------#') 
            
        print(f"#-------STAT: SUCCESS= {success_count}, TOTAL= {fail+success_count} --------#")
        
        i+=1
        
    print('Evaluation Finished')

    # Test data storage
    np.save(load_dir + "/Test_Rewards", Rewards)
