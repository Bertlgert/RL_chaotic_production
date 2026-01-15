import numpy as np

import torch



def single_inference(env, ppo, MultiCategorical):
    
    
    actions = []
    
    ##### test inference of the model
    ##### this is designed in RLlib to really just use the modules, so no logits sampling

    episode_return = 0.0
    done = False

    # Reset the env to get the initial observation.
    obs, info = env.reset()



    module = ppo.get_module()

    module.eval()

    with torch.no_grad():

        while not done:

                
                
            
            
            
            
            ### first I have to transform each of my values in the dict to a torch tensor and add a
            ### batch dimension 
            
            
            
            for key,val in obs.items():
            
                obs[key] = torch.from_numpy(val).unsqueeze(0)
            
                
            ### bring in batch shape:
            
            batch = {"obs": obs}
            
            
            
            
            batch_out = module.forward(batch)
            
            
            logits_out = batch_out['action_dist_inputs']
            
            
            
            
            dist_object = MultiCategorical.from_logits(logits_out)
            
            ### try out regular sampling
            ### works better than deterministic sampling
            ### deterministic sampling can get stuck in some minima
            
            greedy_actions = dist_object.sample()
            
            
            greedy_actions = greedy_actions[0]
            
            actions.append(greedy_actions)
            
            
            #### remove the dummy batch dimension
            
            ### they have correct shape now
            
            
            
            ### system is only waiting
            
            #if torch.unique(greedy_actions).numel() == 1:
                
                #break
            

            # Send the action to the environment for the next step.
            obs, reward, terminated, truncated, info = env.step(greedy_actions)

            # Perform env-loop bookkeeping.
            episode_return += reward
            done = terminated
            
            
    
    
    
            
    return episode_return, actions 
    
    
    
def inference(env, ppo, MultiCategorical, number_of_single_inf = 100):
    
    
    max_return = -100
    
    best_action = []
    
    for k in range(number_of_single_inf):
    
        episode_return, action_of_single_inf = single_inference(env, ppo, MultiCategorical)
        
        if episode_return > max_return:
            
            best_action = action_of_single_inf
            
            max_return = episode_return
            
            
    print(f"Reached max episode return of {max_return}.")
    
    print(f"Best Actions taken {best_action}.")
            
            
    return max_return, best_action