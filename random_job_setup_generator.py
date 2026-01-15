
import numpy as np






def random_job_setup(number_of_subjobs = None, max_number_of_subjobs = 10,  max_seq_length = 10, max_number_machines = 10, max_duration = 10, allow_connections_between_subjobs = True):
    
    
    if number_of_subjobs == None:
    
        number_of_subjobs = int(np.round(np.random.uniform(low = 1, high = max_number_of_subjobs , size=1)).item())
        
        
        
    sequences = []
    
    task_durations = []
    
    connection_between_subjobs = []
    
    
    for k in range(number_of_subjobs):
        
        seq_length = int(np.round(np.random.uniform(low = 1, high = max_seq_length , size=1)).item())
        
        seq = np.round(np.random.uniform(low = 1, high = max_number_machines , size=seq_length)).astype(int).tolist()
        
        sequences.append(seq)
        
        
        durations = np.round(np.random.uniform(low = 1, high = max_duration , size=seq_length)).astype(int).tolist()
    
        task_durations.append(durations)
        
        
        
        
    ### recall: subjobs are counted starting with 0 as the first subjob
        
    for k in range(number_of_subjobs):
        
        
        number_of_connections = int(np.round(np.random.uniform(low = 0, high = number_of_subjobs-1 , size=1)).item())
        
        if (number_of_connections > 0) and (allow_connections_between_subjobs == True):
        
            connections = np.round(np.random.uniform(low = 0, high = number_of_subjobs-1 , size=number_of_connections))
            
            
            ### remove connections of a subjob with itself
            ### also remove double same connections
            
            connections = np.unique(connections[connections != k])
            
            print(f"Connections: {connections}")
            
            
            for g in range(len(connection_between_subjobs)):
            
                connection_np = np.array(connection_between_subjobs[g])
                
                
                connections_temp = connections.copy()
                
                for h in range(len(connections)):
                    
                    prev_job = connections[h]
                    
                    ### prev job has already been used as connection by another job
                    
                    if connection_np[connection_np != prev_job].size != connection_np.size:
                        
                        connections_temp = connections_temp[connections_temp != prev_job]
                        
                        
                    
                connections = connections_temp
                
                

                
            
            
            
        else:
            
            connections = []
            
            
        if isinstance(connections, np.ndarray) and (allow_connections_between_subjobs == True):    
            
            
            connections = connections.astype(int).tolist()
            
            
        connection_between_subjobs.append(connections)
        
        
    return sequences, task_durations, connection_between_subjobs
    
    
    
sequences, task_durations, connection_between_subjobs = random_job_setup(allow_connections_between_subjobs = True, max_duration = 1)






        
        
    
    
