## Reinforcement Learning for Order Scheduling in Chaotic Production



This project explores reinforcement learning approaches to large-scale order scheduling in a chaotic production environment, where classical optimization methods become intractable.


The goal is to learn scheduling policies that assign subjobs to machines over time while respecting precedence constraints and limited machine availability.





### Problem Description



We consider a chaotic production setting with:


 - Hundreds of jobs, each decomposed into multiple subjobs
 - Precedence constraints between subjobs (each job forms a dependency tree / DAG)
 - Each subjob must be processed by a sequence of machines, with machine-specific processing times
 - A limited number of machines, where each machine can process only one subjob at a time



Without coordinated planning, this setting quickly leads to congestion, idle times, and deadlocks.

From an optimization perspective, this is a variant of the job-shop scheduling problem, which is NP-hard in general.


Rather than relying on handcrafted heuristics or exact optimization, this project investigates whether reinforcement learning can learn useful scheduling strategies directly from simulation.





### Environment Design



The environment simulates the execution of multiple jobs over time:


 - Subjobs become available only when their dependencies have completed
 - Machine occupancy is explicitly modeled
 - Invalid schedules (e.g. multiple subjobs assigned to the same machine) are prevented via feasibility constraint handling
 - The reward function encourages valid and efficient progress through the production process



To stabilize learning, the environment supports curriculum learning, starting from simpler scenarios and gradually increasing complexity.





### Model Architecture



The policy network is designed to handle the structured and relational nature of the problem.



**Subjob Encoding**



Each subjob is represented by:


 - Its current progress state
 - Its position within the job’s dependency structure
 - Its remaining machine sequence and processing times



Because machine sequences can be long and machine IDs are high-dimensional:


 - Learned embeddings are used for machines and durations (instead of one-hot encodings)
 - Each subjob’s machine sequence is first processed by a GRU, producing a compact representation




**Graph Neural Network (GNN)**



 - Subjobs and machines are treated as nodes in a graph
 - Edges represent dependency relations between subjobs and connect subjobs with the machines planned to use next
 - A Graph Neural Network propagates information between dependent subjobs, enabling coordination across the job structure




**Policy Head**



 - The GNN outputs node-level embeddings
 - A shared policy head (shared across subjobs) produces action logits
 - This design allows the policy to generalize across varying numbers of jobs and subjobs






### Reinforcement Learning Setup



 - Algorithm: Proximal Policy Optimization (PPO)
 - Framework: Ray RLlib
 - Action handling: Invalid schedules (e.g. machine double-occupation) are prevented via constraint handling, improving training stability
 - Training strategy: Curriculum learning to gradually increase scheduling complexity






### Current Status



**What works:**


 - Stable PPO training on simplified scheduling scenarios
 - Learning of basic, valid ordering strategies
 - Feasibility constraints significantly improve convergence compared to penalty-only approaches



**Limitations/ongoing work:**


 - Full modeling of complex subjob dependency interactions (precedence-aware optimization)
 - Optimization for makespan/time-minimization at scale
 - Extensive evaluation on large, realistic production instances
 - In this version of the code I only allow a subjob to either proceed to next machine or wait. In some szenarios more than a single machine is valid for the next step. Therefore, I will set a curriculum value for maximum output size and mask inactive machines per job/current task at the logits level



This project is intended as a research prototype, not a production-ready scheduler.


### How to Run


**1. Create the environment:**


```bash
conda env create -f environment.yml
conda activate <env-name>
```

**2. Patch RLlib source code:**

This code only runs with ray/rllib version 2.47.1 (mid juli 2025). In this version there was a bug when trying to save weights.
Within the file '/miniforge3/lib/python3.12/site-packages/ray/rllib/algorithms/algorithm.py' search for:
```python
self.env_runner_group.local_env_runner.set_weights(weights)
```
and replace it with:
```python
self.env_runner_group.local_env_runner.set_state(weights)
```

**3. Run training:**

```bash
python train_rllib_ppo.py
```
or
```bash
python -u train_rllib_ppo.py 2>&1 | tee train.log
```
in order to save to log file in parallel.


The script will start a short training run and write logs/checkpoints to the output directory.


If creating env with conda fails, please fall back to requirements.txt.








### Motivation



This project was developed as a self-directed research effort to explore:


 - Reinforcement learning for structured, combinatorial decision problems
 - Graph-based representations for production and scheduling tasks
 - The interaction between environment design, constraints, and RL stability



It complements my applied robotics and machine learning work by focusing on decision-making under complex structural constraints.





**License**



MIT Licens