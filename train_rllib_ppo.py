

import os

import sys

#os.environ["CUDA_VISIBLE_DEVICES"] = ""


print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))


import torch



from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.torch import TorchRLModule

from ray.rllib.models.torch.torch_distributions import TorchMultiCategorical

from ray.rllib.models.distributions import Distribution
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import MAX_LOG_NN_OUTPUT, MIN_LOG_NN_OUTPUT, SMALL_NUMBER
from ray.rllib.utils.typing import TensorType, Union, Tuple
import gymnasium as gym
from ray.rllib.models.torch.torch_distributions import TorchCategorical

from torch.distributions import Categorical

from ray.rllib.core.rl_module.apis import ValueFunctionAPI

import torch.nn as nn



from gymnasium import spaces
import numpy as np
from typing import Optional





import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env
import ray

import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear, LayerNorm

from torch_geometric.data import HeteroData

from torch.nn.utils.rnn import pack_padded_sequence


from torch_geometric.data import Batch 

import json

from pathlib import Path

import pickle

import math


import time

import re

from ray.rllib.algorithms.callbacks import DefaultCallbacks



#### now my own files

from inference import inference

from random_job_setup_generator import random_job_setup

## set up root path 

REPO_PATH = Path(__file__).resolve().parent

print("Repo path: ", REPO_PATH)





class CriticStats(DefaultCallbacks):
    def on_learn_on_batch(self, *, policy, train_batch, result, **kwargs):
        if "vf_preds" in train_batch and "value_targets" in train_batch:
            V = train_batch["vf_preds"]
            T = train_batch["value_targets"]
            diff = V - T
            result["critic/var_targets"] = float(np.var(T))
            result["critic/mae"] = float(np.mean(np.abs(diff)))
            result["critic/bias"] = float(np.mean(diff))
            
        else:
            print("Error accessing value targets!")

print("worked")
input_length = [1]

def get_input_lens():
    return input_length
    
    
    
### start with embeddings that are not random anymore. I want good destinctivity

### I use positional embedding from transformer paper "Attention is all you need"
    
def sinusoidal_embeddings(n, d, scale = 1.0):
    pe = torch.zeros(n, d)
    position = torch.arange(0, n).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d, 2) * -(math.log(10000.0) / d)) * scale
    
    ### use broadcating: new_matr[i,j] = position[i] * div_term[j]
    
    pe[:, 0::2] = torch.sin(position * div_term)
    
    
    ### depending on whether embed dim is odd or even there will be different number of collumns
    ### in div term 0::2 works below as this is same as (0,d,2)
      
    if d % 2 == 0:
    
        pe[:, 1::2] = torch.cos(position * div_term)
        
        
    ## skip the last collumn
    
    else:
    
        pe[:, 1::2] = torch.cos(position * div_term[-1])
        
    
    return pe


@DeveloperAPI
class MultiCategorical(Distribution):                                           ### all lives on gpu or the current device
    """MultiCategorical distribution for MultiDiscrete action spaces."""
    """MultiCategorical distribution for MultiDiscrete action spaces."""
    
    input_lens = None
    
    current_device = None

    @override(Distribution)
    def __init__(self, categoricals: list):
        super().__init__()
        self._cats = categoricals
        
        
        
        
        
    def set_input_lens(input_lens = get_input_lens(), current_device = "cpu"):
    
        MultiCategorical.input_lens = input_lens
        
        MultiCategorical.current_device = current_device
        
        
    def turn_into_tensor():
        
        MultiCategorical.input_lens = torch.tensor(MultiCategorical.input_lens, device = MultiCategorical.current_device)
        

    @override(Distribution)
    def sample(self) -> TensorType:
        # Sample from each categorical and stack the results
        arr = [cat.sample() for cat in self._cats]
        sample_ = torch.stack(arr, dim=-1)
        return sample_

    @override(Distribution)
    def rsample(self, sample_shape=()):
        # Reparameterized sampling
        arr = [cat.rsample() for cat in self._cats]
        sample_ = torch.stack(arr, dim=-1)
        return sample_

    @override(Distribution)
    def logp(self, value: TensorType) -> TensorType:
    
        
        
        # Compute the log-probability for each categorical and sum them
        value = torch.unbind(value, dim=-1)
        logps = torch.stack([cat.logp(act) for cat, act in zip(self._cats, value)])
        
        
        return torch.sum(logps, dim=0)

    @override(Distribution)
    def entropy(self) -> TensorType:
        # Compute the entropy of each categorical and sum them
        return torch.sum(
            torch.stack([cat.entropy() for cat in self._cats], dim=-1), dim=-1
        )

    @override(Distribution)
    def kl(self, other: Distribution) -> TensorType:
        # Compute the KL divergence between two distributions
        kls = torch.stack(
            [cat.kl(oth_cat) for cat, oth_cat in zip(self._cats, other._cats)],
            dim=-1,
        )
        return torch.sum(kls, dim=-1)

    @staticmethod
    @override(Distribution)
    def required_input_dim(space: gym.Space, **kwargs) -> int:
        """Returns the required input dimension for this action space."""
        assert isinstance(space, gym.spaces.MultiDiscrete)
        return int(torch.sum(space.nvec))

    @classmethod
    @override(Distribution)
    def from_logits(
        cls,
        logits: TensorType,
        temperatures: list = None,
        **kwargs,
    ) -> "TorchMultiCategorical":
        """Creates this Distribution from logits (and additional arguments)."""

        
        
        
        
        
        MultiCategorical.current_device = logits.device
        
        
        input_lens = torch.tensor(cls.input_lens, device = MultiCategorical.current_device)

        
            

        

        # Check if the sum of input_lens matches the last dimension of logits
        
        

        # Split the logits into categorical distributions based on input_lens
        categoricals = [
            TorchCategorical(logits=logits[..., start:start + length])
            for start, length in zip(torch.cumsum(torch.cat([torch.tensor([0], device= MultiCategorical.current_device) , input_lens[:-1]]),0), input_lens)
        ]

        return cls(categoricals=categoricals)

    def to_deterministic(self) -> "TorchDeterministic":
        """Converts the distribution to a deterministic (argmax) action."""
        if self._cats[0].probs is not None:
            probs_or_logits = torch.stack(
                [cat.logits for cat in self._cats], dim=-1
            )
        else:
            probs_or_logits = torch.stack(
                [cat.logits for cat in self._cats], dim=-1
            )
            
        

        return TorchDeterministic(loc=torch.argmax(probs_or_logits, dim=-1))
        
        
    def deterministic_sample(self) -> "TorchDeterministic":
        """Converts the distribution to a deterministic (argmax) action."""
        if self._cats[0].probs is not None:
            probs_or_logits = torch.stack(
                [cat.logits for cat in self._cats], dim=-1
            )
        else:
            probs_or_logits = torch.stack(
                [cat.logits for cat in self._cats], dim=-1
            )
            
        

        return torch.argmax(probs_or_logits, dim=-2)


#### dont use residuals in GRU as we use packed sequences. Makes it really hard to use!

class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, bidirectional=False, num_layers = 2):
        super().__init__()
        
        
        
        self.num_layers = num_layers
        
        if bidirectional == True:
            
            D = 2
            
        else:
            
            D = 1
        
        ### I used default with a single layer.
        
        self.gru1 = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        self.bidirectional = bidirectional
        
        self.norm1 = LayerNorm(hidden_dim*D)
        
        if self.num_layers == 2:
        
            self.gru2 = nn.GRU(hidden_dim * D, hidden_dim, batch_first=True, bidirectional=bidirectional)
        
        #self.norm2 = LayerNorm(hidden_dim*D)
        
        

    def forward(self, padded_sequences, sequence_lengths, init_state):
        # Sort sequences by length (required for packing)
        ### important: we use here pytorch sort(), not sorting of lists, so this works with tensors!
        ### padded_sequences is represented as an embedding and is concatenated with the duration for each task
        ### the padded_sequence is of shape (num_sub_jobs, padded_sequence_length, embed_dim)
        ### sequence_lenght is of shape (num_sub_jobs,)
        ### sequence lengths should not include info about embedding dim, they should just
        ### how far into each sequence data is still actually relevant
        
        sequence_lengths, perm_idx = sequence_lengths.sort(descending=True)
        padded_sequences = padded_sequences[perm_idx]
        
        
        
        
        #### pack_padded_sequence takes care of skipping padded parts of the sequence as I give it the unpadded sequence
        #### lengths

        packed = pack_padded_sequence(
            padded_sequences, sequence_lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        
        
        
        outputs, hidden = self.gru1(packed, init_state)
        

        
        
        
        
        if self.num_layers == 2:
      
            
            outputs, hidden = self.gru2(outputs)
        
            
        
        
        
        ### if bidirectional combine forward and backward output
        ### else take the last real hidden output

        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]  # shape: [num_sub_jobs, hidden_dim]
            
            

            
        hidden = self.norm1(hidden)
        
        
        

            
            
        ### apply norm after performing cat
            
            
        ### important: a GRU should not be directly be followed by a relu, as there is already non linearity in there. could follow by linear, then relu if
        ### I need extra flexibility

        # Undo sorting to return in original order
        _, unperm_idx = perm_idx.sort()
        hidden = hidden[unperm_idx]                                                                         ### are on gpu

        return hidden  # shape is [num_sub_jobs, hidden_dim] or [num_sub_jobs, hidden_dim * 2]



class customGNN(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, num_layers = 2, GRU_bidirectional = False):
        super().__init__()

        # Define one layer of message passing per edge type
        ###let the SAGEConv layer determine the input and output dimensions
        
        self.num_layers = num_layers
        
        if GRU_bidirectional == False:
        
            subjob_dim = hidden_dim + 2*embedding_dim
            
        elif GRU_bidirectional == True:
            
            subjob_dim = 2*hidden_dim + 2*embedding_dim
        
        machine_dim = embedding_dim
        
        node_types = ['subjob','machine']
        
        ### message passing layer with 3 edge types
        
        self.conv1 = HeteroConv({
            ('subjob', 'depends_on', 'subjob'): SAGEConv((subjob_dim, subjob_dim), hidden_dim),
            ('subjob', 'uses', 'machine'): SAGEConv((subjob_dim, machine_dim), hidden_dim),
            ('machine', 'used_by', 'subjob'): SAGEConv((machine_dim, subjob_dim), hidden_dim ),
        }, aggr='sum')  # Combine messages from different edge types

        self.linear1 = Linear(hidden_dim, hidden_dim)
        
        self.norms1 = torch.nn.ModuleDict({
        
            node_type: LayerNorm(hidden_dim)
            for node_type in node_types
        
        })
        
        #self.norm_lin1 = LayerNorm(hidden_dim)
        
        if self.num_layers == 2:
        
        
            self.conv2 = HeteroConv({
                ('subjob', 'depends_on', 'subjob'): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                ('subjob', 'uses', 'machine'): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                ('machine', 'used_by', 'subjob'): SAGEConv((hidden_dim, hidden_dim), hidden_dim ),
            }, aggr='sum')  # Combine messages from different edge types

            self.linear2 = Linear(hidden_dim, hidden_dim)
            
            self.norm2 = LayerNorm(hidden_dim)
            
            self.norm3 = LayerNorm(hidden_dim)
            
            self.norm4 = LayerNorm(hidden_dim)
        
        #self.norm_lin2 = LayerNorm(hidden_dim)
        
        ### x_dict is a dict containing the nodes and the corresponding embeddings, one per subjob and machine/location
        ### edge_index_dict contains info about how edges are connected, so which subjobs depend on which or which 
        ### machines want to be used

    def forward(self, x_dict, edge_index_dict):                                                                                  ### are on gpu
        
        
        x_dict = self.conv1(x_dict, edge_index_dict)
        
        ### first normalise
        
        for n_type, x in x_dict.items():
        
            x_dict[n_type] = self.norms1[n_type](x)
        
        ### now relu 
        
        for n_type, x in x_dict.items():
            
            x_dict[n_type] = F.relu(x)
            
        ### now additional linear with relu, normalizing here is optional
        
        x_dict['subjob'] = F.relu(self.norm4(self.linear1(x_dict['subjob'])))
        
        
        
        ### now second GNN layer
        
        if self.num_layers == 2:
            
            ### create a res object
            
            x_res = {}
            
            for n_type, x in x_dict.items():
            
                x_res[n_type] = x.clone()
            
            
            
            x_dict = self.conv2(x_dict, edge_index_dict)
            
            ### first normalise
            
            x_dict['subjob'] = self.norm2(x_dict['subjob'])
            
            ### now relu 
                
            x_dict['subjob'] = F.relu(x_dict['subjob'])
                
            ### now additional linear with relu, normalizing here is optional, but we do as this will be part of output
            
            x_dict['subjob'] = F.relu(self.norm3(self.linear2(x_dict['subjob'])))
            
            
            ### now add residual info, to connect the two layers efficiently!
            
            for n_type, x in x_dict.items():
            
                x_dict[n_type] = x_dict[n_type] + x_res[n_type]
            
            
        
        
        
        
        
        return x_dict  # contains updated node embeddings



class myModule(TorchRLModule, ValueFunctionAPI):


    def setup(self):
    
        super().setup()

            
          
        
        self.hidden_dim = self.model_config["hidden_dim"]
        
        self.number_sub_jobs = self.model_config["num_objects"]
        
        self.padded_seq_length = self.model_config["seq_pad_length"]
        
        
        self.freeze_embeddings = self.model_config["freeze_embeddings"]
        
        self.freeze_gru = self.model_config["freeze_gru"]
        
        
        
        machines_embed_dim = ((self.model_config["cur_learning_max_num_locations"]+1),self.model_config["embedding_dim_machines"])
        
        dur_embed_dim = ((self.model_config["cur_learning_max_dur"]+1),self.model_config["embedding_dim_durations"])
        
        
        ### this is the true sequenth length, since progress wont move beyond the true val
        
        progress_embed_dim = ((self.model_config["cur_learning_max_true_seq_length"]+1),self.model_config["embedding_dim_progres"])


        ### first initialize embeddings randomly

        self.embedding_machines = nn.Embedding(*machines_embed_dim)
       
        
        
        self.embedding_durations = nn.Embedding(*dur_embed_dim)
        
        
        
        
        
        self.embedding_progres = nn.Embedding(*progress_embed_dim)
        
        
        
        #### now freeze embedding rows that are unused during current curriculum cycle 
        
        
        ### rows are also counted starting from zero!
        
        ### I need frozen rows machines to freeze rows of the policy head down the line, not just regarding embeddings!
        
        frozen_rows_machines = list(range(self.model_config["num_locations"]+1,self.model_config["cur_learning_max_num_locations"]+1))
        
        
        
        if self.freeze_embeddings == False:
            
            use_hook = True

        
            
            
            frozen_rows_durations = list(range(self.model_config["max_duration"]+1,self.model_config["cur_learning_max_dur"]+1))
            
            frozen_rows_progress = list(range(self.model_config["max_seq_length"]+1,self.model_config["cur_learning_max_true_seq_length"]+1))
            
            
        else:
            
            use_hook = False
            
            print()
            print("All embeddings are frozen during this training cycle!!")
            print()
            
            ### freeze all parameters explicitly. Afterwards no hook is required anymore!
            
            for para in self.embedding_machines.parameters():
                
                para.requires_grad = False
                
                
            for para in self.embedding_durations.parameters():
                
                para.requires_grad = False   
                

            for para in self.embedding_progres.parameters():
                
                para.requires_grad = False
                
                
            

            
        
        if use_hook == True:
        
        
            if self.model_config["num_locations"] < self.model_config["cur_learning_max_num_locations"]:
            
                frozen_rows_tensor_machines = torch.tensor(frozen_rows_machines, device=self.embedding_machines.weight.device)
                
                def freeze_rows_hook_machine(grad):
                    grad[frozen_rows_tensor_machines] = 0
                    return grad
                self.embedding_machines.weight.register_hook(freeze_rows_hook_machine)
                
                
            
            if self.model_config["max_duration"] < self.model_config["cur_learning_max_dur"]:
                
                
                frozen_rows_tensor_dur = torch.tensor(frozen_rows_durations, device=self.embedding_durations.weight.device)
                
                def freeze_rows_hook_dur(grad):
                    grad[frozen_rows_tensor_dur] = 0
                    return grad
                self.embedding_durations.weight.register_hook(freeze_rows_hook_dur)
                
                
                

            if self.model_config["max_seq_length"] < self.model_config["cur_learning_max_true_seq_length"]:
                
                frozen_rows_tensor_progres = torch.tensor(frozen_rows_progress, device=self.embedding_progres.weight.device)
                
                def freeze_rows_hook_progress(grad):
                    grad[frozen_rows_tensor_progres] = 0
                    return grad
                self.embedding_progres.weight.register_hook(freeze_rows_hook_progress)
            
        
        """
        #### now apply sinousodal weights for good distinctiveness
        
        self.embedding_machines.weight.data = sinusoidal_embeddings(*machines_embed_dim, scale = 1.0)
        self.embedding_durations.weight.data = sinusoidal_embeddings(*dur_embed_dim, scale = 1.3)
        self.embedding_progres.weight.data = sinusoidal_embeddings(*progress_embed_dim, scale = 2.1)
        
        """
        
        
        
        
        
        ### first define a RNN to encode the sequences with
        
        
        gru_input_dim = self.model_config["embedding_dim_machines"] + self.model_config["embedding_dim_durations"]
        
        self.gru = GRUEncoder(gru_input_dim,self.hidden_dim, bidirectional = self.model_config["GRU_bidirectional"], num_layers = self.model_config["number_GRU_layers"])
        
        
        ### freeze GRU parameters
        
        if self.freeze_gru == True:
            
            print()
            print("Freezing GRU during this training cycle!!")
            print()
            
            for para in self.gru.parameters():
                
                para.requires_grad = False            
        
        
        #### now define a GNN to pass output of RNN+additional features to
        
        
        self.GNN = customGNN(self.hidden_dim,self.model_config["embedding_dim_machines"], num_layers = self.model_config["number_GNN_layers"], GRU_bidirectional = self.model_config["GRU_bidirectional"])
        
        
                     
            
        self.norm_policy_head = LayerNorm(self.hidden_dim)
        
        
        
        ### policy head to apply in order to get logits
        
        self.policy_head = nn.Sequential(
        nn.Linear(self.hidden_dim, self.hidden_dim),
        self.norm_policy_head,
        nn.ReLU(),
        
                )
                
        ### important: the final policy layer as well as the value head must not get normalized or relud. Otherwise this will limit expressivity!!
        
        

        self.final_policy_head = torch.nn.Linear(self.hidden_dim, self.model_config["cur_learning_max_outputs"])
            

            
            
            
            
            
            
        if self.model_config["action_masking"] == False:
        
        
            ### freeze the output rows that are currently unused

            if self.model_config["num_locations"] < self.model_config["cur_learning_max_num_locations"]:
            
                frozen_rows_tensor_machines = torch.tensor(frozen_rows_machines, device=self.final_policy_head.weight.device)
                
                def freeze_rows_hook_machine_policy(grad):
                    grad[frozen_rows_tensor_machines] = 0
                    return grad
                self.final_policy_head.weight.register_hook(freeze_rows_hook_machine_policy) 
        
        #### value head to use in order to get value preds
        #### 2 times hidden dim due to mean max pooling
        
        self.value_head = torch.nn.Linear(2 * self.hidden_dim, 1)
        
        


        #### use specific distribution for MultiDiscrete
        
        self.action_dist_cls = MultiCategorical
        
        
        
        ### Wrap the model in torch.compile for speedup
        
        #self = torch.compile(self, dynamic = True, mode= "max-autotune")
        
        
        
        
        



    @property
    def current_device(self):
        return next(self.parameters()).device
        
        
        
    def forward_part_shared_by_forward_and_value_function(self, batch, **kwargs):
        
        

        current_device = self.current_device
        
        

        # Get the batch of observations (usually the first key in the batch)
        
        
        
        observations = batch[Columns.OBS]       ## are on gpu
        
        
        
        
        
    
        # The batch size is the size of the first dimension of a tensor in observations
        ## observations is a dict of batched tensors -> choose one to get batch size
        
        batch_size = observations["sequences"].shape[0]  # This gives the number of experiences in the batch ### plain int ok to use, neither on cpu or gpu
        
        
        
        ### just access entries of the observation dict as always, just be aware that it gives me a 
        ### batch of these entries
        
        
        
        ##### now take padded sequences and padded durations from the observations
        
        seq_padded = observations["sequences"].clone()                          ### are on gpu
        
        task_durations_padded = observations["task_durations"].clone()          ### are on gpu
        
        
        
        
        
        ### first use embedding representation
        ### Important: legs that we want to fuse have to be adjacent and in the order prior to permuting
        ### shape prior to reshping: (batch_size,num_sub_jobs, padded_seq_length) -> is correct for fusing 
        
        embs_sequ = self.embedding_machines(seq_padded.view(batch_size * self.number_sub_jobs * self.padded_seq_length))                    ### are on gpu
        
        
        embs_dur = self.embedding_durations(task_durations_padded.view(batch_size * self.number_sub_jobs * self.padded_seq_length))         ### are on gpu
        
        
        
        
        
        #### now reshape into expected shape
        
        #### this reshaping is correct, because embs_sequ has 
        ### shape (batch_size * self.number_sub_jobs * self.padded_seq_length, embed dims)
        
        embs_sequ = embs_sequ.view(batch_size,self.number_sub_jobs, self.padded_seq_length,self.model_config["embedding_dim_machines"])     ### are on gpu
        
        
        
        embs_dur = embs_dur.view(batch_size,self.number_sub_jobs, self.padded_seq_length,self.model_config["embedding_dim_durations"])      ### are on gpu
        
        #### now cat in order to apply RNN afterwards. cat along the embed dim
        
        
        embs_concated = torch.cat([embs_sequ,embs_dur],dim=-1)                                                                              ### are on gpu            
        
        
        
        ### now reshape and put through GRU. The correct legs are adjacent and order is kept
        
        embs_concated_flattened = embs_concated.view(batch_size*self.number_sub_jobs,self.padded_seq_length,embs_concated.shape[-1])        ### are on gpu
        
        
        ### the leg order is kept when reshaping and they are adjacent. Is the same order as embs_flattened
        ### which we need
        
        true_seq_length = observations['true_seq_length'].clone()                                                                           ### are on gpu    
        
        true_seq_length_flattened = true_seq_length.view(batch_size*self.number_sub_jobs,)                                                  ### are on gpu
        
        
        
        ### init state of gru is the same for each env step, default it to zeros of size (num_layers, batch_size * number_sub_jobs, hidden_size)
        ### number of layers is 1 by default
        
        if self.model_config["GRU_bidirectional"] == False:
            
            D = 1
            
        elif self.model_config["GRU_bidirectional"] == True:
            
            D = 2
        
        init_state = torch.zeros(D*1, batch_size*self.number_sub_jobs, self.hidden_dim, device = current_device)
        
        
        gru_output_flattened = self.gru(embs_concated_flattened, true_seq_length_flattened, init_state)                                                 ### are on gpu
        
        
        
        ### reshape back. Is ok as order of legs is kept and are adjacent. 
        ### GRU Output flattened has size (batch_size * self.number_sub_jobs, hidden dim)
        
        gru_output = gru_output_flattened.view(batch_size,self.number_sub_jobs,self.hidden_dim*D)                                             ### are on gpu
        
        
        
        ### now generate dynamic data the GNN requires
        
        ### important: number of machines is independent of batch size
        ### First give GNN the embedding of each machine
        
        
        
        
        ### next we will use PyGs internal batching procedure. Is flexible with respect to
        ### varying sizes per batch_item
        
        
        ### however I need to append everything in a list per batch
        
        gru_output_list = []
        
        for k in range(batch_size):
        
            gru_output_list.append(gru_output[k])   

            
        
        
        
        
        ### machine features has size (num_machines, embed dim)
        ### also include waiting as a feature, so machine zero
        
        machine_ids = torch.arange(self.model_config["num_locations"]+1, device = current_device)                                             ### put ids on gpu 
        
        
        
        
        
        machine_features = self.embedding_machines(machine_ids)
        
        ### important: we will input to the GNN a flat matrix with node features
        ### -> it will see batch_size*num_sub_jobs many subjobs and label them with [0...batch_size*num_sub_jobs-1]
        ### therefore I need to change labeling in the edge_index tensors accordingly
        
        
        
        edge_indices_subjob_dependencies = observations["dependency_edges_between_subjobs"].clone()                                         ### are on gpu
        
        
        
        
        
        edge_indices_subjob_dependencies_list = []
        
        for k in range(batch_size):
        
            edge_indices_subjob_dependencies_list.append(edge_indices_subjob_dependencies[k])                                               ### are on gpu
                                                                                                                                    ### list of tensors is fine
        
        
        
        
        
        ### important: right now I use the same machines for the entire batch,  
        
        
        
 


        next_locations_batched = observations['next_task_id'].clone()                                                           ### are on gpu
        
        
        
        edge_indices_manual = []
        
        for k in range(batch_size):
        
            next_locations = next_locations_batched[k,:]                                                                        ### are on gpu
        
            dependency_edges_to_tasks = torch.tensor([[],[]], device = current_device)
            
            


            ### check each subjob
            ### subjobs get counted starting with 0

            for i in range(self.number_sub_jobs):

                
                ### we also allow connections to a machine defined as waiting, which can be used in parallel by as many jobs as required!!    
                
                i_tens = torch.tensor(i, device = current_device)
                
                
                
                single_dependency = torch.stack([i_tens,next_locations[i]],dim=0).unsqueeze(1)
                
                
                
                
        
                dependency_edges_to_tasks = torch.cat((dependency_edges_to_tasks,single_dependency),dim=-1)


            edge_indices_manual.append(dependency_edges_to_tasks.clone())                                                   ### are on gpu
            
        
        
        edge_indices_tasks_list = edge_indices_manual
        
        

        
        
        ### important: when using Batch.from_data_list later on PyG automatically offsets edge_indices 
        ### in order to work correctly. So no manual offset required here.
        
        
        ##### also incorporate the current progress into subjob features
        
        current_progress = observations['progress_for_all_subjobs'].clone()
        
        ### now reshape to use with embeddings
        
        current_progress = current_progress.view(batch_size * self.number_sub_jobs,2)
        
        current_progress_in_seq = current_progress[:,0]                                     ### are on gpu
        
        
        
        current_progress_in_each_task = current_progress[:,1]
        
        
        ### now use embeddings 
        
        current_progress_in_seq_embs = self.embedding_progres(current_progress_in_seq)
        
        
        current_progress_in_each_task_embs = self.embedding_durations(current_progress_in_each_task)
        
        
        ### now reshape before using cat
        
        current_progress_in_seq_embs = current_progress_in_seq_embs.view(batch_size,self.number_sub_jobs,current_progress_in_seq_embs.shape[-1])
        
        
        current_progress_in_each_task_embs = current_progress_in_each_task_embs.view(batch_size,self.number_sub_jobs,current_progress_in_each_task_embs.shape[-1])
        
        
        ### now cat those features
        
        additional_subjob_features = torch.cat([current_progress_in_seq_embs,current_progress_in_each_task_embs],dim=-1)
        

        
        
        data_list = []
        
        
        for k in range(batch_size):
        
            data = HeteroData()

            # Add nodes
            data['subjob'].x = torch.cat([gru_output_list[k],additional_subjob_features[k]],dim=-1)         # [N_subjobs, rnn_dim+add_features]
            
            data['machine'].x = machine_features       # [num_machines, feature_dim]

            # Add edges
                                       
            
            ### indices are expected to be given as int64

            data['subjob', 'depends_on', 'subjob'].edge_index = edge_indices_subjob_dependencies_list[k].long()  
            data['subjob', 'uses', 'machine'].edge_index = edge_indices_tasks_list[k].long()    

            ### pytorch style indexing. Does switch rows 0 and 1
            
            data['machine', 'used_by', 'subjob'].edge_index = edge_indices_tasks_list[k][[1, 0]].long() 
            
            
            
            
            
            data_list.append(data)
            
        
            
            
        batch = Batch.from_data_list(data_list)
        
        ### ensure batch is on gpu
        
        batch = batch.to(current_device, non_blocking = True)
        
      
  
        
        
        #### now put through GNN
        
        x_dict = batch.x_dict
        
        edge_index_dict = batch.edge_index_dict
        
        
        updated_x = self.GNN(x_dict, edge_index_dict)
        
        
        
        
        ### Get updated subjob representations
        
        subjob_repres = updated_x['subjob']                     ### are on gpu
        
        
        
        
        
        
        
        
        ### put this through a shared policy head, gives size: (batch * num_sub_jobs,hidden_dim)
        
        pre_logits = self.policy_head(subjob_repres)
        
        
        return pre_logits, batch_size
        
    ##### in RLlib the data is stored as batches (actions,observations,rewards, etc) from several steps or multiple envs
    ### collumns is used to access this data in batch using keywords like OBS
    ### since action space is descrete, the default distribution used on ACTION_DIST_INPUTS is also categorical
    
    def _forward(self, batch, **kwargs):


        pre_logits, batch_size = self.forward_part_shared_by_forward_and_value_function(batch)
        
        
        
        ### now apply final policy head to get logits, gives size: (batch * num_sub_jobs,curriculum_max_num_locations +1)
        
        
        logits = self.final_policy_head(pre_logits)
        
        
        
        
        
        
        
        
        
        
        ### select only the logits that are related to currently activated num_locations
        
        
        
        #### here we will apply a mask per job on GO machines that are inactive for this job/current task
        ### action space size stays the same this way. Leave active logits untouched, add large negative to inactive
        
        
        if self.model_config["action_masking"] == False:
         
        
            logits = logits[: , : self.model_config["num_locations"] + 1 ]                                  ### lives on gpu
            

              
        
        
        logits_flattened = logits.reshape(batch_size,self.number_sub_jobs,logits.shape[-1])
        
        
        logits_flattened = logits_flattened.reshape(batch_size,self.number_sub_jobs*logits.shape[-1])               ### lives on gpu
        
        
  
        
        
        return {Columns.ACTION_DIST_INPUTS: logits_flattened, }
        
        
        
        
    # We implement this RLModule as a ValueFunctionAPI RLModule, so it can be used
    # by value-based methods like PPO or IMPALA.
    
    ### here we do the same as in forward except that we compute value_preds instead of logits at the end
    
    @override(ValueFunctionAPI)
    def compute_values(
        self,
        batch: dict[str, any],
        embeddings: Optional[any] = None,
        ) -> TensorType:
        # Features not provided -> We need to compute them first.
        
        pre_logits, batch_size = self.forward_part_shared_by_forward_and_value_function(batch)
        
        
        
        
        ### now apply final policy head to get logits
        
        

        
        
        ### now apply value head to get value estimate
        
        pre_logits = pre_logits.reshape(batch_size,self.number_sub_jobs,self.hidden_dim)
        
        
        
        ### do mean max pooling over the number of subjobs, so only a single value_pred per batch item
        
        ### this is meant for the case with more than a single subjob, else just use: pre_logits_pooled = pre_logits.squeeze(dim=1)
        
         
        pre_logits_mean = torch.mean(pre_logits, dim=1)
        pre_logits_max, indices = torch.max(pre_logits, dim=1)
        
        pre_logits_pooled = torch.cat([pre_logits_mean,pre_logits_max], dim = -1)
            
   
        
        value_preds = self.value_head(pre_logits_pooled)                        ### lives on gpu
        
        

            
        return value_preds




##### important: define the connections between subjobs in the GNN itself (edges). No need to give it as input
##### to a NN. Here I only have to ckeck, that they are connected correctly.

#### important: init() is only called once to initialize the env. After end of episode (during training) the reset()
#### function is called

class ConfigEnv(gym.Env):
    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        self.sequences = config.get("sequences", [])
        self.task_durations_for_all_objects = config.get("task_durations", [])
        
        self.num_objects = len(self.sequences)
        self.num_locations = config.get("num_locations", 0)
        self.pad_token = 0   # Special token for padding
        self.reward_pad_token_valid = 0.1
        self.obs_pad_length = config.get("obs_pad_length", 256)
        self.seq_pad_length = config.get("seq_pad_length", 100)
        
        
        #### do not return this as part of the observation as it does not change. provide this in the model config
        #### for your model. This stores the subjobs that need to be fullfilled, before this is allowed to start
        self.connection_between_subjobs = config.get("connection_between_subjobs", [])
        
        
        self.last_action = None
        self.actions_stored = None
        
        
        ### current action space. Not dependent on curriculum!

        self.action_space = spaces.MultiDiscrete([self.num_locations + 1] * self.num_objects)  # +1 for 'wait'       
        

        
        
        all_sequences = []
        
        for i in range(self.num_objects):
        
            sequence = self.sequences[i]
            
            all_sequences.append(sequence + [self.pad_token] * (self.seq_pad_length - len(sequence)))
            
        self.all_padded_sequences  = np.array(all_sequences, dtype=np.int32)
        
        
        #### calculate true_seq_length
        
        true_seq_length = []
        
        
        for i in range(self.num_objects):
        
            true_seq_length.append(len(self.sequences[i])) 
            
            
        self.true_seq_length  = np.array(true_seq_length, dtype=np.int32)
        
        
        
        ### now define some of the data to pass to my GNN during forward
        
        ### subjobs start to be counted from 0
        
        unmodified_dependencies = self.connection_between_subjobs
        
        self.dependency_edges_between_subjobs = np.array([[],[]])


        ### check each subjob

        for i in range(len(unmodified_dependencies)):

            if unmodified_dependencies[i]:
            
                for k in range(len(unmodified_dependencies[i])):
                
                    single_dependency = np.array([[unmodified_dependencies[i][k]],[i]])
            
                    self.dependency_edges_between_subjobs = np.concatenate((self.dependency_edges_between_subjobs,single_dependency),axis=-1)


        
        
        
        all_durations_padded = []
        
        for i in range(self.num_objects):
        
            task_duration_for_one_obj = self.task_durations_for_all_objects[i]
            
            all_durations_padded.append(task_duration_for_one_obj + [0] * (self.seq_pad_length - len(task_duration_for_one_obj)))
            
        self.all_durations_padded  = np.array(all_durations_padded, dtype=np.int32)
        
        
        
        
        ### important: The observation space is defined as numpy array, with numpy entries (np.int ect.). Do not use torch arrays
        ### will otherwise throw an error
        
        self.observation_space = spaces.Dict({
        
            'sequences': spaces.Box(low=0, high=self.num_locations, shape=(self.num_objects,self.seq_pad_length), dtype=np.int32),
            'task_durations': spaces.Box(low=0, high=np.max(self.all_durations_padded), shape=(self.num_objects,self.seq_pad_length), dtype=np.int32),
            'next_task_id': spaces.Box(low=0, high=self.num_locations, shape=(self.num_objects,), dtype=np.int32),
            'true_seq_length': spaces.Box(low=0, high=np.max(self.true_seq_length), shape=(self.num_objects,), dtype=np.int32),
            'dependency_edges_between_subjobs': spaces.Box(low=0, high=self.num_locations, shape=(2,self.dependency_edges_between_subjobs.shape[1]), dtype=np.int64),
            #'edges_next_task': spaces.Box(low=0, high=self.pad_token, shape=(2,self.num_objects), dtype=np.int64),
            #'machine_features': spaces.Box(low=0, high=1, shape=(self.num_locations,), dtype=np.int32),
            'progress_for_all_subjobs': spaces.Box(low=0, high= max(np.max(self.all_durations_padded),np.max(self.true_seq_length)), shape=(self.num_objects,2), dtype=np.int32),
        
        
        })
        
            

        self.reset()



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        progress = ([0,0],) * self.num_objects
        self.progress = np.stack(progress,-2)
        
        self.steps_taken = 0
        
        self.invalid_steps_taken = 0
        
        self.number_valid_plans = 0
        
        self.reward_scale = 1
        
        
        return self._get_obs(), {}

    def _get_obs(self):
        obs = {}
        
        #all_sequences = []
        
        next_locations = []

        for i in range(self.num_objects):
       
            sequence = self.sequences[i]
            
            #all_sequences.append(sequence + [self.pad_token] * (self.seq_pad_length - len(sequence)))
            
            
            
            task_duration_per_object = self.all_durations_padded[i]
            current_task_index = self.progress[i][0]
            current_progress_in_task = self.progress[i][1]
            

            
            ### not finished yet with task
            ### the step() updates the current task to work on. This just checks the next location 
            
            ### next locations stays same in size
            
            if current_progress_in_task < task_duration_per_object[current_task_index]:
            
                next_loc = sequence[current_task_index]
                
                
                
            else:
                next_loc = self.pad_token
                
            
            
            next_locations.append(next_loc)
            
            
        
        
        obs['dependency_edges_between_subjobs'] = np.array(self.dependency_edges_between_subjobs, dtype=np.int64)
        
        #obs['edges_next_task'] = np.array(dependency_edges_to_tasks   , dtype=np.int64) 
            
        obs['sequences'] = np.array(self.all_padded_sequences, dtype=np.int32)
        
        obs['next_task_id'] = np.array(next_locations, dtype=np.int32)
        
        obs['task_durations'] = np.array(self.all_durations_padded, dtype=np.int32)
        
        obs['true_seq_length'] = np.array(self.true_seq_length, dtype=np.int32)
        
        obs['progress_for_all_subjobs']= np.array(self.progress, dtype=np.int32)
        

        
        return obs
        
        
        
    ### the step in the env is written, such that it is indep of the number of subjobs (called here num_objects) to be well suited for curriculum learning
    ### also reward is supposed to be on the order of 1, to not supress/be suppressed by signal during sgd.

    def step(self, action):
    
        self.steps_taken += 1


        
        
        chosen_locations = []
        
        
        
        reward = 0
        
        
        ### first correct the given action into valid shape: 0 = wait, 1 = GO_1, 2 = GO_2, ...
        

        
        for obj_idx, loc in enumerate(action):
            
            current_task_index = self.progress[obj_idx][0]
            
            
            

            
            
            ### GO_1, elseif zero, stay with zero = wait
            if (loc == 1) and (current_task_index < len(self.sequences[obj_idx])):
                
                action[obj_idx] = self.sequences[obj_idx][current_task_index]
                
                
            ### if job is finished, but wants to move onwards
                
            elif (loc == 1) and (current_task_index >= len(self.sequences[obj_idx])):
                
                action[obj_idx] = self.sequences[obj_idx][-1]
            
            
            
            
            
        
         
        
        
        ### loop over subjobs
        
        
        for obj_idx, loc in enumerate(action):
        
            
  
            
            current_task_index = self.progress[obj_idx][0]
            current_progress_in_task = self.progress[obj_idx][1]
            
            
            
            
            ### first check if subjob is already allowed to start
            
            previous_subjobs_finished = True
            
            if self.connection_between_subjobs[obj_idx]:
            
                for subjob_index in self.connection_between_subjobs[obj_idx]:
                
                    
                    #### subjob is not finished yet, exit the loop
                
                    if self.progress[subjob_index][0] < len(self.sequences[subjob_index]):
                    
                        previous_subjobs_finished = False
                        
                        break
            
                
            
                
            
            
            
            
        
            ## if it waits, which is ok
            ### make sure once a task started that it finishes, you are not allowed to wait once it started
            ### devide by number of subjobs to make sure that a valid step is still positive
            
            ### only apply small penalty, we will apply penalty per time step in order to reduce total time
                
            if (loc == self.pad_token) and (current_progress_in_task == 0):  # wait
            
                reward = reward - (0.1/len(action))
                continue
                
                
            ### progress waits although it is not allowed to
                
            elif (loc == self.pad_token) and (current_progress_in_task != 0):
                
                
                
                ### penalize and correct choice
                
                ### penalize more strongly, because valid choice will get positive reward later on
                
                reward = reward - 0.35
                
                action[obj_idx] = self.sequences[obj_idx][current_task_index]
                
                
                
                continue
                
                
            ### make sure that finished subjobs only wait
                
            elif (loc != self.pad_token) and (current_task_index >= len(self.sequences[obj_idx])):
                
                ### penalize and correct choice, let it wait
                
                reward = reward - (0.45/len(action))
                
                action[obj_idx] = self.pad_token
                
                continue
                

                               
            ### choose whether it selected a valid action
            ### I give no reward if good action was chosen, as this might lead to positive total reward for a step even if invalid choices where taken
            ### => might learn wrong behaviour!
            
            
                   
            if (current_task_index < len(self.sequences[obj_idx])):
                
                next_machine = self.sequences[obj_idx][current_task_index]
                   
                if (loc == next_machine) and (previous_subjobs_finished == True):
                
                    reward = reward + 0
                    
                    
                elif (loc == next_machine) and (previous_subjobs_finished == False):
                    
                    ### penalize and correct choice, let it wait
                
                    reward = reward - (0.2/len(action))
                    
                    action[obj_idx] = self.pad_token
                    
                    
         
        
        for obj_idx, loc in enumerate(action):
            
            if loc != self.pad_token:
                
                chosen_locations.append(loc)
                
            
        
        
        
        chosen_locations = [a.item() for a in chosen_locations]
        
        stacked_locations = np.array(chosen_locations)
            
        ### detect double occup
        ### shift diff by 1, since two double vals, reduce to a single.
        
        if stacked_locations.size != np.unique(stacked_locations).size:
            
            
            ### apply penalty
        
            reward = reward - (0.25 * (abs(stacked_locations.size - np.unique(stacked_locations).size))+1)
            
            ### now correct action
            
            vals, inv_map = np.unique(action, return_inverse=True)
            
            ### gives positions of identical vals in action
            
            groups_of_identical_vals = [np.where(inv_map == i)[0] for i in range(len(vals))]
            
            
  
            
            
            
            for k in range(len(groups_of_identical_vals)):
                
                current_group = groups_of_identical_vals[k]
                
                ### look at multiple machines requested and skip waiting
                
                if (current_group.size > 1) and action[current_group[0]] != self.pad_token:
                    
                    temp_progress = np.array(self.progress)[current_group, 1]
                    
  
                    
                    ### np where returns a tuple. Take [0] entry
                    
                    running_task = np.where(temp_progress > 0)[0]
                    
                    
 
                    
                    
                    
                    ### one job is already on machine, let the other jobs wait
                    
                    if running_task.size == 1:
                        
                        action[current_group[np.arange(current_group.size) != running_task]] = self.pad_token
                        
                        
                    ### decide randomnly which one to choose
                        
                    elif running_task.size == 0:
                        
                        selected_job = np.round(np.random.uniform(low = 0, high = current_group.size-1 , size=1))[0]
                        
                        
                        action[current_group[np.arange(current_group.size) != selected_job]] = self.pad_token
                        
                        
                    elif running_task.size > 1:
                        
                        print()
                        print("Error: More than one job is working on a machine at once!!")
                        print()
                        
                        raise ValueError("Error: More than one job is working on a machine at once!!")
                    
                    
 
        


        #### entire action has passed the test and is valid
        #### -> update the progress of each job to use for observation and output reward
        
        self.number_valid_plans += 1
        

        
        ### loop over subjobs
        
        
        ### check double occupancy as safety
        
        action_no_wait = np.array(action)
        
        action_no_wait = action_no_wait[action_no_wait != self.pad_token]
        
        if action_no_wait.size != np.unique(action_no_wait).size:
            
            print()
            print(f"Error occured in correcting actions: Double occupation was not corrected precisely!!")
            print() 

            raise ValueError("Error occured in correcting actions: Double occupation was not corrected precisely!!")
        
        
        for obj_idx, loc in enumerate(action):
        
            current_task_index = self.progress[obj_idx][0]
            current_progress_in_task = self.progress[obj_idx][1]
            
            
            if (current_progress_in_task > 0) and (loc != self.sequences[obj_idx][current_task_index]):
                
                print()
                print(f"Error occured in correcting actions: Task was discontinued unintentionally!!")
                print()  
                
                raise ValueError("Error occured in correcting actions: Task was discontinued unintentionally!!")
                            
            
            
            ### valid actions should get positive total reward. Therefore dont punish waiting there = remove/compensate previouly applied penalty
            '''
            if (loc == self.pad_token) and (len(action) > 1):
                
                reward = reward + self.reward_pad_token_valid
                
            '''
            
            
                
            ### works only for actions that do not wait
            
            if loc != self.pad_token:
                
                
                
                ### is sort of a double check that everything worked well
                
                if loc == self.sequences[obj_idx][current_task_index]:
                

            

            
                    if (current_progress_in_task+1) == self.task_durations_for_all_objects[obj_idx][current_task_index]:
                    
                    
                        ### you have completed this task
                        ### move one task further and set steps in that task to zero
                        
                    
                        self.progress[obj_idx][0] += 1
                        self.progress[obj_idx][1] = 0
                        
                        reward += 0.5
                        
                        
                        #### give extra reward if a subjob finishes
                        
                        if self.progress[obj_idx][0] >= len(self.sequences[obj_idx]):
                        
                            reward += 0.5
                        
                        
                    

                        
                    else:
                    
                        ### the object has completed one step in the task
                        ### sort of give an extra reward, as all chosen actions are valid, should speed up learning ;)
                    
                        self.progress[obj_idx][1] += 1
                        
                        reward = reward + 0.5
                            
                
                else:
                    
                    print()
                    print(f"Error occured in correcting actions: Task was assigned that is not the current goal!!")
                    print()  
                    
                    raise ValueError("Error occured in correcting actions: Task was assigned that is not the current goal!!")


        
        #### if all sequences have finished the final task

        done = all(p[0] >= len(seq) for p, seq in zip(self.progress, self.sequences))
        
        if done:
        
            print('All jobs are finished!!')
            

            
            
        ###  apply at each time step a penalty in order to reduce total time required
        ### should be of similar size as pos reward in order to matter, therefore multiply by num machines

        
        reward = reward - 0.05 * self.num_locations
        
        
        return self._get_obs(), 2*reward/(self.num_objects * self.seq_pad_length), done, False, {}
        
        
        
        


def one_training_cycle(sequences = [[2, 3, 3, 2, 5],[2, 4, 2],[3, 5, 5]], task_durations = [[1, 1, 1, 1, 1], [1, 1, 1],[1, 1, 1]], connection_between_subjobs = [[],[],[]], embedding_dim = 3, hidden_dim = 64, cur_learning_max_dur = 3, cur_learning_max_num_locations = 7, cur_learning_max_true_seq_length = 8, min_number_of_training_steps = 10, max_number_of_training_steps = 80, first_entropy_coeff = 0.1, freeze_embeddings = False, freeze_gru = False, add_training_steps = 5, add_randomness_to_weights = False, number_GRU_layers = 2, number_GNN_layers = 2, GRU_bidirectional = False, action_masking = True, cur_learning_max_outputs = 2):
        
        
        
    def env_creator(config):
        return ConfigEnv(config)  # Return a gymnasium.Env instance.

    register_env

    my_env = register_env("my_env", env_creator)
    
    
    ### current val
            
    print("sequences: ", sequences)




    ## I choose to pad sequences with zeros, since this can be used consistently during different curriculum 
    ## learning cycles

    ### same for task duration



    ### current val
    print("task_durations: ", task_durations)



    max_val = 0
    for k in sequences:
        
        max_val = max(max_val,max(k))

    ### current val
    num_locations = max_val







    max_val = 0
    for k in sequences:
        
        max_val = max(max_val,len(k))
        

    ### current val
    ### needs to be one above the actual seq length, due to boundaries
    seq_pad_length = max_val + 1



    ### current val
    num_objects = len(sequences)


    ### subjobs start to be counted from 0
    ### the first inner bracket defines the connection of subjob 0 to all other subjobs (fill bracket with theirrespective numbers), the next bracket of subjob 1 to all others


    ### current val
    print("connection_between_subjobs: ", connection_between_subjobs)






    ### current val
    input_length = [cur_learning_max_outputs] * num_objects
    
    print("Input length: ", input_length)


    MultiCategorical.set_input_lens(input_length)



    env_conf = {"action_masking": action_masking, "cur_learning_max_outputs": cur_learning_max_outputs, "sequences": sequences,"num_locations": num_locations, "seq_pad_length": seq_pad_length, "task_durations":task_durations,"connection_between_subjobs": connection_between_subjobs}


    # Create an env object to know the spaces.
    env = ConfigEnv(env_conf)




    ### current val
    max_duration = np.max(env.all_durations_padded)



    ### current val
    max_seq_length = np.max(env.true_seq_length)



    policy_conf = {"action_masking": action_masking, "cur_learning_max_outputs": cur_learning_max_outputs, "cur_learning_max_true_seq_length": cur_learning_max_true_seq_length, "cur_learning_max_num_locations": cur_learning_max_num_locations,"cur_learning_max_dur": cur_learning_max_dur,"max_seq_length": max_seq_length,"embedding_dim_progres": embedding_dim,"embedding_dim_machines": embedding_dim, "seq_pad_length": seq_pad_length,"embedding_dim_durations": embedding_dim, "max_duration":max_duration,"hidden_dim": hidden_dim, "num_objects": num_objects,"num_locations": num_locations, "freeze_embeddings": freeze_embeddings, "freeze_gru": freeze_gru,"number_GNN_layers": number_GNN_layers, "number_GRU_layers": number_GRU_layers, "GRU_bidirectional": GRU_bidirectional}


    ### set env runners = 2, with 4 envs each


    temp_obs = env.reset()

    print('temp_obs: ',temp_obs)

    # First construct the spec.
    module_spec = RLModuleSpec(
        module_class=myModule,
        
        observation_space=env.observation_space,
        action_space=env.action_space,
        
        # A custom dict that's accessible inside your class as `self.model_config 
        
        model_config= policy_conf
    )
    
    number_of_learners = 1
    
    ### RLIB forwards every parameter that contributes to loss to DDP, in parallel computing (so only if number_of_learners >1)
    
    if number_of_learners > 1:
        
        rlib_foward_para_to_ddp = True
        
    else:
        
        rlib_foward_para_to_ddp = False
        
    
        
    ### Important for use with gpu: GPU only gives advantage at minibatch sizes in range (1000 to 8000) and batch sizes (30k). Also with larger networks.
    ### With lots of small batches or small networks, the CUDA kernel has to constantly relaunch, which is expensive


    ### very important: the number of sgd iters is called, num_epochs in this rllib version!!    


    config = (
        PPOConfig()
        .callbacks(CriticStats)
        .environment(
            "my_env",
            env_config = env_conf,  # `config` to pass to your env class
        )
        .rl_module(
                rl_module_spec = module_spec,
                #reward_normalization = True,
        )
        .env_runners(num_env_runners=2, 
                        num_envs_per_env_runner = 4,
                        rollout_fragment_length = 125,
        
        )
        
        .resources(num_gpus=1,
                    num_gpus_per_worker = 0
        
        )
        
        .learners(
        
            num_learners = number_of_learners,
            
            num_cpus_per_learner = 6,
            
            #### important: need to set the num gpus per learner to 1, to use it during SGD, which is most important.
            
            #num_gpus_per_learner = 1,
            
            #learner_queue_size = 8,
            
            #max_requests_in_flight_per_learner = 5,
        
        
        )
        
        .training(
        use_kl_loss = True,
        #use_gae =False,
        #lr=[[0, 1e-5],[8000, 1e-4]],
        lr = [[0, 4e-4],[5000, 4e-4],[10000, 3e-4],[15000, 2e-4],[30000, 1e-4]],
        train_batch_size_per_learner=1000,
        minibatch_size = 1000,
        clip_param=0.1,
        num_epochs=15,
        #grad_clip_by='value',
        #grad_clip=0.1,
        entropy_coeff=[[0, first_entropy_coeff],[10000, 1e-2],[20000, 1e-3],[30000, 1e-4],[40000, 0]], 
        vf_clip_param = 10,
        vf_loss_coeff=1.5,
        grad_clip=0.5,
        kl_target = 0.01,
        kl_coeff = 1.0,
        #model = {"vf_share_layers": False}
        )
        #.resources(num_learners=1)
        #.experimental(_validate_config=False)
        
        .framework("torch",
        
                    torch_ddp_kwargs={"find_unused_parameters": rlib_foward_para_to_ddp}
        
        )
        
        
    )


    #config["num_learners"] = 1




    #### now build algorithm

    ppo = config.build_algo()




    ### use model from start for now

    



    ### load a already trained model, prior to training
    
    weights_path = f"test_save/weights_shared_embed_dim_{embedding_dim}_shared_hidden_dim_{hidden_dim}_cur_learning_max_dur_{cur_learning_max_dur}_cur_learning_max_num_locations_{cur_learning_max_num_locations}_cur_learning_max_true_seq_length_{cur_learning_max_true_seq_length}_num_GRU_layers_{number_GRU_layers}_GRU_bidirectional_{GRU_bidirectional}_num_GNN_layers_{number_GNN_layers}_action_masking_{action_masking}_cur_learning_max_outputs_{cur_learning_max_outputs}.pkl"

    windows_path = REPO_PATH / weights_path
    
    windows_path.parent.mkdir(parents=True, exist_ok=True)
    


    if windows_path.exists():
        
        print()
        print("Load trained state from disk!")
        print()

        with open(windows_path, "rb") as f:
            weights = pickle.load(f)
        
        
        
    ### important weights is a dict in a dict: first level is key:default_policy. second level are the parts of the default policy
        
    


    #### now I want to nuckle at the parameters of the model a bit to make it suited for curriculum learning
    ### Important: weights is a dict in a dict: first level is key:default_policy. second level are the parts of the default policy
    
    if add_randomness_to_weights == True:
        
        print()
        print("Add some randomness to the weights!!")
        print()

        exploration_factor = 1

        exploration_scale = 0.5*1e-3


        for key,val in weights.items():
            
            
            
            
            for key2,val2 in val.items():
                
                rand_array = exploration_factor * np.random.normal(loc = 0, scale = exploration_scale, size = val2.shape)
                
                
                weights[key][key2] = weights[key][key2] + rand_array
                

            
        
    if windows_path.exists():
        
        
        print()
        print("Use already trained state!")
        print()
        
        ppo.set_weights(weights)






    max_return, best_action = inference(env, ppo, MultiCategorical)
    
    print("sequences: ", sequences)
    
    print("task durations: ", task_durations)


    
    
    ### set a policy entropy covergence val, dependent on the size of the action space, is just approximation
    
    policy_entropy_conv_val = abs(-2 + np.log(sum(input_length)) + 0.1) + 0.1
    
    print("policy entropy conv val: ", policy_entropy_conv_val)

    start_time = time.time()
    
    
    
    add_steps = 0
    
    max_return_per_train = -100

    for current_training_step in range(max_number_of_training_steps):

        ppo_out = ppo.train()
        
        middle_time = time.time()
        
        print("elapsed time: ",middle_time - start_time,"secs")
        
        mean_kl_loss =          ppo_out["learners"]["default_policy"]["mean_kl_loss"]
        policy_loss =           ppo_out["learners"]["default_policy"]["policy_loss"]
        value_function_loss =   ppo_out["learners"]["default_policy"]["vf_loss"]
        episode_return_mean =   ppo_out["env_runners"]["episode_return_mean"]
        policy_entropy =        ppo_out["learners"]["default_policy"]["entropy"]
        explained_variance =    ppo_out["learners"]["default_policy"]["vf_explained_var"]
        training_iteration =    ppo_out["training_iteration"]
        learning_rate =         ppo_out["learners"]["default_policy"]["default_optimizer_learning_rate"]
        #value_target =          ppo_out["critic/var_targets"]
        #MAE =                   ppo_out["critic/mae"]
        #bias =                  ppo_out["critic/bias"]
        
        
        if episode_return_mean > max_return_per_train:
            
            max_return_per_train = episode_return_mean
            
            best_step = current_training_step
        
        
        print('mean_kl_loss:             ',mean_kl_loss)
        print('policy_loss:              ',policy_loss)
        print('value_function_loss:      ',value_function_loss)
        print('episode_return_mean:      ',episode_return_mean)
        print('policy_entropy:           ',policy_entropy)
        print('explained_variance:       ',explained_variance)
        
        weights_path_temp = f"test_save/temp_weights_{current_training_step}.pkl"

        temp_weight_path = REPO_PATH / weights_path_temp 

        temp_weight_path.parent.mkdir(parents=True, exist_ok=True)
        temp_weight_path.resolve()
        
        
        temp_weights = ppo.get_weights()


        with open(temp_weight_path, "wb") as f:
            pickle.dump(temp_weights, f)
        
        if (policy_entropy <= policy_entropy_conv_val) and (current_training_step > min_number_of_training_steps): #and (episode_return_mean >=0.9)
            
            print("Training coverged! Breaking the training loop!")
            
            add_steps = add_steps + 1
            
            if add_steps == add_training_steps:
            
                break
            
            
            
    ### now check how often the jobs have been finished during training
            
            
    regex_line = re.compile(r"All jobs are finished!!.*repeated (\d+)x across cluster\]")
    
    regex_line_upper_bound = re.compile(r"mean_kl_loss:")
    
    times_all_jobs_finished = 0
    
    with open("train.log", "r") as f:
        
        ### read all lines into memory
        
        lines = f.readlines()
        
    
    ### first determine the upper most line up to which, one should check for the number of finished jobs
    
    line_iter_upper_bound = 0
    
    number_occur_of_upper_bound_line = 0
    
    for line in reversed(lines):
            
            ### using search -> regex_line doesnt need to fit the entire line!
            
            line_iter_upper_bound = line_iter_upper_bound + 1
            
            m = regex_line_upper_bound.search(line)
            
            if m:
                
                number_occur_of_upper_bound_line = number_occur_of_upper_bound_line + 1
                
                ### if we break earlier, the system might not reach back to the last number of finished jobs, see log
                
                if number_occur_of_upper_bound_line == 2:
                
                    break
                
    print(f"Search for number of finished jobs up to line {line_iter_upper_bound}")
        
    ### search starting from the back
    
    line_iter = 0
        
    for line in reversed(lines):
        
        ### using search -> regex_line doesnt need to fit the entire line!
        
        line_iter = line_iter + 1
        
        
        
        ### searching has moved past the most recent printing of "All jobs have been finished"
        
        if line_iter >= line_iter_upper_bound:
            
            print("Searching for the most recent number of finishing all jobs, has not found any!!")
            
            break
        
        m = regex_line.search(line)
        
        if m:
            
            times_all_jobs_finished = int(m.group(1))
            
            print(f"All jobs have been finished {times_all_jobs_finished} times during training!!" )
            
            break
            
            
    ### release memory of read file
    
    del lines
            
            
    if (episode_return_mean < 0) or (times_all_jobs_finished < 10):
        
        print("PPO converged to bad point!. Try with higher initial entropy!")
        print("Therefore, updated policy will not be saved!")
        
        return
            
    end_time = time.time()
    
    print("elapsed time: ",end_time - start_time,"secs")
    
    max_return, best_action = inference(env, ppo, MultiCategorical)
    
    print("sequences: ", sequences)
    
    print("task durations: ", task_durations)
            

    
    ### instead of taking the final weight, take the best of the logged ones
    
    print()
    
    print(f"Optimal result was reached at step {best_step} with a reward of {max_return_per_train}!")
    
    print()

    best_weights_path_temp = f"test_save/temp_weights_{best_step}.pkl"

    best_temp_weight_path = REPO_PATH / best_weights_path_temp    
    best_temp_weight_path.resolve()
    
    
    
    
    with open(best_temp_weight_path, "rb") as f:
        weights = pickle.load(f)
    

    

    


    #### important: between saving and loading weights, you have to keep constant:
    ####    - number machines, embeding dimensions, max. duration for one task
    ####    - max. seq length, hidden dim

    with open(windows_path, "wb") as f:
        pickle.dump(weights, f)


    with open(windows_path, "rb") as f:
        weights = pickle.load(f)
        


        
    ppo.set_weights(weights)


    ### shutdown of training


    #torch.destroy_process_group()
    ppo.stop()
    
  
  
### every training cycle saves and loads the weights. => There is no need to return anything!!
  

def curriculum_training(embedding_dim = 3, hidden_dim = 64, cur_learning_max_dur = 3, cur_learning_max_num_locations = 7, cur_learning_max_true_seq_length = 8, max_duration = 3, max_seq_length = 3, max_number_machines = 5, number_of_subjobs = 3, allow_connections_between_subjobs = False, min_number_of_training_steps = 10, max_number_of_training_steps = 80, first_entropy_coeff = 0.1, freeze_embeddings = False, freeze_gru = False, add_training_steps = 5, add_randomness_to_weights = False, number_GRU_layers = 2, number_GNN_layers = 2, GRU_bidirectional = False, action_masking = True, cur_learning_max_outputs = 2):
  
  
    sequences, task_durations, connection_between_subjobs = random_job_setup(number_of_subjobs = number_of_subjobs, allow_connections_between_subjobs = allow_connections_between_subjobs, max_duration = max_duration, max_seq_length = max_seq_length, max_number_machines = max_number_machines)
  
    
    one_training_cycle( sequences, task_durations, connection_between_subjobs, embedding_dim, hidden_dim, cur_learning_max_dur, cur_learning_max_num_locations, cur_learning_max_true_seq_length, min_number_of_training_steps, max_number_of_training_steps, first_entropy_coeff, freeze_embeddings, freeze_gru, add_training_steps, add_randomness_to_weights, number_GRU_layers, number_GNN_layers, GRU_bidirectional, action_masking, cur_learning_max_outputs)


### Below are several curriculum learning steps. They can be executed one after another if starting with an untrained model.



### In this example we work with a more challenging problem, since we start with an already trained model

for curr_iter in range(50):

    curriculum_training(embedding_dim = 3, hidden_dim = 64, cur_learning_max_dur = 3, cur_learning_max_num_locations = 7, cur_learning_max_true_seq_length = 8, max_duration = 3, max_seq_length = 8, max_number_machines = 5, number_of_subjobs = 10, min_number_of_training_steps = 3, max_number_of_training_steps = 80, freeze_embeddings = False, freeze_gru = False, number_GRU_layers = 2, number_GNN_layers = 2, GRU_bidirectional = False, add_training_steps = 15, action_masking = True, cur_learning_max_outputs = 2, first_entropy_coeff = 0.5e-1)
    
    





### first learn the machine embeddings, so 1 subjob with tasks of length 1

#for curr_iter in range(20):

    #curriculum_training(embedding_dim = 3, hidden_dim = 64, cur_learning_max_dur = 3, cur_learning_max_num_locations = 7, cur_learning_max_true_seq_length = 8, max_duration = 3, max_seq_length = 8, max_number_machines = 5, number_of_subjobs = 1, min_number_of_training_steps = 3, max_number_of_training_steps = 40, freeze_embeddings = False, freeze_gru = False, number_GRU_layers = 2, number_GNN_layers = 2, GRU_bidirectional = False, add_training_steps = 10, action_masking = True, cur_learning_max_outputs = 2)
    
    
    
    
    
### now learn the duration embeddings and train the GRU, so 1 subjob with tasks of diff length 

#for curr_iter in range(200):

    #curriculum_training(embedding_dim = 3, hidden_dim = 64, cur_learning_max_dur = 3, cur_learning_max_num_locations = 7, cur_learning_max_true_seq_length = 8, max_duration = 3, max_seq_length = 8, max_number_machines = 5, number_of_subjobs = 1, min_number_of_training_steps = 3, max_number_of_training_steps = 40, freeze_embeddings = False, freeze_gru = False, number_GRU_layers = 2, number_GNN_layers = 2, GRU_bidirectional = False, add_training_steps = 15)
    
    
    
### now combine criteria for two subjobs with short task sequences 

#for curr_iter in range(50):

    #curriculum_training(embedding_dim = 3, hidden_dim = 64, cur_learning_max_dur = 3, cur_learning_max_num_locations = 7, cur_learning_max_true_seq_length = 8, max_duration = 3, max_seq_length = 4, max_number_machines = 5, number_of_subjobs = 2, min_number_of_training_steps = 3, max_number_of_training_steps = 60, freeze_embeddings = False, freeze_gru = False, number_GRU_layers = 2, number_GNN_layers = 2, GRU_bidirectional = False, add_training_steps = 15, action_masking = True, cur_learning_max_outputs = 2)
    
    
    
### now combine criteria for two subjobs with longer task sequences 

#for curr_iter in range(100):

    #curriculum_training(embedding_dim = 3, hidden_dim = 64, cur_learning_max_dur = 3, cur_learning_max_num_locations = 7, cur_learning_max_true_seq_length = 8, max_duration = 3, max_seq_length = 8, max_number_machines = 5, number_of_subjobs = 2, min_number_of_training_steps = 3, max_number_of_training_steps = 60, freeze_embeddings = True, freeze_gru = True, number_GRU_layers = 2, number_GNN_layers = 2, GRU_bidirectional = False, add_training_steps = 15, first_entropy_coeff = 0.1)
    
    
### now combine criteria for three subjobs with short task sequences 

#for curr_iter in range(10):

    #curriculum_training(embedding_dim = 3, hidden_dim = 64, cur_learning_max_dur = 3, cur_learning_max_num_locations = 7, cur_learning_max_true_seq_length = 8, max_duration = 3, max_seq_length = 4, max_number_machines = 5, number_of_subjobs = 3, min_number_of_training_steps = 5, max_number_of_training_steps = 40, freeze_embeddings = True, freeze_gru = True, number_GRU_layers = 2, number_GNN_layers = 2, GRU_bidirectional = False, add_training_steps = 15)
    
    
### now combine criteria for three subjobs with longer task sequences 

#for curr_iter in range(10):

    #curriculum_training(embedding_dim = 3, hidden_dim = 64, cur_learning_max_dur = 3, cur_learning_max_num_locations = 7, cur_learning_max_true_seq_length = 8, max_duration = 3, max_seq_length = 8, max_number_machines = 5, number_of_subjobs = 3, min_number_of_training_steps = 5, max_number_of_training_steps = 40, freeze_embeddings = True, freeze_gru = True, number_GRU_layers = 2, number_GNN_layers = 2, GRU_bidirectional = False, add_training_steps = 15)
    
    

### now combine criteria for four subjobs with longer task sequences 

#for curr_iter in range(10):

    #curriculum_training(embedding_dim = 3, hidden_dim = 64, cur_learning_max_dur = 3, cur_learning_max_num_locations = 7, cur_learning_max_true_seq_length = 8, max_duration = 3, max_seq_length = 8, max_number_machines = 5, number_of_subjobs = 4, min_number_of_training_steps = 5, max_number_of_training_steps = 40, freeze_embeddings = True, freeze_gru = True, number_GRU_layers = 2, number_GNN_layers = 2, GRU_bidirectional = False, add_training_steps = 15)
    
    
    
    
    
