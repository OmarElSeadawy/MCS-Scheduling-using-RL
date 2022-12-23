import gym, ray
from environment_generator import MCOfflineNewEnv,MCOnlineNewEnv
from ray import tune
# from ray.tune.logger import LoggerCallback
# from ray.rllib.agents.dqn import ApexTrainer, DQNTrainer
from ray.rllib.algorithms.apex_dqn import ApexDQN as ApexTrainer
from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn.distributional_q_tf_model import DistributionalQTFModel
# from ray.rllib.agents.dqn.dqn_torch_model import \
#     DQNTorchModel
from gym.spaces import Box, MultiBinary, Discrete, Dict
import numpy as np
import ray.rllib.algorithms.ppo as ppo
from ray.rllib.agents.ppo import PPOTrainer
import ray.rllib.algorithms.dqn as dqn
import ray.rllib.algorithms.dreamer as dreamer
import ray.rllib.algorithms.simple_q as simpleq
import ray.rllib.algorithms.ars as ars
import ray.rllib.algorithms.marwil as cql
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.preprocessors import get_preprocessor, DictFlatteningPreprocessor, NoPreprocessor
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
import tensorflow as tf
from ray.tune.logger import pretty_print
from gym.spaces import flatten_space
from gym.spaces.utils import flatten, flatten_space

import sys
import getopt
import argparse

class DQNModel(DistributionalQTFModel):
    """Parametric action model that handles the dot product and masking.
    This assumes the outputs are logits for a single Categorical action dist.
    Getting this to work with a more complex output (e.g., if the action space
    is a tuple of several distributions) is also possible but left as an
    exercise to the reader.
    """

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 true_obs_shape=(4, ),
                 action_embed_size=5,
                 **kw):
        super(DQNModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, **kw)
        self.prep = DictFlatteningPreprocessor(obs_space.original_space['MCOnlineNewEnv'])
        self.action_embed_model = FullyConnectedNetwork(
            self.prep.observation_space, action_space, action_embed_size,
            model_config, name + "_action_embed")
        print("ACTION EMBED MODEL : ", self.action_embed_model)
        self.register_variables(self.action_embed_model.variables())

    def forward(self, input_dict, state, seq_lens):
        # Compute the predicted action embedding
        # Flatten all before addition
        feat = []
        for k, v in input_dict["obs"]['MCOnlineNewEnv'].items():
            if len(v.shape) > 2:
                v = tf.reshape(v, (-1, v.shape[1]*v.shape[2]))
            feat.append(v)
        print("FEAT ", feat)
        feat = tf.concat(feat, axis=-1)
        avail_actions = input_dict["obs"]["avail_actions"]
        action_mask = input_dict["obs"]["action_mask"]
        print("avail_actions ", avail_actions)
        # Compute the predicted action embedding
        action_embed, _ = self.action_embed_model({
            # "obs": flatten_space(input_dict["obs"]["MCOnlineNewEnv"])
            "obs": feat
        })

        # Expand the model output to [BATCH, 1, EMBED_SIZE]. Note that the
        # avail actions tensor is of shape [BATCH, MAX_ACTIONS, EMBED_SIZE].
        intent_vector = tf.expand_dims(action_embed, 1)

        # Batch dot product => shape of logits is [BATCH, MAX_ACTIONS].
        action_logits = tf.reduce_sum(avail_actions * intent_vector, axis=2)

        # Mask out invalid actions (use tf.float32.min for stability)
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        print("action_logits ", action_logits)

        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()

## The main function, training starts here
## Define Parameters for ApexTrainer depending on the hardware specifications
## The output model is usually in /home/{user}/ray_results/..
if __name__ == "__main__":

    ray.init()

    ##PPO Default Implementation
    # config = ppo.DEFAULT_CONFIG.copy()
    # register_env("MCOnlinePPOLatest2", lambda env_config: MCOnlineNewEnv())
    # config["num_gpus"]=0
    # config["num_workers"]=4
    # config["env"] = "MCOnlinePPOLatest2"

    # alg = ppo.PPO(config=config)
    # print("INITIALIZED")
    # file = open("PPOTrainResultsFileLatest2.txt","a")
    # for i in range(500):
    #     print("-----------", i, "---------------")
    #     result = alg.train()
    #     print(pretty_print(result))
    #     file.write(pretty_print(result))
    #     if(i%50) == 0:
    #         checkpoint = alg.save()
    #         print("checkpoint saved at", checkpoint)
    # file.close()

    # ## DQN Default Implementation
    # config = dqn.DEFAULT_CONFIG.copy()
    # config["num_gpus"] = 0
    # config["num_workers"] = 4
    # register_env("MCOnlineDQNLatest", lambda env_config: MCOnlineNewEnv())
    # alg = dqn.DQN(config=config, env="MCOnlineDQNLatest")
    # # alg.restore('/home/omarelseadawy/ray_results/DQN_MCOnlineDQNOG_2022-11-07_11-54-04sbhys6zy/checkpoint_000201/')
    # print("INITIALIZED")
    # file1 = open("DQNTrainResultsFile.txt","a")
    # for i in range(500):
    #     print("-----------", i, "---------------")
    #     result = alg.train()
    #     print(pretty_print(result))
    #     file1.write(pretty_print(result))
    #     if (i % 50) == 0:
    #         checkpoint = alg.save()
    #         print("checkpoint saved at", checkpoint)
    # file1.close()

    # # ## ARS Implementation
    # config = ars.DEFAULT_CONFIG.copy()
    # config["num_gpus"] = 0
    # config["num_workers"] = 4
    # register_env("MCOnlineARSLatest", lambda env_config: MCOnlineNewEnv())
    # alg = ars.ARS(config=config,env="MCOnlineARSLatest")
    # print("INITIALIZED ARS")
    # file3 = open("ARSTrainResultsNew.txt","a")
    # for i in range(1500):
    #     print("-----------", i, "---------------")
    #     result = alg.train()
    #     print(pretty_print(result))
    #     file3.write(pretty_print(result))
    #     if (i % 50) == 0:
    #         checkpoint = alg.save()
    #         print("checkpoint saved at", checkpoint)
    # file3.close()


    ### APEX Important
    register_env("MCOnlineNewRewardPreempt2", lambda env_config: MCOnlineNewEnv())
    
    ModelCatalog.register_custom_model("debug_model", DQNModel)
    
    
    tune.run(ApexTrainer, checkpoint_freq=50, stop={"training_iteration": 10000},  config={"env": "MCOnlineNewRewardPreempt2", "num_workers": 2 , "num_cpus_per_worker": 1,
                                                                                           "num_gpus": 0})