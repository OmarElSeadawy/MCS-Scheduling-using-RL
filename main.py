import gym, ray
from environment_generator import MCOfflineNewEnv,MCOnlineNewEnv
from ray import tune
# from ray.tune.logger import LoggerCallback
from ray.rllib.agents.dqn import ApexTrainer, DQNTrainer
from ray.tune.registry import register_env
from ray.rllib.agents.dqn.distributional_q_tf_model import \
    DistributionalQTFModel
from ray.rllib.agents.dqn.dqn_torch_model import \
    DQNTorchModel
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.preprocessors import get_preprocessor, DictFlatteningPreprocessor, NoPreprocessor
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
import tensorflow as tf
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

        self.register_variables(self.action_embed_model.variables())

    def forward(self, input_dict, state, seq_lens):
        # Compute the predicted action embedding
        # Flatten all before addition
        feat = []
        for k, v in input_dict["obs"]['MCOnlineNewEnv'].items():
            if len(v.shape) > 2:
                v = tf.reshape(v, (-1, v.shape[1]*v.shape[2]))
            feat.append(v)

        feat = tf.concat(feat, axis=-1)
        avail_actions = input_dict["obs"]["avail_actions"]
        action_mask = input_dict["obs"]["action_mask"]

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
        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()

## arg_parser and get_opt were used for running the model using CLI, but PyCharm was used for the majority of experiments
## So they can be skipped
def arg_parser(self):
    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-a", "--firstop", required=True,
                    help="Name of Experiment (Name of Environment)")
    ap.add_argument("-b", "--secondop", required=True,
                    help="CheckPoints_Frequency")
    ap.add_argument("-c", "--thirdop", required=True,
                    help="Number of Iterations")
    ap.add_argument("-d", "--fourthop", required=True,
                    help="Number of Workers")
    ap.add_argument("-e", "--fifthop", required=True,
                    help="Percentage of CPUs per Worker")
    ap.add_argument("-f", "--sixthop", required=True,
                    help="Gamma Value, default is 1")
    ap.add_argument("-g", "--seventhop", required=True,
                    help="Parametric Actions Option, True/False")
    args = vars(ap.parse_args())
    return args

def get_opt():

    # Get the arguments from the command-line except the filename
    argv = sys.argv[1:]
    sum = 0

    try:
        # Define the getopt parameters
        opts, args = getopt.getopt(argv, 'a:b:c:d:e:f:g:')
        # Check if the options' length is 2 (can be enhanced)
        if len(opts) == 0 or len(opts) <7 :
            print('usage: rllib_train.py -a <Exp_Name> -b<Checks_Freq> <Iterations_Num> <Worker_Num> <CPUs/Worker> <Gamma> <Parametric_Actions>')
        else:
            for opt, arg in opts:
                args.append(arg)
            args[1]=  int(args[1])
            args[2] = int(args[2])
            args[3] = int(args[3])
            args[4] = float(args[4])
            args[5] = float(args[5])
            args[6]= bool(args[6])
            return args
    except getopt.GetoptError:
        # Print something useful
        print('usage: rllib_train.py -a <Exp_Name> -b<Checks_Freq> <Iterations_Num> <Worker_Num> <CPUs/Worker> <Gamma> <Parametric_Actions>')
        sys.exit(2)


## The main function, training starts here
## Define Parameters for ApexTrainer depending on the hardware specifications
## The output model is usually in /home/{user}/ray_results/..
if __name__ == "__main__":

    ray.init()

    ModelCatalog.register_custom_model("pa_model_intent", DQNModel)

    register_env("MCOnlineNewEnvNewAdjustment", lambda env_config: MCOnlineNewEnv())

    tune.run(ApexTrainer, checkpoint_freq=20, stop={"training_iteration": 10000},  config={"env": "MCOnlineNewEnvNewAdjustment", "num_workers": 12, "num_cpus_per_worker": 1,
                                                                                           "num_gpus": 1, "gamma": 1})