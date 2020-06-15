from torch import nn as nn
import pygame
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.torch.dqn.dqn import DQNTrainer
from rlkit.torch.networks import Mlp
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from rlkit.launchers.launcher_util import run_experiment_here


from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import torch
import uuid
from rlkit.core import logger

from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import torch
import uuid
from rlkit.core import logger
from environment import *

filename = str(uuid.uuid4())

def simulate_policy(args):
    args.file = '/home/yujr/rlkit/data/Test/Test_2020_06_08_21_52_33_0000--s-79802/params.pkl'
    data = torch.load(args.file)
    print("data loaded", data['evaluation/policy'])
    policy = data['evaluation/policy']
    # env = data['evaluation/env']

    arg = getArgs()
    env = environment(arg)

    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
    while True:
        path = rollout(
            env,
            policy,
            max_path_length=args.H,
            render=True,
        )

        # if hasattr(env, "log_diagnostics"):
        #     env.log_diagnostics([path])
        logger.dump_tabular()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('file', type=str,
    #                     help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true', default=True)
    args = parser.parse_args()

    simulate_policy(args)
