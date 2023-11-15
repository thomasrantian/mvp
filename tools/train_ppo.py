#!/usr/bin/env python3

"""Train a policy with PPO."""

import hydra
import omegaconf
import os

from mvp.utils.hydra_utils import omegaconf_to_dict, print_dict, dump_cfg
from mvp.utils.hydra_utils import set_np_formatting, set_seed
from mvp.utils.hydra_utils import parse_sim_params, parse_task
from mvp.utils.hydra_utils import process_ppo
import uuid
import torch.distributed as dist
import torch
# Find the path to the parent directory of the folder containing this file.
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
# exlucde the last folder
DIR_PATH = os.path.dirname(DIR_PATH)
DIR_PATH = os.path.dirname(DIR_PATH)

@hydra.main(config_name="config", config_path="../configs/ppo")
def train(cfg: omegaconf.DictConfig):

    # Assume no multi-gpu training
    #assert cfg.num_gpus == 1
    
    # Change the num_gpu_here
    cfg.num_gpus = 1
    # Set up distributed env
    if cfg.num_gpus > 1:
        dist.init_process_group("nccl")
        local_rank = dist.get_rank() % cfg.num_gpus
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0

    # Change the log dir in the mvp_exp_data folder
    # generate a unique id for the experiment
    cfg.logdir = DIR_PATH + "/mvp_exp_data/rl_runs/9_26_paper_results_tcc_kuka/" + str(uuid.uuid4())
    cfg.task.env.numEnvs = 35
    
    # Set the reward type
    cfg.train.learn.reward_type = "OT"
    # Set the encoder type
    cfg.train.learn.encoder_type = "resnet"
     
    # Parse the config
    cfg_dict = omegaconf_to_dict(cfg)
    # # Overwrite the obs encoder
    # cfg_dict["train"]["preference_encoder"]["pretrain_dir"] = DIR_PATH + "/mvp_exp_data/mae_encoders/"
    
    print_dict(cfg_dict)

    #For test mode only, use only one environment
    # cfg.logdir = "/home/thomastian/workspace/mvp_exp_data/rl_runs/9_14_paper_results_RLHF_franka/300/8a5e66a4-f3af-4a1f-be8b-d7f5768f88bc/"
    # cfg.test = True
    # cfg.headless = False
    # cfg.resume = 2000
    # cfg.task.env.numEnvs = 50
    # cfg_dict = omegaconf_to_dict(cfg)

    # Create logdir and dump cfg
    if not cfg.test:
        os.makedirs(cfg.logdir, exist_ok=True)
        dump_cfg(cfg, cfg.logdir)

    seed = cfg.train.seed * cfg.num_gpus + local_rank
    set_np_formatting()
    set_seed(seed)
    # set_np_formatting()
    # set_seed(cfg.train.seed, cfg.train.torch_deterministic)
    # Construct task
    sim_params = parse_sim_params(cfg, cfg_dict)
    env = parse_task(cfg, cfg_dict, sim_params) # This is the vec_env

    # Perform training
    ppo = process_ppo(env, cfg, cfg_dict, cfg.logdir, cfg.cptdir)
    ppo.run(num_learning_iterations=cfg.train.learn.max_iterations, log_interval=cfg.train.learn.save_interval)

    # Clean up
    if cfg.num_gpus > 1:
        dist.destroy_process_group()
if __name__ == '__main__':
    train()
