'''
After starting a sweep, run this to parallelize the sweep over multiple GPUs
'''

import sys
import os

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# add sibling directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.config as config

config.config_cuda_device = 'cuda:4' # SET the cuda device to be used in host4

from src.wandb_master_trainer import main

import wandb

from src.utils import cleanup_after_training
import traceback

import train_clip

def main():

    
    wandb.init() 

    try:
        # do training
        train_clip.main()
        wandb.finish() 
    except Exception as e:
        # print('Exception in training ', e.with_traceback())

        print('Exception in master trainer ')
        traceback.print_exc()
        cleanup_after_training()
        wandb.finish()
        # delete cache batches
        return 



            
sweep_id = 'hl3u3vxp'

# wandb.agent(sweep_id='nrjuh2de', function=main, project="clipverse")
wandb.agent(sweep_id=sweep_id, function=main, project="clipverse")
 

