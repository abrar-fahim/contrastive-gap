import sys
import os

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# add sibling directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb

import train_clip
from src.utils import generate_csv_file_name, cleanup_after_training

from src.config import training_hyperparameters

    


def set_sweep_config(training_hyperparameters: dict, sweep_config: dict) -> dict:
    # set correct key values from wandb

    for key in training_hyperparameters.keys():
        if key not in sweep_config['parameters']:
            sweep_config['parameters'][key] = {'value': training_hyperparameters[key]}
    return sweep_config


def main():

    
    wandb.init() 

    # print('wandb config ', wandb.config)

    # set_hypers() # no need to set hypers anymore, wandb automatically does this



    # in case train_clip.py throws error, we can still finish the run

    
    try:
        # do training
        train_clip.main()
        wandb.finish() 
    except Exception as e:
        print('Exception in training ', e)
        cleanup_after_training()
        wandb.finish()
        # delete cache batches
        return 



    # do training
    # train_clip.main()
    # wandb.finish() 

# if main 
if __name__ == "__main__":


    sweep_configuration = {
        # "method": "grid",
        "method": "bayes",
        # "method": "random",
        # "name": "Checking AGAIN whether same inputs cause modality gap or no",
        "name": "default loss shuffle on 512D, 128b, full MSCOCO, tuning lr, wdecay=0.35",
        "metric": {"goal": "maximize", "name": "val_image_classification_accuracy"},
        "parameters": {
            "temperature": {"values": [0.01]},
            "encoder1_modality": {"values": ["image"]},
            "encoder2_modality": {"values": ["text"]},

            'clip_projection_dim': {'values': [512]}, # 512

            'intra_modality_loss': {'values': [False]},
            'uniformity_loss': {'values': [False]},
            'weight_decay': {'values': [0.35]}, 

            "lr": {"max": 1e-3, "min": 5e-5},
            # "lr": {'values': [0.000015]}, # 1.5e-5, optimized for 0.01 temp
            # "lr": {'values': [5e-4]}, # 5e-4, from CyClip paper

            # "lr": {'values': [1e-6, 1e-5, 5e-5, 1e-4 ]}, # 1.5e-5, optimized for 0.01 temp
            # 'seed': {'values': [42, 10, 100]},
            'seed': {'values': [2]},



        },
    }

    sweep_configuration = set_sweep_config(training_hyperparameters, sweep_configuration)



    sweep_id = wandb.sweep(sweep=sweep_configuration, project="clipverse")

    print()
    print('--- SWEEP ID ---')
    print(sweep_id)
    print()


    # wandb.agent(sweep_id='nrjuh2de', function=main, project="clipverse")
    wandb.agent(sweep_id=sweep_id, function=main, project="clipverse")
 


