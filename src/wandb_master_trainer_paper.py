import sys
import os

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# add sibling directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb


from src.utils import generate_csv_file_name, cleanup_after_training

import src.config as config

config.config_cuda_device = 'cuda:0' 

training_hyperparameters = config.training_hyperparameters

training_hyperparameters['cuda_device'] = config.config_cuda_device
ClipDatasets = config.ClipDatasets

import train_clip


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
        "method": "grid",
        "name": "Gap persists even after accounting for all factors, scheduling, T=0.01, POST PAPER, visualizing embeds",
        "metric": {"goal": "minimize", "name": "train_intermodality_loss"},
        "parameters": {
            "temperature": {"values": [0.01]}, # learnable temperature now, so this i s the starting temp
            'learnable_temperature': {'values': [False]},

            
            # CUDA: 0


            # TRAINING STUFF
            'encoder1_modality': {'values': ['image']},
            'encoder2_modality': {'values': ['image']},
            'same_inputs': {'values': [True]},
            
            'clip_projection_dim': {'values': [512]}, # 512
            'weight_decay': {'values': [0.1]},
            'batch_size': {'values': [128]},
            
            'vision_model': {'values': ['VIT']}, # RN50 or VIT

            'use_scheduler': {'values': ['EXP']},
            'schedule_every': {'values': [50]}, # num steps, NOT epochs


            'n_warmup_steps': {'values': [100]}, # 10000
            'W_layer_gap': {'values': [0]}, # 0 means no gap, 1 means full gap. -1 means no W layer
            
            "lr": {'values': [1e-4]}, # 5e-4, from CyClip paper
            'n_epochs': {'values': [1000]}, 
            'num_workers': {'values': [12]}, # SET



            # LOSS STUFF
            'intra_modality_loss': {'values': [False]},
            'uniformity_loss': {'values': [False]},
            'alignment_loss': {'values': [False]},
            'cross_uniformity_loss': {'values': [False]},
            'remove_contrastive_loss': {'values': [False]},
            'cyclip_loss': {'values': [False]},



           
            'use_train_as_val': {'values': [True]}, # SET

            'train_from_scratch': {'values': [True]},
            'continue_from_checkpoint': {'values': [False]},
            'train_from_pretrained': {'values': [False]},
            'finetune_multi_layer_projection': {'values': [False]},

           

            # DATASET STUFF
            'dataset': {'values': [ClipDatasets.MSCOCO.value]},
            'validation_dataset_size': {'values': [2048]},
            'validation_batch_size': {'values': [2048]},
            'use_small_trainloader': {'values': [True]}, 
            'small_train_loader_dataset_size': {'values': [2048]},
            'cifar10_acc': {'values': [False]}, # Don't measure cifar10 val acc for these toy runs
            'save_encoder_hidden_states': {'values': [True]},
            'n_embeds_to_save': {'values': [1000]},
            'save_every': {'values': [200]},

            
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
 


