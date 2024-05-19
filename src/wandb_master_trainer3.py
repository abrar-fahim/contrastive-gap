import sys
import os

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# add sibling directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb

from src.utils import generate_csv_file_name, cleanup_after_training

import src.config as config




config.config_cuda_device = 'cuda:2' 

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

# if main 
if __name__ == "__main__":


    sweep_configuration = {
        "method": "grid",
        # "method": "bayes",
        # "method": "random",``
        # "name": "Checking AGAIN whether same inputs cause modality gap or no",
        "name": "LR TUNE RUN from pretrained CLIP (finetuning CLIP backbone), VIT/B-32, CLIP default no weights  loss batch_size=64 64D, full MSCOCO, val as val, 0.01T",
        # "metric": {"goal": "maximize", "name": "val_image_classification_accuracy"},
        "metric": {"goal": "minimize", "name": "train_intermodality_loss"},
        "parameters": {
            "temperature": {"values": [0.01]}, # learnable temperature now, so this is the starting temp
            'learnable_temperature': {'values': [False]},

            # CUDA: 2

            # TRAINING STUFF
            'clip_projection_dim': {'values': [64]}, # 512
            'batch_size': {'values': [64]},
            'vision_model': {'values': ['VIT']}, # RN50 or VIT or VIT16
            'use_scheduler': {'values': ['EXP']},
            'schedule_every': {'values': [200]}, # num steps, NOT epochs
            'n_warmup_steps': {'values': [10000]},
            'weight_decay': {'values': [0.1]},
            'train_from_scratch': {'values': [False]},
            'continue_from_checkpoint': {'values': [False]},
            'train_from_pretrained': {'values': [True]},
            'finetune_clip_backbone': {'values': [True]},
            'finetune_multi_layer_projection': {'values': [False]},



            # LOSS STUFF
            'intra_modality_loss': {'values': [False]},
            'uniformity_loss': {'values': [False]},
            'alignment_loss': {'values': [False]},
            'cross_uniformity_loss': {'values': [False]},
            'remove_contrastive_loss': {'values': [False]},
            'cyclip_loss': {'values': [False]},
            # 'weight_decay': {'min': 0.2, 'max': 0.6,},


            # "lr": {"max": 2e-4, "min": 4e-5},and
            # "lr": {'values': [0.000015]}, # 1.5e-5, optimized for 0.01 temp
            "lr": {'values': [1e-5]}, # 5e-4, from CyClip paper
            'n_epochs': {'values': [25]},
            'num_workers': {'values': [12]},
            'zero_shot_acc_num_workers': {'values': [4]},

            # DATASET STUFF
            'dataset': {'values': [ClipDatasets.MSCOCO.value]},
            'validation_dataset_size': {'values': [512]},
            'validation_batch_size': {'values': [512]},
            'use_small_trainloader': {'values': [False]}, 
            'cifar10_acc': {'values': [True]}, 
            'use_train_as_val': {'values': [False]}, # SET

            'save_encoder_hidden_states': {'values': [False]},
            'n_embeds_to_save': {'values': [512]},

            'seed': {'values': [42]},
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
 


