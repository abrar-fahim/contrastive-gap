import sys
import os

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# add sibling directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_clip

import src.config as config

import wandb



def set_hypers():
    config.training_hyperparameters['seed'] = wandb.config.seed
    config.training_hyperparameters['temperature'] = wandb.config.temperature
    config.training_hyperparameters['intra_modality_temperature'] = wandb.config.temperature
    config.training_hyperparameters['intra_modality_loss'] = wandb.config.intra_modality_loss
    config.training_hyperparameters['rsa_loss'] = wandb.config.rsa_loss
    config.training_hyperparameters['pearson_loss'] = wandb.config.pearson_loss
    config.training_hyperparameters['lr'] = wandb.config.lr
    config.training_hyperparameters['encoder1_modality'] = wandb.config.encoder1_modality
    config.training_hyperparameters['encoder2_modality'] = wandb.config.encoder2_modality
    config.training_hyperparameters['second_caption_offset'] = wandb.config.second_caption_offset
    config.training_hyperparameters['same_encoder'] = wandb.config.same_encoder
    config.training_hyperparameters['same_inputs'] = wandb.config.same_inputs

    # reload config
    # importlib.reload(config)
    # importlib.reload(train_clip)






sweep_configuration = {
    "method": "grid",
    # "method": "random",
    "name": "Text only configs AFTER CODE CLEAN UP, 1024 batch size",
    "metric": {"goal": "maximize", "name": "val_image_classification_accuracy"},
    "parameters": {
        "temperature": {"values": [0.01]},
        # "intra_modality_loss": {"values": [True, False]},
        "intra_modality_loss": {"values": [False]},
        "rsa_loss": {"values": [False]},
        "pearson_loss": {"values": [False]},
        "training_hyperparameters": {"values": [config.training_hyperparameters]}, # just to keep track of hypers used for this sweep.
        "encoder1_modality": {"values": ["text", "image"]},
        "encoder2_modality": {"values": ["text", "image"]},
        "same_encoder": {"values": [True, False]},
        "same_inputs": {"values": [False, True]},
        'second_caption_offset': {'values': [False, True]},

        # "lr": {"max": 7e-5, "min": 1e-6},
        "lr": {'values': [0.000015]}, # 1.5e-5, optimized for 0.01 temp
        # "lr": {'values': [1e-6, 1e-5, 5e-5, 1e-4 ]}, # 1.5e-5, optimized for 0.01 temp
        # 'seed': {'values': [42, 10, 100]},
        'seed': {'values': [2]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="clipverse")



def main():
    wandb.init()

    if wandb.config.encoder1_modality != wandb.config.encoder2_modality:
        if wandb.config.same_encoder or wandb.config.same_inputs or wandb.config.second_caption_offset:
            print("Can't set same_encoder, same_inputs, second_caption_offset to True when modalites are different")
            return
        
    



    set_hypers()

    # do training
    train_clip.main()
    wandb.finish() 


# wandb.agent(sweep_id='nrjuh2de', function=main, project="clipverse")
wandb.agent(sweep_id=sweep_id, function=main, project="clipverse")
 