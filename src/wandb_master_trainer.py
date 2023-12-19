import sys
import os

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# add sibling directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_clip

import src.config as config
import importlib

import wandb


def set_hypers():
    config.training_hyperparameters['seed'] = wandb.config.seed
    config.training_hyperparameters['temperature'] = wandb.config.temperature
    config.training_hyperparameters['intra_modality_temperature'] = wandb.config.temperature
    config.training_hyperparameters['intra_modality_loss'] = wandb.config.intra_modality_loss
    config.training_hyperparameters['lr'] = wandb.config.lr

    # reload config
    # importlib.reload(config)
    # importlib.reload(train_clip)






sweep_configuration = {
    "method": "grid",
    "name": "sweep_temp_imloss_consistency",
    "metric": {"goal": "maximize", "name": "val_image_classification_accuracy"},
    "parameters": {
        "temperature": {"values": [0.1, 0.01]},
        "intra_modality_loss": {"values": [True, False]},
        # "lr": {"max": 7e-5, "min": 1e-6},
        "lr": {'values': [0.000015]}, # 1.5e-5, optimized for 0.01 temp
        'seed': {'values': [42, 10, 100]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="clipverse")


def main():
    wandb.init()

    set_hypers()

    # do training
    train_clip.main()
    wandb.finish()


wandb.agent(sweep_id, function=main)
