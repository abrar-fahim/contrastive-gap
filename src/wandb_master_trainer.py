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


sweep_configuration = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "val_acc"},
    "parameters": {
        "temperature": {"values": [0.1, 0.01]},
        "intra_modality_loss": {"values": [True, False]},
        "lr": {"max": 1e-3, "min": 1e-6},
        'seed': {'values': [42, 10, 100]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="clipverse")


def main():
    wandb.init()

    set_hypers()

    # do training
    train_clip.main()


wandb.agent(sweep_id, function=main, count=20)

def set_hypers():
    config.training_hyperparameters['seed'] = wandb.config.seed
    config.training_hyperparameters['temperature'] = wandb.config.temperature
    config.training_hyperparameters['intra_modality_temperature'] = wandb.config.temperature
    config.training_hyperparameters['intra_modality_loss'] = wandb.config.intra_modality_loss
    config.training_hyperparameters['lr'] = wandb.config.lr

    # reload config
    importlib.reload(config)
    importlib.reload(train_clip)

    # run training
    train_clip.main()



