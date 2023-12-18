import sys
import os

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# add sibling directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_clip

import src.config as config
import importlib

seeds_to_try = [42, 10, 100]

temps_to_try = [0.1, 0.01]

intra_modality_losses = [True, False]

for seed in seeds_to_try:
    for temp in temps_to_try:

        if temp == 0.01:
            # skip True intra_modality_loss for temp 0.01
            intra_modality_losses = [False]
        else:
            intra_modality_losses = [True, False]


        for intra_modality_loss in intra_modality_losses:

            print('seed ', seed)
            print('temp ', temp)
            print('intra_modality_loss ', intra_modality_loss)

            # set seed
            config.training_hyperparameters['seed'] = seed
            config.training_hyperparameters['temperature'] = temp
            config.training_hyperparameters['intra_modality_temperature'] = temp
            config.training_hyperparameters['intra_modality_loss'] = intra_modality_loss

            # run training
            train_clip.main()









