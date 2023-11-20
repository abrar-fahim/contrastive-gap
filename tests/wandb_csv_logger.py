import wandb
import pandas as pd

import sys
import os

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# add sibling directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import training_hyperparameters

FILENAME = "experiments.csv"
loaded_experiment_df = pd.read_csv(FILENAME)

start_line = 29




PROJECT_NAME = "Converted Experiments"

EXPERIMENT_NAME_COL = "Experiment"
NOTES_COL = "Notes"
TAGS_COL = "Tags"
CONFIG_COLS = ["Num Layers"]
SUMMARY_COLS = ["Final Train Acc", "Final Val Acc"]
METRIC_COLS = ["Training Losses"]



for i, row in loaded_experiment_df.iterrows():

    val_image_accuracy = row['val_image_accuracy']
    train_image_accuracy = row['train_image_accuracy']
    cosine_sim_metric = row['cosine_similarity_metric']
    train_loss = row['train_loss']
    step = int(row['epoch']) * 100 + int(row['index'])

    run = wandb.init(project='clipverse')

    run.log(
        data={
            'val_image_accuracy': val_image_accuracy.item(),
            'train_image_accuracy': train_image_accuracy.item(),
            'cosine_sim_metric': cosine_sim_metric.item(),
            'train_loss': train_loss,
            
        },
        step= step

    )
    run.finish()