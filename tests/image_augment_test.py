from torchvision.transforms import v2
import wandb
import torch
# import cv2
from matplotlib import pyplot as plt




import sys
import os


# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# add sibling directory to path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))# def 

from dataset_processors.mscoco_processor import MSCOCOProcessor
from src.config import training_hyperparameters

wandb.init(config=training_hyperparameters)

torch.manual_seed(wandb.config['seed'])

mscoco = MSCOCOProcessor()

for img1, img2 in mscoco.train_dataloader:

    # img1 shape: ([32, 3, 224, 224])

    # display img1[0] and img2[0] side by side
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img1[0].permute(1, 2, 0) )
    plt.subplot(1, 2, 2)
    plt.imshow(img2[0].permute(1, 2, 0))
    plt.show()






    








    break