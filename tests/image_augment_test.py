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

from src.utils import generate_csv_file_name

from dataset_processors.mscoco_processor import MSCOCOProcessor
from src.config import training_hyperparameters

wandb.init(config=training_hyperparameters)

torch.manual_seed(wandb.config['seed'])

mscoco = MSCOCOProcessor()


mscoco_val_dataloader = torch.utils.data.DataLoader(mscoco.val_dataset, batch_size=128, collate_fn=mscoco.collate_fn, generator=torch.Generator().manual_seed(wandb.config['seed']))

# mscoco_batch_file_path = f"datasets/mscoco/val_batch_cache_{generate_csv_file_name()}.pt"
mscoco_val_batch_file_path = f"datasets/mscoco/val_batch_cache_T0.07_Lit_2_scratch_I1C2E1E2_512.pt"


mscoco_train_batch_file_path = f"datasets/mscoco/val_batch_cache_T0.07_Lit_2_scratch_I1C2E1E2_512_train_as_val_2048.pt"



i = 2

for imgs, caps in mscoco.train_dataloader:
# for imgs, caps in mscoco_val_dataloader:
    
    train_imgs, train_caps = torch.load(mscoco_val_batch_file_path)

    val_imgs, val_caps = torch.load(mscoco_train_batch_file_path)

    # img1 shape: ([32, 3, 224, 224])

    print('train cap0, ', train_caps)

    print('val cap0, ', val_caps)

    

    # display 10 imgs from train and val side by side
    plt.figure()
    for j in range(10):
        plt.subplot(2, 10, j + 1)
        plt.imshow(train_imgs[j].permute(1, 2, 0))
        plt.subplot(2, 10, j + 11)
        plt.imshow(val_imgs[j].permute(1, 2, 0))
    plt.show()


    break



# for img1, img2 in mscoco.train_dataloader:

#     # img1 shape: ([32, 3, 224, 224])

#     # display img1[0] and img2[0] side by side
#     plt.figure()
#     plt.subplot(1, 2, 1)
#     plt.imshow(img1[0].permute(1, 2, 0) )
#     plt.subplot(1, 2, 2)
#     plt.imshow(img2[0].permute(1, 2, 0))
#     plt.show()



#     break