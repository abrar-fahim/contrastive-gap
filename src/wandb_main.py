import sys
import os
# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# add sibling directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wandb

from src.utils import cleanup_after_training
import traceback



import train_clip


wandb.init() 

try:
    # do training
    train_clip.main()
    wandb.finish() 
except Exception as e:
    # print('Exception in training ', e.with_traceback())

    print('Exception in master trainer ')
    traceback.print_exc()
    cleanup_after_training()
    wandb.finish()
    # delete cache batches
    exit()


