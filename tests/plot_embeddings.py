import torch

import sys
import os

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# add sibling directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.utils import plot_embeddings
from clips.hf_clip import HFClip
from dataset_processors.mscoco_processor import MSCOCOProcessor


# if main
if __name__ == '__main__':

    # load clip model
    clip_model = HFClip()
    # get mscoco dataset processor
    dataset_processor = MSCOCOProcessor()

    val_dataset = dataset_processor.val_dataset

    # dataloader
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1024,
                                            collate_fn=dataset_processor.collate_fn,
                                            generator=torch.Generator().manual_seed(42))

    plot_embeddings(clip_model, val_dataloader)
