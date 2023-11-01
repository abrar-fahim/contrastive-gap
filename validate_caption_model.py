from training_utils import do_validation, collate_fn
import torch
from hf_clip import HFClip
from torch.utils.data import DataLoader, Subset
import clip
import torchvision.datasets as dset
import torchvision.transforms as transforms

from config import *


def main():
    # set seed
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    print('device ', device)

    model, preprocess = clip.load(training_hyperparameters['openai_clip_model'], device=device)




    train_dataset = dset.CocoCaptions(root = './datasets/mscoco/val2014',
                            annFile = 'datasets/mscoco/annotations/captions_val2014.json',
                            transform=transforms.PILToTensor() if clip_caption_model_train_hyperparameters['show_real_images'] else preprocess
                            # transform=preprocess
    )

    clip_model = HFClip().to(device)


    subset_indices = torch.randint(0, len(train_dataset) , (training_hyperparameters['small_train_loader_dataset_size'],)) # always defined and exists, but only used when small training loader is used, and we're not loading from checkpoint at start

    train_data_subset = Subset(train_dataset, subset_indices)

    train_dataloader = DataLoader(train_data_subset, batch_size=training_hyperparameters['small_train_loader_batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=0)

    dataloader = train_dataloader


    # get 100 indices that are NOT in train_data_subset

    # this doesnt technically ensure the same for when training clip captioning model, cuz thats a different random seed, FIX THIS LATER
    val_indices = torch.randint(0, len(train_dataset) , (training_hyperparameters['validation_dataset_size'],))
    j = 0
    while j < training_hyperparameters['validation_dataset_size']:
        while val_indices[j] in subset_indices:
            val_indices[j] = torch.randint(0, len(train_dataset) , (1,))
        j += 1
    print('j ', j)

    val_data_subset = Subset(train_dataset, val_indices)

    val_dataloader = DataLoader(val_data_subset, batch_size=training_hyperparameters['validation_batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=0)



    # just for now, create a dataloader with just the first 1k images from train dataset.
    # temp_dataset = Subset(train_dataset, torch.arange(0, clip_caption_model_train_hyperparameters['dataset_size']))

    # temp_dataloader = DataLoader(temp_dataset, batch_size=clip_caption_model_train_hyperparameters['dataset_size'], shuffle=True, collate_fn=collate_fn, num_workers=0)
    # shuffle here is irrelevant, since this is just shuffling minibatches during validation. I dont care about that. Also, torch random seed is set, so it should be reproducible.



    # dataloader = temp_dataloader

    for (imgs, captions) in dataloader:

        clip_model.eval()



        do_validation(val_dataloader, clip_model, index=0, captioning_model=True)

        break


if __name__ == '__main__':
    main()