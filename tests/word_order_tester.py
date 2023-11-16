import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



import torchvision

import clip
from clips.hf_clip import HFClip

import torchvision.datasets as dset
from src.config import *
from torch.utils.data import DataLoader, Subset
from src.utils import collate_fn, do_validation
import random
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, preprocess = clip.load(training_hyperparameters['openai_clip_model'], device=device)

model = HFClip().to(device)


train_dataset = dset.CocoCaptions(root = './datasets/mscoco/val2014',
    annFile = 'datasets/mscoco/annotations/captions_val2014.json',
    # transform=[transforms.PILToTensor()])
    transform=preprocess,)
 
# create dataloader with first 1k images
subset_indices = torch.arange(0, 100)
train_data_subset = Subset(train_dataset, subset_indices)
train_dataloader = DataLoader(train_data_subset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=0)


dataloader = train_dataloader

# do_validation(dataloader, model)
# set random seed
torch.manual_seed(42)
random.seed(42)




num_corrects = 0

total = 0

for (imgs, caps) in tqdm(train_dataloader):
    with torch.no_grad():

        # print('caps before ', caps)



        # caps here only has one caption. Shuffle the words in the caption to create 5 new captions
        org_caption_list = caps[0].split()

        shuffled_captions = [caps[0]]

        # shuffle the words in the caption

        for i in range(5):
            random.shuffle(org_caption_list)
            
            shuffled_captions.append(' '.join(org_caption_list))

        # print('shuffled_caps ', shuffled_captions)

        outputs = model(imgs, shuffled_captions, output_loss=False, return_all=True)

        logits_per_image = outputs.logits_per_image
        # correct label is always the first one, check if this is the case
        label_probs = logits_per_image.softmax(dim=-1)

        print('label_probs ', label_probs[0])

        if torch.argmax(label_probs[0]) == 0:
            num_corrects += 1
        total += 1


print('num_corrects / total', num_corrects / total)

print('num_corrects ', num_corrects)
print('total ', total)





