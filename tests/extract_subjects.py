'''
Setup evalutation metric for image+text queries using CLIP embeddings
'''



import sys
import os

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# add sibling directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nltk import pos_tag
from dataset_processors.mscoco_processor import MSCOCOProcessor
from tqdm import tqdm
import torch
import random


'''
1. Sample all subjects from sentutences in dataset
'''

# set seed
torch.manual_seed(42)
random.seed(42)


dataset_processor = MSCOCOProcessor(return_only_captions=True)

train_dataloader = dataset_processor.train_dataloader

pos_to_get = ['NN', 'NNP',]

# sample all subjects from sentences in dataloader
all_subjects = []
for batch in tqdm(train_dataloader):
    original_captions = batch
    for caption in original_captions:
        split_text = caption.split()    
        tokens_tag = pos_tag(split_text)
        sub_toks = [tok[0] for tok in tokens_tag if (tok[1] in pos_to_get) ]
        if len(sub_toks) > 0:
            all_subjects.append(sub_toks[0].lower())


# print(all_subjects)

# get unique subjects
unique_subjects = list(set(all_subjects))
# print(unique_subjects)

# save unique subjects to file
with open('unique_subjects.txt', 'w') as f:
    for subject in unique_subjects:
        f.write(subject + '\n')