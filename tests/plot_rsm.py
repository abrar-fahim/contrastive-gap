import torch

import sys
import os

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# add sibling directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clips.hf_clip import HFClip
import torch.optim as optim
from src.config import *

import random
from dataset_processors.mscoco_processor import MSCOCOProcessor

from scipy import stats
from matplotlib import pyplot as plt
from src.utils import do_validation
from tqdm import tqdm


def plot_scatter(image_image_cosine_similarity, text_text_cosine_similarity, title, interchanged_indices=None, n=None, color=None, x='', y=''):
    # plot scatter plot of image-image and text-text cosine similarities

    if interchanged_indices is None:
        plt.scatter(image_image_cosine_similarity.cpu().numpy(), text_text_cosine_similarity.cpu().numpy(), c=color)

        plt.xlabel(x)
        plt.ylabel(y)

        plt.show()


    else:

        dummy_rsm = torch.zeros(n, n)

        print('interchanged_indices: ', interchanged_indices)

        # change dummy rsm rows and columns at interchanged indices to 1
        dummy_rsm[interchanged_indices, :] += torch.ones(1, n)
        dummy_rsm[:, interchanged_indices] += torch.ones(n, 1)

        # image-text overlaps will now be 1
        # same-modality overlaps will now be 2



        print('dummy_rsm: ', dummy_rsm)

        dummy_rsm_tril = dummy_rsm[torch.tril(torch.ones(dummy_rsm.shape[0], dummy_rsm.shape[1]), diagonal=-1).bool()]

        print('dummy_rsm_tril: ', dummy_rsm_tril)




        colors = ['purple'] * image_image_cosine_similarity.shape[0] # non interchanged indices are blue
        for i, c in enumerate(dummy_rsm_tril):
            if c == 1:
                colors[i] = 'orange' # interchanged indices are red
            elif c == 2:
                colors[i] = 'green' # same modality indices are green
        plt.scatter(image_image_cosine_similarity.cpu().numpy(), text_text_cosine_similarity.cpu().numpy(), c=colors)

    # set axis limits
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    # set title
    plt.title(title)
    plt.show()


# set seed
torch.manual_seed(training_hyperparameters['seed'])
random.seed(training_hyperparameters['seed'])
device = torch.device(training_hyperparameters['cuda_device'] if torch.cuda.is_available() else "cpu")

batch_size = 256
shuffle_ratio = 0.5 # percentage of texts and images to shuffle


clip_model = HFClip().to(device) # checkpoint CLIP

dataset_processor = MSCOCOProcessor()


print('VISUALIZING')

val_dataset = dataset_processor.val_dataset

# creating dataloader seperately here instead of using the one inside dataset_processor to set the manual seed explicitly so that I get same batch each time
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, collate_fn=dataset_processor.collate_fn, generator=torch.Generator().manual_seed(training_hyperparameters['seed']))

with torch.no_grad():

    for batch in val_dataloader:
        # (val_imgs, val_captions) = next(iter(val_dataloader))
        (val_imgs, val_captions) = batch
    val_outputs = clip_model(val_imgs, val_captions, output_loss=False, return_all=True)

    # cosine similarities between image-text pairs
    logits = val_outputs.logits_per_image

    # scale logits with temp
    logits = logits * clip_model.temperature


    '''
    image-text
    '''

    # cosine similarities between image-text pairs
    image_text_cosine_similarities = val_outputs.logits_per_image * clip_model.temperature # shape: ([batch_size, batch_size])

    image_text_RSM = image_text_cosine_similarities[torch.tril(torch.ones(image_text_cosine_similarities.shape[0], image_text_cosine_similarities.shape[1]), diagonal=-1).bool()]

    '''
    1. text-text
    '''
    text_encoder_outputs = clip_model.encode_text(val_captions) # shape: ([batch_size, 512])

    # normalize features
    text_encoder_outputs = text_encoder_outputs / torch.norm(text_encoder_outputs, dim=1, keepdim=True)

    # cosine similarities between text-text pairs
    text_text_cosine_similarities = text_encoder_outputs @ text_encoder_outputs.t() # shape: ([batch_size, batch_size])

    # get elements in lower traingle excluding diagonal

    text_RSM = text_text_cosine_similarities[torch.tril(torch.ones(text_text_cosine_similarities.shape[0], text_text_cosine_similarities.shape[1]), diagonal=-1).bool()]
    
    '''
    2. Image-image
    '''

    image_encoder_outputs = clip_model.encode_image(val_imgs) # shape: ([batch_size, 512])

    # normalize features
    image_encoder_outputs = image_encoder_outputs / torch.norm(image_encoder_outputs, dim=1, keepdim=True)

    # cosine similarities between image-image pairs
    image_image_cosine_similarities = image_encoder_outputs @ image_encoder_outputs.t() # shape: ([batch_size, batch_size])


    # get elements in lower traingle excluding diagonal

    image_RSM = image_image_cosine_similarities[torch.tril(torch.ones(image_image_cosine_similarities.shape[0], image_image_cosine_similarities.shape[1]), diagonal=-1).bool()]

    '''
    3. Plot
    '''

    # plot scatter plot of intra and inter modality RSMs

    plot_scatter(image_RSM, image_text_RSM, 'Image-Image vs Image-Text', x='Image-Image', y='Image-Text')








    