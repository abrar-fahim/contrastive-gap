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

VISUALIZE = False




# set seed
torch.manual_seed(training_hyperparameters['seed'])
random.seed(training_hyperparameters['seed'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 16
shuffle_ratio = 0.5 # percentage of texts and images to shuffle


clip_model = HFClip().to(device) # checkpoint CLIP

dataset_processor = MSCOCOProcessor()

if not VISUALIZE:
    do_validation(dataset_processor, clip_model)

else:

    val_dataset = dataset_processor.val_dataset

    # creating dataloader seperately here instead of using the one inside dataset_processor to set the manual seed explicitly so that I get same batch each time
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, collate_fn=dataset_processor.collate_fn, generator=torch.Generator().manual_seed(training_hyperparameters['seed']))

    with torch.no_grad():

        for batch in val_dataloader:
            # (val_imgs, val_captions) = next(iter(val_dataloader))
            (val_imgs, val_captions) = batch
        val_outputs = clip_model(val_imgs, val_captions, output_loss=False, return_all=True)


        # text-text
        text_encoder_outputs = clip_model.encode_text(val_captions) # shape: ([batch_size, 512])

        # normalize features
        text_encoder_outputs = text_encoder_outputs / torch.norm(text_encoder_outputs, dim=1, keepdim=True)

        # cosine similarities between text-text pairs
        text_text_cosine_similarities = text_encoder_outputs @ text_encoder_outputs.t() # shape: ([batch_size, batch_size])

        # heatmap of cosine similarities
        plt.imshow(text_text_cosine_similarities.cpu().numpy())
        # show legend
        # set title
        plt.title('text-text cosine similarities')
        plt.colorbar()
        plt.show()

        # get elements in lower traingle excluding diagonal

        text_text_cosine_similarity = text_text_cosine_similarities[torch.tril(torch.ones(text_text_cosine_similarities.shape[0], text_text_cosine_similarities.shape[1]), diagonal=-1).bool()]

        # image-image

        image_encoder_outputs = clip_model.encode_image(val_imgs) # shape: ([batch_size, 512])

        # normalize features
        image_encoder_outputs = image_encoder_outputs / torch.norm(image_encoder_outputs, dim=1, keepdim=True)

        # cosine similarities between image-image pairs
        image_image_cosine_similarities = image_encoder_outputs @ image_encoder_outputs.t() # shape: ([batch_size, batch_size])

        # heatmap of cosine similarities
        plt.imshow(image_image_cosine_similarities.cpu().numpy())
        # show legend
        plt.colorbar()
        # set title
        plt.title('image-image cosine similarities')
        plt.show()







        # get elements in lower traingle excluding diagonal

        image_image_cosine_similarity = image_image_cosine_similarities[torch.tril(torch.ones(image_image_cosine_similarities.shape[0], image_image_cosine_similarities.shape[1]), diagonal=-1).bool()]

        # compute spearman correlation between text-text and image-image cosine similarities
        result = stats.spearmanr(text_text_cosine_similarity.cpu(), image_image_cosine_similarity.cpu())

        print('spearman correlation between text-text and image-image cosine similarities: ', result.statistic)
        print('p value: ', result.pvalue)

        # after shuffling

        # get random indices 

        num_elements_to_switch = int(shuffle_ratio * text_encoder_outputs.shape[0])

        random_indices = random.sample(range(text_encoder_outputs.shape[0]), num_elements_to_switch)

        # switch elements at random indices

        text_encoder_outputs[random_indices], image_encoder_outputs[random_indices] = image_encoder_outputs[random_indices], text_encoder_outputs[random_indices]

        # cosine similarities between text-text pairs
        text_text_cosine_similarities = text_encoder_outputs @ text_encoder_outputs.t() # shape: ([batch_size, batch_size])

        # plot heatmap of cosine similarities
        plt.imshow(text_text_cosine_similarities.cpu().numpy())
        # show legend
        plt.colorbar()
        # set title
        plt.title('text-text cosine similarities after shuffling')
        plt.show()


        # get elements in lower traingle excluding diagonal
        text_text_cosine_similarity = text_text_cosine_similarities[torch.tril(torch.ones(text_text_cosine_similarities.shape[0], text_text_cosine_similarities.shape[1]), diagonal=-1).bool()]

        # cosine similarities between image-image pairs
        image_image_cosine_similarities = image_encoder_outputs @ image_encoder_outputs.t() # shape: ([batch_size, batch_size])

        # plot heatmap of cosine similarities
        plt.imshow(image_image_cosine_similarities.cpu().numpy())
        # show legend
        plt.colorbar()
        # set title
        plt.title('image-image cosine similarities after shuffling')
        plt.show()
        

        # get elements in lower traingle excluding diagonal
        image_image_cosine_similarity = image_image_cosine_similarities[torch.tril(torch.ones(image_image_cosine_similarities.shape[0], image_image_cosine_similarities.shape[1]), diagonal=-1).bool()]

        # compute spearman correlation between text-text and image-image cosine similarities
        result = stats.spearmanr(text_text_cosine_similarity.cpu(), image_image_cosine_similarity.cpu())
        print('spearman correlation between text-text and image-image cosine similarities after shuffling: ', result.statistic)
        print('p value: ', result.pvalue)















