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
from src.validate import do_validation
from tqdm import tqdm

VISUALIZE = True

heatmap = False

hist = False

scatter = True

image_text_matching = False
image_text_non_matching = False


compare_corresponding_RSMs = False
compare_inter_intra_RMSs = True


def plot_heatmap(cosine_similarities, title):
    # plot heatmap of cosine similarities
    plt.imshow(cosine_similarities.cpu().numpy())
    # show legend
    plt.colorbar()
    # set title
    plt.title(title)
    plt.show()

def plot_histogram(cosine_similarities, title):
    # plot histogram of cosine similarities
    plt.hist(cosine_similarities.cpu().numpy(), bins=100)
    # set axis limits
    plt.xlim(-1, 1)
    # set title
    plt.title(title)
    plt.show()

def plot_scatter(image_image_cosine_similarity, text_text_cosine_similarity, title, interchanged_indices=None, n=None, color=None):
    # plot scatter plot of image-image and text-text cosine similarities

    if interchanged_indices is None:
        plt.scatter(image_image_cosine_similarity.cpu().numpy(), text_text_cosine_similarity.cpu().numpy(), c=color)
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

if not VISUALIZE:
    print('VALIDATING WITHOUT VISUALIZING')
    do_validation(dataset_processor, clip_model)

else:

    print('VISUALIZING')

    val_dataset = dataset_processor.val_dataset

    # creating dataloader seperately here instead of using the one inside dataset_processor to set the manual seed explicitly so that I get same batch each time
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, collate_fn=dataset_processor.collate_fn, generator=torch.Generator().manual_seed(training_hyperparameters['seed']))

    with torch.no_grad():

        for batch in val_dataloader:
            # (val_imgs, val_captions) = next(iter(val_dataloader))
            (val_imgs, val_captions) = batch
        val_outputs = clip_model(val_imgs, val_captions, output_loss=False, return_all=True)

        '''
        0. image-text histogram
        '''

        # cosine similarities between image-text pairs
        logits = val_outputs.logits_per_image

        # scale logits with temp
        logits = logits * clip_model.temperature

        
        if image_text_matching:
           

            # get elements on the diagonal
            diagonal_image_text_cosine_similarity = logits[torch.eye(logits.shape[0]).bool()]

           
            # plot both histograms with different colors
            plt.hist(diagonal_image_text_cosine_similarity.cpu().numpy(), bins=100, color='blue')
           
            # set axis limits
            plt.xlim(-1, 1)
            # set title
            plt.title(f'image-text ON-diagonal cosine similarities histogram, T = {clip_model.temperature}')
            plt.show()

        if image_text_non_matching:
            # get elements excluding diagonal
            non_diagonal_image_text_cosine_similarity = logits[~torch.eye(logits.shape[0], dtype=bool)]
            plt.hist(non_diagonal_image_text_cosine_similarity.cpu().numpy(), bins=100, color='red')
            # set axis limits
            plt.xlim(-1, 1)
            # set title
            plt.title(f'image-text OFF-diagonal cosine similarities histogram, T = {clip_model.temperature}')
            plt.show()



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

        if heatmap:
            plot_heatmap(text_text_cosine_similarities, f'text-text cosine similarities, T = {clip_model.temperature}')


        # get elements in lower traingle excluding diagonal

        text_RSM = text_text_cosine_similarities[torch.tril(torch.ones(text_text_cosine_similarities.shape[0], text_text_cosine_similarities.shape[1]), diagonal=-1).bool()]
        
        if hist:
            plot_histogram(text_RSM, f'text-text cosine similarities (text_RSM) histogram, T = {clip_model.temperature}')

        '''
        2. Image-image
        '''

        image_encoder_outputs = clip_model.encode_image(val_imgs) # shape: ([batch_size, 512])

        # normalize features
        image_encoder_outputs = image_encoder_outputs / torch.norm(image_encoder_outputs, dim=1, keepdim=True)

        # cosine similarities between image-image pairs
        image_image_cosine_similarities = image_encoder_outputs @ image_encoder_outputs.t() # shape: ([batch_size, batch_size])
        
        if heatmap:
            plot_heatmap(image_image_cosine_similarities, f'image-image cosine similarities, T = {clip_model.temperature}')


        # get elements in lower traingle excluding diagonal

        image_RSM = image_image_cosine_similarities[torch.tril(torch.ones(image_image_cosine_similarities.shape[0], image_image_cosine_similarities.shape[1]), diagonal=-1).bool()]


        if hist:
            plot_histogram(image_RSM, f'image-image cosine similarities (image_RSM) histogram, T = {clip_model.temperature}')

        # compute spearman correlation between text-text and image-image cosine similarities
        result = stats.spearmanr(text_RSM.cpu(), image_RSM.cpu())

        if scatter:
            plot_scatter(image_RSM, text_RSM, f'image_RSM vs text_RSM , T = {clip_model.temperature}')

        print('spearman correlation between text_RSM and image_RSM: ', result.statistic)
        print('p value: ', result.pvalue)

        '''
        3. Interchanging text and image features
        '''

        # get random indices 

        num_elements_to_switch = int(shuffle_ratio * text_encoder_outputs.shape[0])

        random_indices = random.sample(range(text_encoder_outputs.shape[0]), num_elements_to_switch)

        # switch elements at random indices

        text_encoder_outputs[random_indices], image_encoder_outputs[random_indices] = image_encoder_outputs[random_indices], text_encoder_outputs[random_indices]

        '''
        4. Text-text
        '''

        # cosine similarities between text-text pairs
        interchanged_text_text_cosine_similarities = text_encoder_outputs @ text_encoder_outputs.t() # shape: ([batch_size, batch_size])


        if heatmap:
            plot_histogram(interchanged_text_text_cosine_similarities, f'text-text cosine similarities after interchanging, T = {clip_model.temperature}')


        # get elements in lower traingle excluding diagonal
        interchanged_text_RSM = interchanged_text_text_cosine_similarities[torch.tril(torch.ones(interchanged_text_text_cosine_similarities.shape[0], interchanged_text_text_cosine_similarities.shape[1]), diagonal=-1).bool()]


        if hist:
            plot_histogram(interchanged_text_RSM, f'text-text cosine similarities (interchanged_text_RSM)  histogram after interchanging, T = {clip_model.temperature}')

        '''
        5. image-image
        '''

        # cosine similarities between image-image pairs
        interchanged_image_image_cosine_similarities = image_encoder_outputs @ image_encoder_outputs.t() # shape: ([batch_size, batch_size])


        if heatmap:
            plot_heatmap(interchanged_image_image_cosine_similarities, f'image-image cosine similarities after interchanging, T = {clip_model.temperature}')
        

        # get elements in lower traingle excluding diagonal
        interchanged_image_RSM = interchanged_image_image_cosine_similarities[torch.tril(torch.ones(interchanged_image_image_cosine_similarities.shape[0], interchanged_image_image_cosine_similarities.shape[1]), diagonal=-1).bool()]

        if hist:
            plot_histogram(interchanged_image_RSM, f'image-image cosine similarities (interchanged_image_RSM) histogram after interchanging, T = {clip_model.temperature}')

        if scatter:
            if compare_corresponding_RSMs:
                plot_scatter(interchanged_image_RSM, interchanged_text_RSM, f'interchanged_image_RSM vs interchanged_text_RSM, T = {clip_model.temperature}', interchanged_indices=random_indices, n=text_encoder_outputs.shape[0])
                result = stats.spearmanr(interchanged_image_RSM.cpu(), interchanged_text_RSM.cpu())
            elif compare_inter_intra_RMSs:

               
                # plot_scatter(image_text_RSM, image_RSM, f'image_text_RSM vs image_RSM, T = {clip_model.temperature}', color='purple', n=text_encoder_outputs.shape[0])
                # plot_scatter(image_RSM, text_RSM, f'image_RSM vs text_RSM, T = {clip_model.temperature}', color='orange', n=text_encoder_outputs.shape[0])
                result = stats.spearmanr(image_RSM.cpu(), image_text_RSM.cpu())
                print('spearman correlation between image_RSM and image_text_RSM: ', result.statistic)
                print('p value: ', result.pvalue)
                result = stats.spearmanr(image_RSM.cpu(), text_RSM.cpu())
                print('spearman correlation between image_RSM and text_RSM: ', result.statistic)
                print('p value: ', result.pvalue)


                # plt.scatter(stackx.cpu().numpy(), stacky.cpu().numpy(), c=colors)
                plt.scatter(image_RSM.cpu().numpy(), text_RSM.cpu().numpy(), c='orange')
                plt.scatter(image_RSM.cpu().numpy(), image_text_RSM.cpu().numpy(), c='purple')
                
                
                plt.xlim(-1, 1)
                plt.ylim(-1, 1)

                # set x axis title
                plt.xlabel('Image RSM values')
                plt.ylabel('Text RSM / Image-Text RSM values')


                # set legend
                # plt.legend(['image_text_RSM vs image_RSM', 'image_RSM vs text_RSM'])
                plt.legend(['image_RSM vs text_RSM', 'image_RSM vs image_text_RSM'])

                # set title
                plt.title(f'inter-modality vs intra-modality RSMs, T = {clip_model.temperature}')
                plt.show()



            else:
                plot_scatter(interchanged_image_RSM, image_RSM, f'interchanged_image_RSM vs image_RSM, T = {clip_model.temperature}', interchanged_indices=random_indices, n=text_encoder_outputs.shape[0])
                result = stats.spearmanr(interchanged_image_RSM.cpu(), image_RSM.cpu())

        # compute spearman correlation between text-text and image-image cosine similarities
        
        print('spearman correlation between interchanged_image_RSM and image_RSM: ', result.statistic)
        print('p value: ', result.pvalue)