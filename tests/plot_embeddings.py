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
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import src.config as config



def plots():


    # plot 3d scatter plot in different subplots
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(image_coordinates_same[:, 0], image_coordinates_same[:, 1], image_coordinates_same[:, 2])
    ax.scatter(caption_coordinates_same[:, 0], caption_coordinates_same[:, 1], caption_coordinates_same[:, 2])
    ax.scatter(image_centroid_same[0], image_centroid_same[1], image_centroid_same[2], c='r')
    ax.scatter(caption_centroid_same[0], caption_centroid_same[1], caption_centroid_same[2], c='r')
    ax.set_title('Same')

    # set axes limits
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)





    # draw lines between caption and image points
    # for i in range(len(image_coordinates_same)):
    #     x = [image_coordinates_same[i, 0], caption_coordinates_same[i, 0]]
    #     y = [image_coordinates_same[i, 1], caption_coordinates_same[i, 1]]
    #     z = [image_coordinates_same[i, 2], caption_coordinates_same[i, 2]]
    #     ax.plot(x, y, z, c='grey', alpha=0.5)

    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(image_coordinates_different[:, 0], image_coordinates_different[:, 1], image_coordinates_different[:, 2])
    ax.scatter(caption_coordinates_different[:, 0], caption_coordinates_different[:, 1], caption_coordinates_different[:, 2])
    ax.scatter(image_centroid_different[0], image_centroid_different[1], image_centroid_different[2], c='r')
    ax.scatter(caption_centroid_different[0], caption_centroid_different[1], caption_centroid_different[2], c='r')

    # draw lines between caption and image points
    # for i in range(len(image_coordinates_different)):
    #     x = [image_coordinates_different[i, 0], caption_coordinates_different[i, 0]]
    #     y = [image_coordinates_different[i, 1], caption_coordinates_different[i, 1]]
    #     z = [image_coordinates_different[i, 2], caption_coordinates_different[i, 2]]
    #     ax.plot(x, y, z, c='grey', alpha=0.5)
    ax.set_title('Different')

    # set axes to be equal
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    # print distance between centroids
    print('Distance between centroids same: ', np.linalg.norm(image_centroid_same - caption_centroid_same))
    print('Distance between centroids different: ', np.linalg.norm(image_centroid_different - caption_centroid_different))
    plt.show()




# if main
if __name__ == '__main__':

    settings = [
    {
        'text_only': False,
        'same_encoder': False,
        'same_captions': False,
        'name': 'Default'
    },
    {
        'text_only': True,
        'same_encoder': False,
        'same_captions': False,
        'name': 'DCDE'
    },
    {
        'text_only': True,
        'same_encoder': True,
        'same_captions': False,
        'name': 'DCSE'
    },
    {
        'text_only': True,
        'same_encoder': False,
        'same_captions': True,
        'name': 'SCDE'

    },
    ]

    clip_models = []
    val_dataloaders = []

    for setting in settings:
        config.training_hyperparameters['text_only'] = setting['text_only']
        config.training_hyperparameters['same_encoder'] = setting['same_encoder']
        config.training_hyperparameters['same_captions'] = setting['same_captions']
        clip_model = HFClip()
        clip_models.append(clip_model)

        # get mscoco dataset processor
        dataset_processor = MSCOCOProcessor()

        val_dataset = dataset_processor.val_dataset

        # dataloader
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1024,
                                                collate_fn=dataset_processor.collate_fn,
                                                generator=torch.Generator().manual_seed(42))
        val_dataloaders.append(val_dataloader)
        





    
    
    plot_embeddings(clip_models, val_dataloaders, names=[setting['name'] for setting in settings])

