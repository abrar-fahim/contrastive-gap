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



def plots():
    # load npy files
    image_coordinates_same = np.load('image_coordinates_same.npy')
    caption_coordinates_same = np.load('caption_coordinates_same.npy')

    image_centroid_same = np.load('image_centroid_same.npy')
    caption_centroid_same = np.load('caption_centroid_same.npy')

    image_coordinates_different = np.load('image_coordinates_different.npy')
    caption_coordinates_different = np.load('caption_coordinates_different.npy')

    image_centroid_different = np.load('image_centroid_different.npy')
    caption_centroid_different = np.load('caption_centroid_different.npy')

    # translate image coordinates further away from caption coordinates
    # find vector from image centroid to caption centroid
    vector_same = caption_centroid_same - image_centroid_same
    # translate image coordinates further away from caption coordinates
    # image_coordinates_same += vector_same * 2
    # caption_coordinates_same -= vector_same * 2

    # find vector from image centroid to caption centroid
    vector_different = caption_centroid_different - image_centroid_different
    # # translate image coordinates further away from caption coordinates
    # image_coordinates_different += vector_different * 2
    # caption_coordinates_different -= vector_different * 2


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

    plots()
    exit()
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

    

