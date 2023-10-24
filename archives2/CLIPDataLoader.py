
from PIL import Image
import requests
from archives2.CLIPWrapper import CLIPWrapper
import json
import random
import matplotlib.pyplot as plt

import numpy as np


class CLIPDataLoader:

    def __init__(self, n, dataset='mscoco'):
        '''
        Randomly sample n captions from the given dataset (MSCOCO by default), and get the corresponding images
        '''


        f = open('datasets/mscoco/annotations/captions_val2014.json')

        data = json.load(f)
        random.seed(0)


        # SAMPLE CAPTIONS

        sampled_captions = []

        sampled_caption_ids = []


        for i in random.sample(range(len(data['annotations'])), n):

            # only append if caption id is not already in sampled_caption_ids

            if data['annotations'][i]['image_id'] not in sampled_caption_ids:

                sampled_captions.append(data['annotations'][i]['caption'])

                sampled_caption_ids.append(data['annotations'][i]['image_id'])


        # SAMPLE IMAGES

        sampled_images = []

        for image_id in sampled_caption_ids:
            for image in data['images']:
                if image['id'] == image_id:
                    # print progress 
                    if len(sampled_images) % 10 == 0:
                        print(len(sampled_images), ' / ', len(sampled_caption_ids))
                    image = Image.open(requests.get(image['coco_url'], stream=True).raw)
                    sampled_images.append(image)
                    break # break out of inner loop since we found the image we were looking for


        # in sampled_images, ith image corresponds to ith caption in sampled_captions

        print('NUM IMAGES SAMPLED ', len(sampled_images))
        print('NUM CAPTIONS SAMPLED ', len(sampled_captions))

        self.sampled_captions = sampled_captions

        self.sampled_images = sampled_images
        

