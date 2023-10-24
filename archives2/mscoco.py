'''
Testing the hypothesis that the distances between the text embeddings for one concept is the same as the distance between the text and image embeddings for that concept

MSCOCO has 5 captions per image
'''


from PIL import Image
import requests
from archives2.CLIPWrapper import CLIPWrapper
import json
import random

from matplotlib import pyplot as plt

similarity_measure = 'cosine_similarity'

# similarity_measure = 'euclidean_distance'

# similarity_measure = 'euclidean_similarity'

pca_dims = 5

f = open('datasets/mscoco/annotations/captions_val2014.json')

data = json.load(f)

# data keys: (['info', 'licenses', 'images', 'annotations'])

# data['images'][0] keys: (['license', 'file_name', 'coco_url', 'height', 'width', 'date_captured', 'flickr_url', 'id'])

# data['annotation'] keys: (['image_id', 'id', 'caption'])

# find corresponding captions for the first image, there are multiple captions per image
captions = []

for annotation in data['annotations']:
    if annotation['image_id'] == data['images'][0]['id']:
        captions.append(annotation['caption'])
        

# load image from url
image = Image.open(requests.get(data['images'][0]['coco_url'], stream=True).raw)

image1 = Image.open(requests.get('https://github.com/rmokady/CLIP_prefix_caption/raw/main/Images/COCO_val2014_000000165547.jpg', stream=True).raw)

image2 = Image.open(requests.get('https://images.fineartamerica.com/images/artworkimages/mediumlarge/2/zebra-in-living-room-smelling-rug-side-matthias-clamer.jpg', stream=True).raw)




captions = ['A dining table and chairs in a room']




print('CAPTION ', captions)


images = [image1, image2] # since processor expects a list of images

'''
1. For one concept, The text embeddings corresponding to that concept are the same distance to each other on average compared to the distance between the text and image embeddings.
'''
print()
print(' --- 1 ---')
print()

clip_1 = CLIPWrapper(captions, images, similarity_measure)

print('logits_per_image ', clip_1.get_logits_per_image_and_probs()[0])

print('logits_per_text ', clip_1.outputs.logits_per_text)

print('label probs ', clip_1.get_logits_per_image_and_probs()[1])



print(f'average text-text {similarity_measure}', clip_1.get_average_text_text_similarity())

print(f'average text-image {similarity_measure}', clip_1.get_average_text_image_similarity())

plt.imshow(image)

plt.show()

exit()

'''
2. The average distance between captions referring to different concepts is further than that between captions referring to the same concept
'''

print()
print(' --- 2 ---')
print()


# captions of same concept is captions variable

# get captions of different concepts, sample randomly

pca_dims = 5

captions_different = []

count = 0

for annotation in data['annotations']:
    if annotation['image_id'] != data['images'][0]['id']:
        captions_different.append(annotation['caption'])
        count += 1
    if count == 5: # since there are 5 captions per image, to make fair comparison
        break

print('captions different ', captions_different)

clip_2 = CLIPWrapper(captions_different, images, similarity_measure, pca_dims=pca_dims) # here, images is just a placeholder, and not used

print(f'average {similarity_measure} for captions of same concept', clip_1.get_average_text_text_similarity())

print(f'average {similarity_measure} for captions of different concepts', clip_2.get_average_text_text_similarity())



'''
3. The average distance between captions referring to different concepts is further than that between captions and images referring to the same concept
'''

print()
print(' --- 3 ---')
print()



print(f'average {similarity_measure} between captions and image of same concept', clip_1.get_average_text_image_similarity())

print(f'average {similarity_measure} for captions of different concepts', clip_2.get_average_text_text_similarity())


'''
4. The average distance between images referring to different concepts is further than that between captions and images referring to the same concept
'''

print()
print(' --- 4 ---')
print()

different_images = []

count = 0


# remove 5th image, since it is very similar to the fourth image

modified_data_images = data['images'][:4] + data['images'][5:]

for current_image_data in modified_data_images:
    image = Image.open(requests.get(current_image_data['coco_url'], stream=True).raw)
    different_images.append(image)
    count += 1
    if count == 5: # since there are 5 captions per image, to make fair comparison
        break


# show different images

# for image in different_images:
#     plt.imshow(image)
#     plt.show()


# captions in clip_3 is just placeholder

clip_3 = CLIPWrapper(captions, different_images, similarity_measure)


print(f'average {similarity_measure} between captions and image of same concept', clip_1.get_average_text_image_similarity())


print(f'average {similarity_measure} between two images of different concepts', clip_3.get_average_image_image_similarity())

'''
5. The average distance between an image and its corresponding captions is smaller than that between images and captions referring to different concepts.
'''

print()
print(' --- 5 ---')
print()


clip_4_image_data = data['images'][5]

clip_4_image = Image.open(requests.get(clip_4_image_data['coco_url'], stream=True).raw)

# show clip_4 image
# plt.imshow(clip_4_image)
# plt.show()

clip_4_image = [clip_4_image]

clip_4 = CLIPWrapper(captions_different, clip_4_image, similarity_measure)

print(f'average {similarity_measure} between captions and image of same concept', clip_1.get_average_text_image_similarity())

print(f'average {similarity_measure} between captions and image of different concepts', clip_4.get_average_text_image_similarity())


'''
6. Testing difference between average distance between all texts and that between all images.
'''

print()
print(' --- 6 ---')
print()

# set random seed

random.seed(0)
n = 100

sampled_captions = []

sampled_caption_ids = []

print('GETTING ALL CAPTIONS')

# for annotation in data['annotations']:
#     sampled_captions.append(annotation['caption'])

# randomly sample 100 captions from data['annotations']

for i in random.sample(range(len(data['annotations'])), n):

    # only append if caption id is not already in sampled_caption_ids

    if data['annotations'][i]['image_id'] not in sampled_caption_ids:

        sampled_captions.append(data['annotations'][i]['caption'])

        sampled_caption_ids.append(data['annotations'][i]['image_id'])

print('NUM CAPTIONS SAMPLED ', len(sampled_captions))

print('sampled captions ', sampled_captions[:5])



sampled_images = []

print('GETTING ALL IMAGES')

# there are 40k something images 

# randomly sample 100 images from data['images']

for i in random.sample(range(len(data['images'])), n):
    # print progress
    if len(sampled_images) % 10 == 0:
        print(len(sampled_images), ' / ', n)
    current_image_data = data['images'][i]
    image = Image.open(requests.get(current_image_data['coco_url'], stream=True).raw)
    sampled_images.append(image)




# for current_image_data in data['images']:

#     # print progress
#     if len(all_images) % 10 == 0:
#         print(len(all_images), ' / ', len(data['images']))
#     image = Image.open(requests.get(current_image_data['coco_url'], stream=True).raw)
#     all_images.append(image)



clip_5 = CLIPWrapper(sampled_captions, sampled_images, similarity_measure)

print('GETTING AVERAGE TEXTS SIMILARITY')

print(f'average {similarity_measure} between all captions', clip_5.get_average_text_text_similarity())

print('GETTING AVERAGE IMAGES SIMILARITY')

print(f'average {similarity_measure} between all images', clip_5.get_average_image_image_similarity())

print('GETTING AVERAGE TEXT-IMAGE SIMILARITY')

print(f'average {similarity_measure} between all texts and images', clip_5.get_average_text_image_similarity())