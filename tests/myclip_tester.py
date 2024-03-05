from clips.my_clip import MyClip, MyClipLoss
import torch
from clips.hf_clip import HFClip
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as dset
from PIL import Image
import requests
import clip

from src.utils import collate_fn
from clips.openai_clip import OpenAIClip


# set seed
torch.manual_seed(42)




device = torch.device(training_hyperparameters['cuda_device'] if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/32", device=device)
validation_dataset = dset.CocoCaptions(root = './datasets/mscoco/val2014',
    annFile = 'datasets/mscoco/annotations/captions_val2014.json',
    # transform=[transforms.PILToTensor()])
    transform=preprocess,
    )

openai_clip_model = OpenAIClip().to(device)
# myclip_model = MyClip().to(device)

hf_clip_model = HFClip().to(device)


'''
Testing on random images and captions
'''

# image1 = Image.open(requests.get('https://m.media-amazon.com/images/M/MV5BMTM3OTUwMDYwNl5BMl5BanBnXkFtZTcwNTUyNzc3Nw@@._V1_FMjpg_UX1000_.jpg', stream=True).raw)
# image1 = Image.open(requests.get('https://i.natgeofe.com/n/548467d8-c5f1-4551-9f58-6817a8d2c45e/NationalGeographic_2572187_square.jpg', stream=True).raw)
image1 = Image.open(requests.get('https://www.purina.co.uk/sites/default/files/2020-12/How%20to%20Stop%20Cats%20Scratching%20FurnitureTEASER.jpg', stream=True).raw)
# image1 = Image.open(requests.get('https://mms.businesswire.com/media/20150312005220/en/457282/5/2015Mustang_t5e9701_%281%29.jpg', stream=True).raw)
# image1 = Image.open(requests.get('https://img.freepik.com/premium-photo/light-bulb-with-plant-inside-it-white-surface-with-green-leaves-surrounding-it-generative-ai_97167-4973.jpg?w=2000', stream=True).raw)
# image1 = Image.open(requests.get('https://d163axztg8am2h.cloudfront.net/static/img/6e/dc/8f4d8aa1bbaee091316efcc557a7.webp', stream=True).raw)
image1 = Image.open(requests.get('https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/800px-Cat_November_2010-1a.jpg', stream=True).raw)



# captions = ['orange cat sitting on black couch.', 'black cat sitting on orange couch.']
# captions = ['black cat sitting on brown couch.', 'brown cat sitting on black couch.']
captions = ['a lightbulb surrounding some plants', 'some plants surrounding a lightbulb']

captions = ['a white cat sitting on a red couch.', 'a red cat sitting on a white couch.']

captions = ['photo of a brown cat.', 'photo of a black cat.']
# captions = ['picture of a red car traveling on a black road.', 'picture of a black car traveling on a red road.']


preprocessed_image = preprocess(image1)

preprocessed_image = preprocessed_image.unsqueeze(0).to(device)

myclip_outputs = hf_clip_model(preprocessed_image, captions, output_loss=False) # so that I get cosine similarities directly

logits_per_image, logits_per_text = myclip_outputs # shape of both: ([batch_size, batch_size])

print('label probs for MYCLIP', logits_per_image.softmax(dim=-1))

print('logits_per_image for MYCLIP', logits_per_image)



openai_clip_outputs = openai_clip_model(preprocessed_image, captions)

logits_per_image, logits_per_text = openai_clip_outputs # shape of both: ([batch_size, batch_size])

print('label probs for OPENAI CLIP', logits_per_image.softmax(dim=-1))

print('logits_per_image for OPENAI CLIP', logits_per_image)




exit()

'''
Testing on self made validation set
'''

# validation hypers

validation_hyperparameters = {
    'small_train_loader_batch_size': 64,
    'small_train_loader_dataset_size': 100
}





# just for validation
subset_indices = torch.randint(0, len(validation_dataset) , (validation_hyperparameters['small_train_loader_dataset_size'],))

# get 100 indices that are not in train_data_subset
val_indices = torch.randint(0, len(validation_dataset) , (100,))
i = 0
while i < 100:
    while val_indices[i] in subset_indices:
        val_indices[i] = torch.randint(0, len(validation_dataset) , (1,))
    i += 1
print('i ', i)



# get 100 images from train_dataset that are not in train_data_subset
val_data_subset = Subset(validation_dataset, val_indices)

val_dataloader = DataLoader(val_data_subset, batch_size=validation_hyperparameters['small_train_loader_batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=0)

# test model with validation set
clip_model.eval()


with torch.no_grad():
    

    for (img, caption) in val_dataloader:
        outputs = clip_model(img, caption, scale=False) # so tha I get cosine similarities directly
        logits_per_image, logits_per_text = outputs # shape of both: ([batch_size, batch_size])

        # print('logits_per_image ', logits_per_image)

        # print logits per image for first 5 images
        # print('logits_per_image ', logits_per_image[:5, :5])
        cosine_similarities = logits_per_image.diag() # shape: [64]
        # get median cosine similarity
        median_cosine_similarity = torch.median(cosine_similarities)
        print('median cosine similarity ', median_cosine_similarity)

        # get median of elements that are not on the diagonal
        non_similar_median_cosine_similarity = logits_per_image[~torch.eye(logits_per_image.shape[0], dtype=bool)].median()
        print('non_similar_median_cosine_similarity ', non_similar_median_cosine_similarity)






