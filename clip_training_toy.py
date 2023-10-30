'''
- Setup image encoder
- Setup text encoder
- use nonlinear projection layer
- Setup toy dataset
- Setup loss fn
- Setup training loop

- Train model on toy dataset using minibatches
- Test to see if model can get high cosine similarities between images and captions of same concept
- This works as sanity check to test algorithm in general

'''

'''
- Setup toy dataset, MSCOCO for now
'''

from my_clip import MyClip, MyClipLoss
from grad_cache_wrapper import GradCacheWrapper
from openai_clip import OpenAIClip
from hf_clip import HFClip
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch
from training_utils import do_validation
import clip
import os
import torchvision.datasets as dset
import matplotlib.pyplot as plt
from PIL import Image
import requests

# set seed
torch.manual_seed(42)

training_hyperparameters = {
    'batch_size': 16,
    'grad_cache': False,
    'grad_cache_multiplier': 32,
    'n_epochs': 1,
    'lr': 1e-5,
    'weight_decay': 0.2,
    'model_path': 'checkpoints/my_clip_checkpoint.pt',
    'validation_dataset_size': 256,
    'validation_batch_size': 256,
    'do_checkpointing': False,
    'start_new': False,
    'use_small_trainloader': True,
    'small_train_loader_batch_size': 256,
    'small_train_loader_dataset_size': 10000,
    }



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print('device ', device)

model, preprocess = clip.load("ViT-B/32", device=device)
# model, preprocess = clip.load("ViT-B/16", device=device)
train_dataset = dset.CocoCaptions(root = './datasets/mscoco/val2014',
                        annFile = 'datasets/mscoco/annotations/captions_val2014.json',
                        # transform=[transforms.PILToTensor()])
                        transform=preprocess,
)


print('Number of samples: ', len(train_dataset))


# clip_model = MyClip().to(device)
# clip_model = OpenAIClip().to(device)

clip_model = HFClip().to(device)



# print parameters that are trainable
for name, param in clip_model.named_parameters():
    if param.requires_grad:
        print(name)


'''
- Setup training loop
'''

# setup adamW optimizer

optimizer = optim.AdamW(clip_model.parameters(), lr=training_hyperparameters['lr'], weight_decay=training_hyperparameters['weight_decay'])

# setup loss function
clip_loss = MyClipLoss()

def collate_fn(batch):
    '''
    batch is a list of tuples?
    each tuple is of the form (image, caption)
    image is a tensor of shape [3, 224, 224]
    caption is a tuple of strings
    '''

    imgs, og_captions = zip(*batch)

    # keep only first caption for each image
    captions = [caption[0] for caption in og_captions]

    # caption2 = [caption[0] for caption in og_captions]
    # return (caption2, captions)
    return (torch.stack(imgs), captions)


n_epochs = training_hyperparameters['n_epochs']

clip_model.train()


'''
checkpointing stuff
'''


i_loaded_from_checkpoint = False

subset_indices = torch.randint(0, len(train_dataset) , (training_hyperparameters['small_train_loader_dataset_size'],)) # always defined and exists, but only used when small training loader is used, and we're not loading from checkpoint at start

if os.path.exists(training_hyperparameters['model_path']) and not training_hyperparameters['start_new'] and training_hyperparameters['do_checkpointing']:

    # load checkpoint
    checkpoint = torch.load(training_hyperparameters['model_path'], map_location=device)
    clip_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    losses = checkpoint['losses']
    train_dataloader = checkpoint['train_dataloader']
    i = checkpoint['dataloader_enumerator_index']
    median_cosine_similarities = checkpoint['median_cosine_similarities']
    i_loaded_from_checkpoint = True

else:
    epoch = 0
    i = 0
    losses = []
    median_cosine_similarities = []

    if training_hyperparameters['use_small_trainloader']:

        '''
        Prepare subset of training dataset
        '''


        train_data_subset = Subset(train_dataset, subset_indices)

        train_dataloader = DataLoader(train_data_subset, batch_size=training_hyperparameters['small_train_loader_batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=0)

        '''
        display all subset indices images as small tiles
        '''

        # plt.figure()

        # #subplot(r,c) provide the no. of rows and columns
        # f, axarr = plt.subplots(10,10) 
        # # hide axis labels and numbers
        # for ax in axarr:
        #     for axi in ax:
        #         axi.axis('off')

        # # reduce space between subplots
        # f.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

        # for i in range(10):
        #     for j in range(10):
        #         img, target = train_dataset[subset_indices[i * 10 + j]]
        #         axarr[i][j].imshow(img.permute(1, 2, 0))

        # plt.show()
    else:

        train_dataloader = DataLoader(train_dataset, batch_size=training_hyperparameters['batch_size'], shuffle=True, collate_fn=collate_fn)

dataloader = train_dataloader



'''
Build validation dataset
- This only works when using small train loader
'''


# get 100 indices that are not in train_data_subset
val_indices = torch.randint(0, len(train_dataset) , (training_hyperparameters['validation_dataset_size'],))
j = 0
while j < training_hyperparameters['validation_dataset_size']:
    while val_indices[j] in subset_indices:
        val_indices[j] = torch.randint(0, len(train_dataset) , (1,))
    j += 1
print('j ', j)

val_data_subset = Subset(train_dataset, val_indices)

val_dataloader = DataLoader(val_data_subset, batch_size=training_hyperparameters['validation_batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=0)




# training loop
while epoch < n_epochs:


    running_loss = 0.0

    if not i_loaded_from_checkpoint:
        i = 0

    if training_hyperparameters['grad_cache']:
        clip_model_grad_cache = GradCacheWrapper(clip_model)

        clip_model_grad_cache.clip_model.train()

        cache_x = []
        cache_y = []
        closures_x = []
        closures_y = []

        for step, sub_batch in enumerate(dataloader):  
            imgs, captions = sub_batch
            r_imgs, c_imgs = clip_model_grad_cache.get_image_projections(imgs)

            r_txts, c_txts = clip_model_grad_cache.get_text_projections(captions)

            # print progress in place
            # print('\rstep: ' + str(step), end='')

            # print progress every 5 steps
            if step % 5 == 0:
                print('step: ', step)



            
            cache_x.append(r_imgs)
            cache_y.append(r_txts)
            closures_x.append(c_imgs)
            closures_y.append(c_txts)

            # print size of cache x
            # print('len(cache_x) ', len(cache_x))
            
            if (step + 1) % training_hyperparameters['grad_cache_multiplier'] == 0:

                loss = clip_model_grad_cache.contrastive_loss(cache_x, cache_y)
                # print loss
                print('loss ', loss)
                
                loss.backward()
            
                # TEST THESE FOR LOOPS LATER 
                for f, r in zip(closures_x, cache_x):
                    f(r)
                for f, r in zip(closures_y, cache_y):
                    f(r)

                cache_x = []
                cache_y = []
                closures_x = []
                closures_y = []
            
                optimizer.step()
                # scaler.update()
                optimizer.zero_grad()

                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, loss.item()))

                do_validation(val_dataloader, clip_model_grad_cache.clip_model)

                if i % 100 == 0 and training_hyperparameters['do_checkpointing']:
                    checkpoint_to_save = {
                        'epoch': epoch,
                        'model_state_dict': clip_model_grad_cache.clip_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'losses': losses,
                        'train_dataloader': dataloader,
                        'dataloader_enumerator_index': i,
                        'median_cosine_similarities': median_cosine_similarities
                        }
                    torch.save(checkpoint_to_save, training_hyperparameters['model_path'])
                i += 1

    else:

        for (imgs, captions) in dataloader:

            # print('img ', img)
            # print('caption ', caption)

            # evaluate model
            clip_model.eval()
            
            

            do_validation(val_dataloader, clip_model, index=i, captioning_model=True)
                

            clip_model.train()  

            # zero the parameter gradients
            optimizer.zero_grad()

            # caption WAS a list of tuples, where first tuple corresponds to first captions of all the images in the batch

            # caption is now a list of 64 strings 

            

            # forward + backward + optimize
            _, _, loss = clip_model(imgs, captions, output_loss=True)
            # loss = clip_loss(*outputs)
            loss.backward()

            optimizer.step()

            # print statistics
            running_loss += loss.item()

            losses.append(loss.item())
            # if i % 2 == 1:    # print every 2 mini-batches
            print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 1))
            running_loss = 0.0


            # save model 
            if i % 100 == 0 and training_hyperparameters['do_checkpointing']:
                checkpoint_to_save = {
                    'epoch': epoch,
                    'model_state_dict': clip_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'losses': losses,
                    'train_dataloader': dataloader,
                    'dataloader_enumerator_index': i,
                    'median_cosine_similarities': median_cosine_similarities
                    }
                torch.save(checkpoint_to_save, training_hyperparameters['model_path'])
            i += 1
    
    i_loaded_from_checkpoint = False
    epoch +=1

# # plot losses and similarities
# import matplotlib.pyplot as plt
# plt.plot(losses)
# plt.plot(median_cosine_similarities)
# plt.title('losses')
# plt.show()




        # median_cosine_similarities.append(median_cosine_similarity.item())

exit()

import torch

from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTImageProcessor
from PIL import Image
import requests

from transformers import AutoImageProcessor, ViTModel
import torch
from datasets import load_dataset

# dataset = load_dataset("huggingface/cats-image")
# image = dataset["test"]["image"][0]

'''
- Setup Image Encoder
'''



url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
feature_extractor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224')
image_encoder = ViTModel.from_pretrained('google/vit-large-patch16-224')
# classification_model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')
inputs = feature_extractor(images=image, return_tensors="pt")

print('inputs ', inputs['pixel_values'].shape)
# outputs = model(**inputs)

outputs = image_encoder(**inputs)


pooler_output = outputs.pooler_output # ([1, 1024])

print('pooler_output ', pooler_output.shape)


'''
- Setup Text Encoder
'''

from transformers import AutoTokenizer, GPT2Model, GPT2Config
import torch

configuration = GPT2Config(summary_type="cls_index")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")

inputs = tokenizer("Hello my dog is cute.", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state

print('last_hidden_states ', last_hidden_states.shape)

eos_representation = last_hidden_states[:, -1, :] # ([1, 768])
print('eos_representation ', eos_representation.shape)

'''
- Setup nonlinear projection layer
'''

import torch.nn as nn





# print('outputs ', outputs)
# logits = outputs.logits
# # model predicts one of the 1000 ImageNet classes
# predicted_class_idx = logits.argmax(-1).item()
# print("Predicted class:", model.config.id2label[predicted_class_idx])


