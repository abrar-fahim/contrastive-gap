import torch

import sys
import os
import wandb
import pickle
import io

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# add sibling directory to path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))# def 


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

import matplotlib.pyplot as plt

embeddings_path = 'embeddings/T0.01_uniform_align_2_finetune_I1C2E1E2_3_train_as_val_512_mscoco_VIT16_pretrained.pt'

# embeddings = torch.load(embeddings_path, map_location='cpu')
with open(embeddings_path, 'rb') as f:
    embeddings = CPU_Unpickler(f).load()
n_plots = 10

#subplots of 3d scatter plots
fig = plt.figure(figsize=(10, 10))

# set title
fig.suptitle('Train dataset, default CLIP')

for i, step_data in enumerate(embeddings):
    print('step ', step_data['step'])
    print('epoch ', step_data['epoch'])
    print('index ', step_data['index'])

    step = step_data['step']
    epoch = step_data['epoch']

    image_embeds = step_data['encoder1_final_embeds']
    text_embeds = step_data['encoder2_final_embeds']

    # 3D scatter plot
    ax = fig.add_subplot(3, 3, i+1, projection='3d')
    ax.set_title(f'epoch {epoch}')
    ax.scatter(image_embeds[:, 0], image_embeds[:, 1], image_embeds[:, 2], label='image')
    ax.scatter(text_embeds[:, 0], text_embeds[:, 1], text_embeds[:, 2], label='text')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()

    


    if i >= n_plots-2:
        break


    



plt.show()


    



