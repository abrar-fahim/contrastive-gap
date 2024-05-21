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

# pca on the embeddings
from sklearn.decomposition import PCA
import numpy as np

plt.rcParams.update({'font.size': 18})


# embeddings_path = 'embeddings/T0.01_uniform_align_2_finetune_I1C2E1E2_3_train_as_val_512_mscoco_VIT16_pretrained.pt'

def plot_cumulative_explained_vars(image_embeddings, text_embeddings):

    # embeddings shape: (n, 3)

    # calculate PCA

    image_pca = PCA(n_components=3)

    text_pca = PCA(n_components=3)

    _ = text_pca.fit_transform(text_embeddings)

    pca_embeddings = image_pca.fit_transform(image_embeddings)

    image_explained_vars = image_pca.explained_variance_ratio_  
    text_explained_vars = text_pca.explained_variance_ratio_

    average_explained_vars = (image_explained_vars + text_explained_vars) / 2  





        
    width=0.45
  
    variances_cumsum = np.cumsum(average_explained_vars)
  
  
    # for i, explained_var in enumerate(explained_vars):
          
          
  
    plt.plot(torch.arange(start=1, end=4), variances_cumsum)
        # plt.plot(np.arange(start=1, stop=64, step=8), np.cumsum(all_average_variances[i]), label='{}'.format(''), color=f'tab:blue')
  
      
        
    # plt.xticks(np.arange(8) + width / 2, ('0-7', '8-15', '16-23', '24-31', '32-39', '40-47', '48-55', '56-63'))
  
    # plt.legend(fontsize="22")
  
    plt.xlabel('Dimensions')

    # only show integers in x axis
    plt.xticks(np.arange(1, 4), np.arange(1, 4))

    # y axis numbers to 2 decimal places
    plt.gca().yaxis.set_major_formatter('{:.2f}'.format)
  
    # plt.ylabel('PCA Explained variance')
  
    plt.show()





embeddings_path = 'embeddings/T0.01_Lit_2_scratch_I1C2E1E2_3_train_as_val_1024_mscoco_VIT_pretrained.pt'

# embeddings = torch.load(embeddings_path, map_location='cpu')
with open(embeddings_path, 'rb') as f:
    embeddings = CPU_Unpickler(f).load()
n_plots = 10

#subplots of 3d scatter plots
fig = plt.figure(figsize=(6, 6))

# set title
# fig.suptitle('Train dataset, default CLIP')


print('len of embeddings ', len(embeddings))


# pseudo_epochs_to_visualize = [0, 4, 10]

start=26
pseudo_epochs_to_visualize = np.arange(2, 25)

pseudo_epochs_to_visualize = [24]
# pseudo_epochs_to_visualize = [24] # 275
pca = PCA(n_components=3)

# collect embeddings 

cat_image_embeddings = torch.cat([embeddings[i]['encoder1_final_embeds'] for i in pseudo_epochs_to_visualize], dim=0)

cat_text_embeds = torch.cat([embeddings[i]['encoder2_final_embeds'] for i in pseudo_epochs_to_visualize], dim=0)

print('cat shape ', cat_image_embeddings.shape) # n_total, 512



# # normalize
# image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
# text_embeds /= text_embeds.norm(dim=-1, keepdim=True)

# 3D scatter plot
# ax = fig.add_subplot(3, 3, n_plots-1, projection='3d')
# ax.set_title(f'PCA')
# pca_cat_image_embeds: np.ndarray = pca.fit_transform(cat_image_embeddings)
# pca_cat_text_embeds = pca.fit_transform(cat_text_embeds)

# # normalize
# pca_cat_image_embeds /= np.linalg.norm(pca_cat_image_embeds, axis=-1, keepdims=True)
# pca_cat_text_embeds /= np.linalg.norm(pca_cat_text_embeds, axis=-1, keepdims=True)

for i, pseudo_epoch in enumerate(pseudo_epochs_to_visualize):

    step_data = embeddings[pseudo_epoch]
    print('step ', step_data['step'])
    print('epoch ', step_data['epoch'])
    print('index ', step_data['index'])

    step = step_data['step']
    epoch = step_data['epoch']

    start = i*cat_image_embeddings.shape[0]//len(pseudo_epochs_to_visualize)

    end = (i+1)*cat_image_embeddings.shape[0]//len(pseudo_epochs_to_visualize)

    print('start ', start)
    print('end ', end)

    # fig.suptitle(f'Epoch {epoch}: Cumulative PCA Explained Variance')
    fig.suptitle(f'Epoch {epoch}')

    


    # image_embeds = pca_cat_image_embeds[start:end]

    # text_embeds = pca_cat_text_embeds[start:end]

    image_embeds = step_data['encoder1_final_embeds']
    text_embeds = step_data['encoder2_final_embeds']


    # pca = PCA(n_components=3)
    # pca_image_embeds: np.ndarray = pca.fit_transform(image_embeds)
    # pca_text_embeds = pca.fit_transform(text_embeds)

    pca_image_embeds = image_embeds
    pca_text_embeds = text_embeds

    plot_cumulative_explained_vars(pca_image_embeds, pca_text_embeds)



    # 3D scatter plot
    ax = fig.add_subplot(1, 1, i+1, projection='3d')
    ax.set_title(f'epoch {epoch}')
    ax.scatter(pca_image_embeds[:, 0], pca_image_embeds[:, 1], pca_image_embeds[:, 2], label='image', color='tab:red')
    ax.scatter(pca_text_embeds[:, 0], pca_text_embeds[:, 1], pca_text_embeds[:, 2], label='text', color='tab:blue')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # fix axes
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)


    # show sphere in 3D plot
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='b', alpha=0.1)

    # ax.legend()

    if i >= n_plots-2:
        break



# # for i, step_data in enumerate(embeddings):
# for i, pseudo_epoch in enumerate(pseudo_epochs_to_visualize):
#     # i = epoch
#     step_data = embeddings[pseudo_epoch]
#     print('step ', step_data['step'])
#     print('epoch ', step_data['epoch'])
#     print('index ', step_data['index'])

#     step = step_data['step']
#     epoch = step_data['epoch']

#     # image_embeds = step_data['encoder1_final_embeds']
#     # text_embeds = step_data['encoder2_final_embeds']

#     image_embeds = cat_image_embeddings

#     # normalize
#     image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
#     text_embeds /= text_embeds.norm(dim=-1, keepdim=True)

#     # 3D scatter plot
#     ax = fig.add_subplot(3, 3, i+1, projection='3d')
#     ax.set_title(f'epoch {epoch}')
#     ax.scatter(image_embeds[:, 0], image_embeds[:, 1], image_embeds[:, 2], label='image')
#     ax.scatter(text_embeds[:, 0], text_embeds[:, 1], text_embeds[:, 2], label='text')
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     ax.legend()

    


#     if i >= n_plots-2:
#         break


    



plt.show()


    




