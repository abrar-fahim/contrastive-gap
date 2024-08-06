'''
Making do_validation function simpler by adding an evaluator class
'''

import sys
import os
import wandb
import torch.nn.functional as F
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle


from scipy import stats
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA


import sys
import os

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# add sibling directory to path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))# def 

from src.utils import generate_csv_file_name, get_embeddings_path
from src.config import *
from src.my_ce_loss import MyCrossEntropyLoss, MyCEAlignmentLoss
from clips.hf_clip import HFClipOutput, HFClip
from dataset_processors.mscoco_processor import MSCOCOProcessor  # change this to dataset_processor_parent later, after you write the abstract functions there.

from dataset_processors.dataset_processor_parent import DatasetProcessorParent
from dataset_processors.cifar10_processor import CIFAR10Processor
from dataset_processors.food101_processor import Food101Processor
from dataset_processors.cifar100_processor import CIFAR100Processor
from dataset_processors.sun397_processor import SUN397Processor
from dataset_processors.dtd_processor import DTDProcessor
from dataset_processors.caltech101_processor import Caltech101Processor
from dataset_processors.fgvc_aircraft_processor import FGVCAircraftProcessor
from dataset_processors.stanford_cars_processor import StanfordCarsProcessor
from dataset_processors.conceptual_captions_processor import ConceptualCaptionsProcessor



class Evaluator():
    def __init__(self, dataset_processor: DatasetProcessorParent, val_batch_cache_file=None, load_train_dataset=True):
        self.dataset_processor = dataset_processor

        if val_batch_cache_file is not None:
            self.mscoco_batch_file_path = val_batch_cache_file
        else:
            self.mscoco_batch_file_path = f"datasets/{wandb.config['dataset']}/val_batch_cache_{generate_csv_file_name()}.pt"


        # self.mscoco_train_dataset_batch_file_path = self.mscoco_batch_file_path
        self.mscoco_train_dataset_batch_file_path = f"datasets/{wandb.config['dataset']}/train_batch_cache_{generate_csv_file_name()}.pt"

        self.mscoco_val_dataloader: torch.utils.data.DataLoader = None
        self.mscoco_train_dataloader: torch.utils.data.DataLoader = None

        self.val_outputs: HFClipOutput = None

        self.train_outputs: HFClipOutput = None

        self.outputs_to_use: HFClipOutput = None # outputs used inside evaluate_model depending on whether Im using training or val data

        self.pca: PCA = PCA(n_components=2) # for visualizing embeddings
        self.pca_fitted = False


        '''
        Setup linear probe datasets
        '''
        # print()
        # print('setting up linear probe datasets')
        # self.linear_probe_datasets: list[DatasetProcessorParent] = [CIFAR10Processor(), Food101Processor()]
        # self.linear_probe_datasets: list[DatasetProcessorParent] = [ Caltech101Processor(), CIFAR10Processor(), CIFAR100Processor(), DTDProcessor()]


        '''
        Setting up zero shot acc datasets
        '''

        if wandb.config['encoder2_modality'] == 'image' or wandb.config['encoder1_modality'] == 'text':
            # since 1 should be image and 2 should be text, this means that both modalities are same, so zero shot performance is NOT possible
            self.zero_shot_datasets: list[DatasetProcessorParent] = [None]
        else:

            # self.zero_shot_datasets: list[DatasetProcessorParent] = [CIFAR10Processor()]    
            self.zero_shot_datasets: list[DatasetProcessorParent] = [CIFAR10Processor()]    


        self.encoder1_pooled_hidden_states = []
        self.encoder2_pooled_hidden_states = []
        # self.layers_to_use = [0, 3, 6, 9, 12] # these are the layers to use for computing mean cosine similarity
        self.layers_to_use = [-1] # these are the layers to use for computing mean cosine similarity


        # return    

        '''
        setting dataloaders
        '''


        if os.path.exists(self.mscoco_batch_file_path) and wandb.config['delete_val_batch_first']:
            print('deleting batch from cache')
            os.remove(self.mscoco_batch_file_path)


        if not (os.path.exists(self.mscoco_batch_file_path) and wandb.config['use_cached_val_batch']):

            if wandb.config['use_train_as_val']:

                if type(self.dataset_processor) == MSCOCOProcessor:
                    mscoco_val_dataset = self.dataset_processor.train_dataset

                elif type(self.dataset_processor) == ConceptualCaptionsProcessor:
                    mscoco_val_dataset =  self.dataset_processor.train_data_pipe
                # batch_size = wandb.config['small_train_loader_batch_size'] # MAYBE CHANGE LATER
                batch_size = wandb.config['validation_batch_size']
            else:

                if type(self.dataset_processor) == MSCOCOProcessor:
                    mscoco_val_dataset = self.dataset_processor.val_dataset
                elif type(self.dataset_processor) == ConceptualCaptionsProcessor:
                    mscoco_val_dataset =  self.dataset_processor.val_data_pipe
                batch_size = wandb.config['validation_batch_size']

            collate_fn = self.dataset_processor.collate_fn
            mscoco_val_dataloader = torch.utils.data.DataLoader(mscoco_val_dataset, batch_size=batch_size, collate_fn=collate_fn, generator=torch.Generator().manual_seed(wandb.config['seed']))



            # for now, using val dataset as train dataset.
            # self.train_dataloader = mscoco_val_dataloader


        
        '''
        Loading cached batch from file
        '''



        if os.path.exists(self.mscoco_batch_file_path) and wandb.config['use_cached_val_batch']:
            print('loading VAL batch from cache ', self.mscoco_batch_file_path)

            (mscoco_val_imgs, mscoco_val_captions) = torch.load(self.mscoco_batch_file_path)
            print('loading VAL cache done')

            if os.path.exists(self.mscoco_train_dataset_batch_file_path) and load_train_dataset:
                print('loading TRAIN batch from cache ', self.mscoco_train_dataset_batch_file_path)

                (mscoco_train_imgs, mscoco_train_captions) = torch.load(self.mscoco_train_dataset_batch_file_path)
                print('loading TRAIN cache done')



        else:

            # mscoco_val_imgs = torch.empty(wandb.config['validation_dataset_size'], 3, 224, 224)
            # mscoco_val_captions = [None] * wandb.config['validation_dataset_size']
            print('LOADING VAL BATCH')
            # for i, batch in tqdm(enumerate(mscoco_val_dataloader)):
            for batch in mscoco_val_dataloader:
                print('loading val batch')
                # (val_imgs, val_captions) = next(iter(val_dataloader))

                # val_imgs, val_caps = batch
                (mscoco_val_imgs, mscoco_val_captions) = batch

                # mscoco_val_imgs[i * batch_size: (i + 1) * batch_size] = val_imgs
                # mscoco_val_captions[i * batch_size: (i + 1) * batch_size] = val_caps
                print('val batch loading done')

                break

            if load_train_dataset:


                print('loading TRAIN batch')

                mscoco_train_imgs = torch.empty(wandb.config['validation_dataset_size'], 3, 224, 224)

                mscoco_train_captions = [None] * wandb.config['validation_dataset_size']

                

                train_batch_size = wandb.config['batch_size']

                num_train_batches_to_get = wandb.config['validation_dataset_size'] / train_batch_size

                for i, batch in tqdm(enumerate(dataset_processor.train_dataloader)):

                    if i >= num_train_batches_to_get:
                        break
                    
                    # (val_imgs, val_captions) = next(iter(val_dataloader))

                    (train_imgs, train_caps) = batch


                    # handling case when batch size is NOT factor of val datasetr size
                    length = len(mscoco_train_imgs[i * train_batch_size: (i + 1) * train_batch_size])

                    mscoco_train_imgs[i * train_batch_size: (i + 1) * train_batch_size] = train_imgs[:length]
                    mscoco_train_captions[i * train_batch_size: (i + 1) * train_batch_size] = train_caps[:length]

                    

                print('TRAIN batch loading done')

            
                

            if wandb.config['use_cached_val_batch']:
                print('saving VAL batch to cache ', self.mscoco_batch_file_path)
                # save batch to cache
                torch.save((mscoco_val_imgs, mscoco_val_captions), self.mscoco_batch_file_path)

                if load_train_dataset:

                    print('saving TRAIN batch to cache')

                    torch.save((mscoco_train_imgs, mscoco_train_captions), self.mscoco_train_dataset_batch_file_path)



                del mscoco_val_dataloader

        self.mscoco_val_imgs = mscoco_val_imgs
        self.mscoco_val_captions = mscoco_val_captions

        if load_train_dataset:

            self.mscoco_train_imgs = mscoco_train_imgs
            self.mscoco_train_captions = mscoco_train_captions


    def set_outputs_to_use(self, split='val'):
        if split == 'val':
            self.outputs_to_use = self.val_outputs
        elif split == 'train':
            self.outputs_to_use = self.train_outputs
        else:
            raise ValueError('split should be either val or train')

    def evaluate_model(self, clip_model: HFClip, epoch: int, index: int, is_train_data=False):

        # is_train_data = False for validation dataset, true for training dataset.

        with torch.no_grad():

            self.set_val_outputs(clip_model, is_train_data=is_train_data)

            if is_train_data:
                self.outputs_to_use = self.train_outputs
            else:
                self.outputs_to_use = self.val_outputs
            # self.set_pooled_hidden_states(clip_model)

            # save pooled hidden states to file
            if wandb.config['save_encoder_hidden_states']:

                save_path = get_embeddings_path()
                self.save_pooled_hidden_states_to_file(save_path, epoch, index)

            '''
            log to wandb
            '''



            average_intra_modality_cosine_sim = (self.get_text_text_similarity() + self.get_image_image_similarity() ) / 2

            mods_same = wandb.config['encoder1_modality'] == wandb.config['encoder2_modality']

            val_loss = self.get_val_loss()

            rsa_correlations = self.get_rsa_correlations(clip_model.get_temperature())

            # rsa_correlations_between_diff_layers = self.get_rsa_correlations_between_diff_layers()
            # intra_modality_similarities_within_diff_layers = self.get_intra_modality_similarities_within_diff_layers()

            linear_probe_accuracies = {
                'cifar10': -1,
                'cifar100': -1,
                'dtd': -1,
                'caltech101': -1,

            }

            # for dataset_processor in self.linear_probe_datasets:
            #     linear_probe_accuracies[dataset_processor.keyname] = self.get_dataset_linear_probe_accuracy(clip_model, dataset_processor)

            average_linear_probe_accuracy = np.mean(list(linear_probe_accuracies.values()))
            std_dev_linear_probe_accuracy = np.std(list(linear_probe_accuracies.values()))

            ranks = self.get_rank()



            n_s_buckets = 8

            # group S values into buckets, and get mean of each bucket
            # S shape: (clip_projection_dim, )

            image_S_buckets = torch.chunk(ranks['image_S'], n_s_buckets) 
            text_S_buckets = torch.chunk(ranks['text_S'], n_s_buckets)

            image_pca_variance_ratio_buckets = torch.chunk(ranks['image_explained_variance_ratios'], n_s_buckets)
            text_pca_variance_ratio_buckets = torch.chunk(ranks['text_explained_variance_ratios'], n_s_buckets)

            if len(image_S_buckets) < n_s_buckets:
                image_S_buckets += tuple([torch.tensor([0.0]) for _ in range(n_s_buckets - len(image_S_buckets))])

                image_pca_variance_ratio_buckets += tuple([torch.tensor([0.0]) for _ in range(n_s_buckets - len(image_pca_variance_ratio_buckets))])
            
            if len(text_S_buckets) < n_s_buckets:
                text_S_buckets += tuple([torch.tensor([0.0]) for _ in range(n_s_buckets - len(text_S_buckets))])

                text_pca_variance_ratio_buckets += tuple([torch.tensor([0.0]) for _ in range(n_s_buckets - len(text_pca_variance_ratio_buckets))])

            image_S_bucket_means = [torch.mean(bucket) for bucket in image_S_buckets]
            text_S_bucket_means = [torch.mean(bucket) for bucket in text_S_buckets]

            image_pca_variance_ratio_bucket_sums = [torch.sum(bucket) for bucket in image_pca_variance_ratio_buckets]

            text_pca_variance_ratio_bucket_sums = [torch.sum(bucket) for bucket in text_pca_variance_ratio_buckets]

            print('temp ', clip_model.get_temperature())



            if wandb.config['cifar10_acc']:

                zero_shot_dataset_modality_gap_metrics = self.get_dataset_metrics(clip_model, self.zero_shot_datasets[0])
            else:
                zero_shot_dataset_modality_gap_metrics = {
                    'image_uniformity_loss': 100,
                    'mean_cosine_similarity': 100,
                    'centroid_euclidean_distance': 100,
                    'inter_modality_loss': 100,
                    'temp_scaled_inter_modality_loss': 100,
                }

            data: dict = {
                        'image_classification_accuracy': self.get_val_image_classification_acc(),
                        'image_retrieval_accuracy': self.get_val_image_retrieval_acc(),
                        'intramodality_loss': val_loss['intra_modality'],
                        'intermodality_loss': val_loss['inter_modality'],
                        'alignment_loss': val_loss['alignment'],
                        'uniformity_loss': val_loss['uniformity'],
                        'cyclic_loss': val_loss['cyclic'],
                        'cyclic_dir_loss': val_loss['cyclic_direction'],
                        'uniform_cyclic_loss': val_loss['uniform_cyclic'],
                        'cross_uniformity_loss': val_loss['cross_uniformity'],
                        'uniform_cyclic_loss': val_loss['uniform_cyclic'],
                        'rsa_loss': val_loss['rsa'],
                        'pearson_loss': val_loss['pearson_rsa'],
                        'svd': val_loss['svd'],
                        'uniformity': val_loss['uniformity'],
                        'total_loss':val_loss['total'],
                        'mean_pairwise_euclidean_distance':  self.get_mean_pairwise_euclidean_distance(),
                        'mean_cosine_similarity': self.get_mean_cosine_similarity(clip_model.get_temperature()),
                        'linear_seperability_accuracy': self.get_linear_seperability(),
                        'centroid_cosine_similarity': self.get_centroid_cosine_similarity(),
                        'centroid_euclidean_distance': self.get_centroid_euclidean_distance(),


                        # temperature
                        'temperature': clip_model.get_temperature(),

                        # rank stuff
                        'image_rank': ranks['image_rank'],
                        'text_rank': ranks['text_rank'],
                        'full_image_rank': ranks['full_image_rank'],
                        'full_text_rank': ranks['full_text_rank'],
                        'first_lt1_value': ranks['first_lt1_value'],
                        'avg_S': ranks['avg_S'],

                        # bucketed S values

                        'image_S0': image_S_bucket_means[0],
                        'image_S1': image_S_bucket_means[1],
                        'image_S2': image_S_bucket_means[2],
                        'image_S3': image_S_bucket_means[3],
                        'image_S4': image_S_bucket_means[4],
                        'image_S5': image_S_bucket_means[5],
                        'image_S6': image_S_bucket_means[6],
                        'image_S7': image_S_bucket_means[7],
                        'text_S0': text_S_bucket_means[0],
                        'text_S1': text_S_bucket_means[1],
                        'text_S2': text_S_bucket_means[2],
                        'text_S3': text_S_bucket_means[3],
                        'text_S4': text_S_bucket_means[4],
                        'text_S5': text_S_bucket_means[5],
                        'text_S6': text_S_bucket_means[6],
                        'text_S7': text_S_bucket_means[7],

                        # bucketed explained variance ratio values

                        'image_variance0': image_pca_variance_ratio_bucket_sums[0],
                        'image_variance1': image_pca_variance_ratio_bucket_sums[1],
                        'image_variance2': image_pca_variance_ratio_bucket_sums[2],
                        'image_variance3': image_pca_variance_ratio_bucket_sums[3],
                        'image_variance4': image_pca_variance_ratio_bucket_sums[4],
                        'image_variance5': image_pca_variance_ratio_bucket_sums[5],
                        'image_variance6': image_pca_variance_ratio_bucket_sums[6],
                        'image_variance7': image_pca_variance_ratio_bucket_sums[7],

                        'text_variance0': text_pca_variance_ratio_bucket_sums[0],
                        'text_variance1': text_pca_variance_ratio_bucket_sums[1],
                        'text_variance2': text_pca_variance_ratio_bucket_sums[2],
                        'text_variance3': text_pca_variance_ratio_bucket_sums[3],
                        'text_variance4': text_pca_variance_ratio_bucket_sums[4],
                        'text_variance5': text_pca_variance_ratio_bucket_sums[5],
                        'text_variance6': text_pca_variance_ratio_bucket_sums[6],
                        'text_variance7': text_pca_variance_ratio_bucket_sums[7],


                        # 'image_S0': ranks['image_S'][0],
                        # 'image_S1': ranks['image_S'][1],
                        # 'image_S2': ranks['image_S'][2],
                        # 'image_S3': ranks['image_S'][3],
                        # 'image_S4': ranks['image_S'][4],
                        # 'image_S5': ranks['image_S'][5],
                        # 'image_S6': ranks['image_S'][6],
                        # 'image_S7': ranks['image_S'][7],
                        # 'text_S0': ranks['text_S'][0],
                        # 'text_S1': ranks['text_S'][1],
                        # 'text_S2': ranks['text_S'][2],
                        # 'text_S3': ranks['text_S'][3],
                        # 'text_S4': ranks['text_S'][4],
                        # 'text_S5': ranks['text_S'][5],
                        # 'text_S6': ranks['text_S'][6],
                        # 'text_S7': ranks['text_S'][7],



                        # linear probe acc
                        'cifar10_linear_probe_accuracy': linear_probe_accuracies['cifar10'],
                        # 'food101_linear_probe_accuracy': linear_probe_accuracies['Food101Processor'],
                        'cifar100_linear_probe_accuracy': linear_probe_accuracies['cifar100'],
                        'dtd_linear_probe_accuracy': linear_probe_accuracies['dtd'],
                        'caltech101_linear_probe_accuracy': linear_probe_accuracies['caltech101'],
                        # 'fgvc_aircraft_linear_probe_accuracy': linear_probe_accuracies['fgvcaircraft'],
                        ''



                        'average_linear_probe_accuracy': average_linear_probe_accuracy,
                        'std_dev_linear_probe_accuracy': std_dev_linear_probe_accuracy,

                        # logging hidden state metrics
                        # 'e1_e2_mean_cosine_similarities': self.get_mean_cosine_similarity_between_diff_layers() if mods_same else None,
                        # 'mean_e1_e2_centroid_euclidean_distances': self.get_mean_pairwise_euclidean_distance_between_diff_layers() if mods_same else None,
                        # 'e1_e2_linear_seperability_accuracies': self.get_linear_seperability_between_diff_layers() if mods_same else None,
                        # 'e1_e2_rsas': rsa_correlations_between_diff_layers['e1_e2_rsas'] if mods_same else None,
                        # 'e1_e2_inter_intra_rsas': rsa_correlations_between_diff_layers['e1_e2_inter_intra_rsas'] if mods_same else None,
                        # 'e1_cosine_similarities': intra_modality_similarities_within_diff_layers['e1_cosine_similarities'],
                        # 'e2_cosine_similarities': intra_modality_similarities_within_diff_layers['e2_cosine_similarities'],

                        # back to other stuff
                        'non_similar_mean_cosine_similarity': self.non_similar_mean_cosine_similarity(clip_model.get_temperature()),
                        'mean_text_text_cosine_similarity': self.get_text_text_similarity(),
                        'mean_image_image_cosine_similarity': self.get_image_image_similarity(),
                        'average_intra_modality_cosine_similarity': average_intra_modality_cosine_sim,

                        # RSA STUFF

                        'rsa_before_interchanging':  rsa_correlations['rsa_before_interchanging'],
                        'text_intermodality_rsa': rsa_correlations['text_intermodality_rsa'],
                        'image_intermodality_rsa': rsa_correlations['image_intermodality_rsa'],
                        'pearson_rsa_before_interchanging':rsa_correlations['pearson_rsa_before_interchanging'],
                        'pearson_text_intermodality_rsa': rsa_correlations['pearson_text_intermodality_rsa'],
                        'pearson_image_intermodality_rsa': rsa_correlations['pearson_image_intermodality_rsa'],


                        'cifar10_val_image_classification_accuracy': self.get_dataset_zero_shot_acc(clip_model, self.zero_shot_datasets[0]) if wandb.config['cifar10_acc'] else 100,

                        'cifar10_image_uniformity_loss': zero_shot_dataset_modality_gap_metrics['image_uniformity_loss'],
                        'cifar10_mean_cosine_similarity': zero_shot_dataset_modality_gap_metrics['mean_cosine_similarity'],
                        'cifar10_centroid_euclidean_distance': zero_shot_dataset_modality_gap_metrics['centroid_euclidean_distance'],
                        'cifar10_inter_modality_loss': zero_shot_dataset_modality_gap_metrics['inter_modality_loss'],
                        'cifar10_temp_scaled_inter_modality_loss': zero_shot_dataset_modality_gap_metrics['temp_scaled_inter_modality_loss'],
                    }



            # changing key names to reflect train or val
            wandb_log_data = data.copy()

            metric_prefix = 'train_' if is_train_data else 'val_'

            for key in data.keys():
                wandb_log_data[f'{metric_prefix}{key}'] = wandb_log_data.pop(key)
            if wandb is not None:
                wandb.log(
                    data=wandb_log_data,
                    # step = int(epoch * (len(dataset_processor.train_dataloader) // wandb.config['batch_size']) + index) # this may not work with WIT dataset, check later
                    # step= int(epoch * 470 + index), # by 100 to maintain fair comparison with existing runs data
                    step = int(epoch * self.dataset_processor.get_num_batches() + index), # by 100 to maintain fair comparison with existing runs data
                    

                )

            if wandb.config['visualize_embeddings']:
                self.plot_embeddings()

    def plot_embeddings(self):
        '''
        plot embeddings
        '''

        encoder1_embeddings = self.val_outputs.image_embeds
        encoder2_embeddings = self.val_outputs.text_embeds

        # normalize embeddings
        encoder1_embeddings = encoder1_embeddings / torch.norm(encoder1_embeddings, dim=1, keepdim=True)
        encoder2_embeddings = encoder2_embeddings / torch.norm(encoder2_embeddings, dim=1, keepdim=True)

        # pca to 2d

        all_embeddings = torch.cat((encoder1_embeddings, encoder2_embeddings), dim=0)

        if not self.pca_fitted:
            self.pca.fit(all_embeddings.cpu().numpy())
            self.pca_fitted = True
        
        all_embeddings_pca = self.pca.transform(all_embeddings.cpu().numpy())



        # plot
        plt.scatter(all_embeddings_pca[:encoder1_embeddings.shape[0], 0], all_embeddings_pca[:encoder1_embeddings.shape[0], 1], label='image', color='red')
        plt.scatter(all_embeddings_pca[encoder1_embeddings.shape[0]:, 0], all_embeddings_pca[encoder1_embeddings.shape[0]:, 1], label='text', color='blue')



        # draw lines between embeddings
        for i in range(encoder1_embeddings.shape[0]):
            plt.plot([all_embeddings_pca[i, 0], all_embeddings_pca[i + encoder1_embeddings.shape[0], 0]], [all_embeddings_pca[i, 1], all_embeddings_pca[i + encoder1_embeddings.shape[0], 1]], color='black', alpha=0.5)

        # fix x and y limits
        plt.xlim(-0.5, 0.5)
        plt.ylim(-0.5, 0.5)

        plt.legend()
        plt.show()

        


        pass

    def get_clip_features_and_labels(self, clip_model: HFClip, dataloader: DataLoader) -> dict:
        '''
        get image features and labels for the linear probe evaluation datasets
        '''

        vision_model = clip_model.get_image_encoder().image_model.vision_model
        # need this vision model to get embeddings BEFORE projection head
        # FIX THIS FOR RN50 MODEL LATER

        all_image_features = torch.empty(0, vision_model.config.hidden_size)

        all_labels = []

        for batch in tqdm(dataloader):
            (imgs, indices) = batch

            imgs = imgs.to(clip_model.device)

            vision_model_outputs = vision_model(pixel_values=imgs)

            image_features = vision_model_outputs[1] # pooled_output, since vision model outputs is a sequence. This is from https://github.dev/huggingface/transformers/blob/v4.39.0/src/transformers/models/clip/modeling_clip.py

            all_image_features = torch.cat((all_image_features, image_features.to(all_image_features.device)), dim=0)

            # append image labels
            all_labels.extend(indices.tolist())

        print('ALL IMAGE FEATURES ', all_image_features.shape)

        return {
            'image_features': all_image_features,
            'labels': all_labels
        }



    def get_dataset_metrics(self, clip_model: HFClip, dataset_processor: DatasetProcessorParent):
        '''
        get modality gap related metrics for a custom dataset
        useful to see if train and test distributions are too different
        '''

        # get image features and labels

        

        with torch.no_grad():

            contrastive_loss = MyCrossEntropyLoss()

            image_uniformity_loss_runsum = 0

            mean_cosine_similarity_runsum = 0
            
            inter_modality_loss_runsum = 0

            temp_scaled_inter_modality_loss_runsum = 0

            cifar_classes = dataset_processor.classes

            # tokenize captions
            # cifar_tokenized_classes = clip_model.tokenize_captions(cifar_classes)


            '''
            Get text embeddings
            '''

            text_embeddings = []

            templates = dataset_processor.templates

            for c in cifar_classes:
                text = [template(c) for template in templates]

                tokenized_text = clip_model.get_text_encoder().tokenize_captions(text)
                text_embedding = clip_model.encode_text(tokenized_text, output_dict=True)['embeds']
                text_embedding /= text_embedding.norm(dim = -1, keepdim = True)
                text_embedding = text_embedding.mean(dim = 0)
                text_embedding /= text_embedding.norm()
                text_embeddings.append(text_embedding)



            text_embeddings = torch.stack(text_embeddings, dim = 1).to(clip_model.device) # shape: ([dim, n_classes] == [512, 10])

            classes_centroid = text_embeddings.mean(dim = 1) # shape: ([512])
            


            '''
            Get image embeddings
            '''
  
            for batch in tqdm(dataset_processor.val_dataloader):

                (cifar_val_imgs, cifar_val_labels) = batch
                cifar_val_labels = cifar_val_labels.to(clip_model.device) # shape: ([64])

                cifar_image_embeddings = clip_model.encode_image(cifar_val_imgs, output_dict=True)['embeds']

                cifar_image_embeddings /= cifar_image_embeddings.norm(dim = -1, keepdim = True)

                cifar_val_logits_per_image = cifar_image_embeddings @ text_embeddings # shape: ([64, 10])

                # get mean cosine similarity
                mean_cosine_similarity = cifar_val_logits_per_image[:, cifar_val_labels].mean()

                mean_cosine_similarity_runsum += mean_cosine_similarity

                # centroid euclidean distance
                # get centroid of images
                image_centroid = cifar_image_embeddings.mean(dim = 0) # shape: ([512])

                centroid_euclidean_distance = torch.norm(image_centroid - classes_centroid)

                # get contrastive loss
                inter_modality_loss = contrastive_loss(cifar_val_logits_per_image, cifar_val_labels)
                inter_modality_loss_runsum += inter_modality_loss

                # get temperature scaled uniformity loss
                temp_scaled_inter_modality_loss = contrastive_loss(cifar_val_logits_per_image * clip_model.logit_scale.exp(), cifar_val_labels)
                temp_scaled_inter_modality_loss_runsum += temp_scaled_inter_modality_loss



                # get uniformity

                image_sq_pdist = torch.pdist(cifar_image_embeddings, p=2).pow(2)
                image_uniformity_loss = image_sq_pdist.mul(-2).exp().mean().log()

                

                image_uniformity_loss_runsum += image_uniformity_loss
            
            image_uniformity_loss = image_uniformity_loss_runsum / len(dataset_processor.val_dataloader)

            mean_cosine_similarity = mean_cosine_similarity_runsum / len(dataset_processor.val_dataloader)

            inter_modality_loss = inter_modality_loss_runsum / len(dataset_processor.val_dataloader)

            temp_scaled_inter_modality_loss = temp_scaled_inter_modality_loss_runsum / len(dataset_processor.val_dataloader)









        print('CIFAR10 image_uniformity_loss ', image_uniformity_loss)
        print('CIFAR10 mean_cosine_similarity ', mean_cosine_similarity)
        print('CIFAR10 centroid_euclidean_distance ', centroid_euclidean_distance)
        print('CIFAR10 inter_modality_loss ', inter_modality_loss)
        print('CIFAR10 temp scaled inter_modality_loss ', temp_scaled_inter_modality_loss)


        return {
            'image_uniformity_loss': image_uniformity_loss,
            'mean_cosine_similarity': mean_cosine_similarity,
            'centroid_euclidean_distance': centroid_euclidean_distance,
            'inter_modality_loss': inter_modality_loss,
            'temp_scaled_inter_modality_loss': temp_scaled_inter_modality_loss

        }

    def get_dataset_linear_probe_accuracy(self, clip_model: HFClip, dataset_processor: DatasetProcessorParent):
        '''
        train a linear classifier on the image embeddings to predict the class labels
        input features are from just before the projection head
        '''


        print()
        print('getting clip features for ', dataset_processor.name)

        '''
        USING VAL AS TRAIN FOR NOW TO SPEED UP LINEAR CLASSIFIER FIT
        '''
        # train_clip_outputs = self.get_clip_features_and_labels(clip_model, dataset_processor.train_dataloader)
        train_clip_outputs = self.get_clip_features_and_labels(clip_model, dataset_processor.val_dataloader)
        all_train_features = train_clip_outputs['image_features']
        all_train_labels = train_clip_outputs['labels']

        # do train val split 
        n_train = int(0.5 * len(all_train_features))
        all_val_features = all_train_features[n_train:]
        all_val_labels = all_train_labels[n_train:]

        all_train_features = all_train_features[:n_train]
        all_train_labels = all_train_labels[:n_train]


        # setup linear classifier
        linear_classifier = LogisticRegression(max_iter=400, verbose=1)
        # max iters to 500 FOR NOW TO SPEED UP VALIDATION AND TRAINING



        print('fitting linear classifier on ', dataset_processor.name)


        # train linear classifier
        linear_classifier.fit(all_train_features, all_train_labels)

        # get test features


        # print()
        # print('getting val features for ', dataset_processor.name)
        # val_clip_outputs = self.get_clip_features_and_labels(clip_model, dataset_processor.val_dataloader)
        # all_val_features = val_clip_outputs['image_features']
        # all_val_labels = val_clip_outputs['labels']


        print('evaluating linear classifier on ', dataset_processor.name)

        # get accuracy
        # accuracy = linear_classifier.score(all_val_features, all_val_labels)
        accuracy = dataset_processor.get_accuracy(linear_classifier, all_val_features, all_val_labels)

        print(f'{dataset_processor.name}_linear_probe_accuracy ', accuracy)

        del all_train_features, all_train_labels, all_val_features, all_val_labels, linear_classifier


        return accuracy


        
    def set_val_outputs(self, clip_model: HFClip, output_loss=True, is_train_data=False):

        # only set output_loss to false when you're calling this outside of evaluator
        # if is_train_data = False, set val outputs. Otherwise, set train outputs.

        
        
        with torch.no_grad():
            clip_model.eval()

            if is_train_data:
                train_outputs: HFClipOutput = clip_model(self.mscoco_train_imgs, self.mscoco_train_captions, output_loss=output_loss, return_all=True, output_hidden_states=False, output_intra_modality_loss=True) 

                self.train_outputs = train_outputs
            else:

                val_outputs: HFClipOutput = clip_model(self.mscoco_val_imgs, self.mscoco_val_captions, output_loss=output_loss, return_all=True, output_hidden_states=False, output_intra_modality_loss=True) # outputting everything all at once and storing them
                self.val_outputs = val_outputs


    def set_pooled_hidden_states(self, clip_model: HFClip):
        # both image and text encoders have 13 outputs in the hidden_states list
        # image encoder each hidden state shape = ([batch_size, 50, 768]) == (batch_size, sequence_length, hidden_size)
        # text encoder each hidden state shape = ([batch_size, 26, 512]) == (batch_size, sequence_length, hidden_size)

        # get mean cosine similarity between encoder outputs at different layers
        encoder1_hidden_states = self.val_outputs.encoder1_hidden_states
        encoder2_hidden_states = self.val_outputs.encoder2_hidden_states

        for layer in self.layers_to_use:

            image_embeds = self.val_outputs.image_embeds
            text_embeds = self.val_outputs.text_embeds

            if layer < 0:
                e1_pool = image_embeds
                e2_pool = text_embeds

            else:


                # pool hidden states to convert from sequence to single value

                e1_pool = clip_model.encoder1.pool_hidden_state(encoder1_hidden_states[layer], self.val_outputs.encoder1_input_ids)

                e2_pool = clip_model.encoder2.pool_hidden_state(encoder2_hidden_states[layer], self.val_outputs.encoder2_input_ids)


            # pooled_hidden_states shape: ([batch_size, hidden_size])

            # following assertions work because in HFClipOutput, image_embeds = encoder1 and text_embeds = encoder2

            

            assert e1_pool.shape == (image_embeds.shape[0], clip_model.encoder1.hidden_size), f'e1_pool.shape = {e1_pool.shape}, expected shape = ({image_embeds.shape[0]}, {clip_model.encoder1.hidden_size})'

            assert e2_pool.shape == (text_embeds.shape[0], clip_model.encoder2.hidden_size), f"e2_pool.shape = {e2_pool.shape}, expected shape = ({text_embeds.shape[0]}, {clip_model.encoder2.hidden_size})"



            # normalize features
            e1_pool = e1_pool / torch.norm(e1_pool, dim=1, keepdim=True)
            e2_pool = e2_pool / torch.norm(e2_pool, dim=1, keepdim=True)

            self.encoder1_pooled_hidden_states.append(e1_pool)
            self.encoder2_pooled_hidden_states.append(e2_pool)

    def get_mscoco_uniformity(self, image_embeds=None, text_embeds=None):
        '''
        get uniformity of mscoco dataset
        '''

        with torch.no_grad():

            if image_embeds == None:
                image_embeds = self.val_outputs.image_embeds
                text_embeds = self.val_outputs.text_embeds
            else:
                image_embeds = image_embeds
                text_embeds = text_embeds


            # device = self.val_outputs.image_embeds.device

            device = image_embeds.device

            image_sq_pdist = torch.pdist(image_embeds, p=2).pow(2)
            image_uniformity_loss = image_sq_pdist.mul(-2).exp().mean().log()

            text_sq_pdist = torch.pdist(text_embeds, p=2).pow(2)
            text_uniformity_loss = text_sq_pdist.mul(-2).exp().mean().log()

            total_uniformity_loss = (image_uniformity_loss + text_uniformity_loss) / 2

            off_diagonal_ones = torch.ones((len(image_embeds), len(text_embeds))).to(device).tril(diagonal = -1) 

            off_diagonal_ones += torch.ones((len(image_embeds), len(text_embeds))).to(device).triu(diagonal = 1)
            
            off_diagonal_ones = off_diagonal_ones.to(device)

            cross_encoder_uniform_loss  = torch.masked_select(torch.cdist(image_embeds.unsqueeze(0), text_embeds.unsqueeze(0))[0], off_diagonal_ones == 1).square().mul(-2).exp().mean().log()


            return {
                'image_uniformity_loss': image_uniformity_loss,
                'text_uniformity_loss': text_uniformity_loss,
                'total_uniformity_loss': total_uniformity_loss,
                'cross_encoder_uniform_loss': cross_encoder_uniform_loss
            }
        
    def get_mscoco_alignment(self, image_embeds=None, text_embeds=None):
        '''
        get alignment of mscoco dataset
        '''

        if image_embeds == None:
            image_embeds = self.val_outputs.image_embeds
            text_embeds = self.val_outputs.text_embeds

        else:
            image_embeds = image_embeds
            text_embeds = text_embeds

        with torch.no_grad():


            align = (image_embeds - text_embeds).norm(dim=1).pow(2).mean()

            return align

    def save_pooled_hidden_states_to_file(self, save_path:str, epoch:int, index:int):

        print('saving encoder hidden states to ', save_path)

        n_to_save = wandb.config['n_embeds_to_save']

        # take first n_to_save embeds from each layer
        encoder1_pooled_hidden_states_to_save = [e1_pool[:n_to_save] for e1_pool in self.encoder1_pooled_hidden_states]

        # e_pool shape = [batch_size, CLIPConfig.hidden_size]

        encoder2_pooled_hidden_states_to_save = [e2_pool[:n_to_save] for e2_pool in self.encoder2_pooled_hidden_states]

        image_embeds = self.outputs_to_use.image_embeds
        text_embeds = self.outputs_to_use.text_embeds

        # normalize features
        normalized_encoder1_embeds = image_embeds / torch.norm(image_embeds, dim=1, keepdim=True)
        normalized_encoder2_embeds = text_embeds / torch.norm(text_embeds, dim=1, keepdim=True)

        step = int(epoch * self.dataset_processor.get_num_batches() + index)

        print()
        print(f'--- Saving encoder hidden states for step {step} ---')


        step_data = {
                'step': step,
                'epoch': epoch,
                'index': index,
                'encoder1_pooled_hidden_states': encoder1_pooled_hidden_states_to_save, # normalized. 
                'encoder2_pooled_hidden_states': encoder2_pooled_hidden_states_to_save, # normalized
                'encoder1_final_embeds': normalized_encoder1_embeds[:n_to_save],
                'encoder2_final_embeds': normalized_encoder2_embeds[:n_to_save],
            }

        # append step_data to array saved in save_path

        if os.path.exists(save_path):
            # load existing data
            with open(save_path, 'rb') as f:
                data = pickle.load(f)
                data.append(step_data)

            # save data
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
        else:
            # save data
            with open(save_path, 'wb') as f:
                pickle.dump([step_data], f)




    def get_rank(self, linalg=True) ->  dict:
        '''
        Get rank (threshold = 1) using SVD, of image and text embeds
        Rank of only first config['validation_batch_size'] embeds
        '''

        # data needs to be in (n, d) format for PCA




        # image_embeds = self.val_outputs.image_embeds[:wandb.config['validation_batch_size']]
        # text_embeds = self.val_outputs.text_embeds[:wandb.config['validation_batch_size']]
        image_embeds = self.outputs_to_use.image_embeds
        text_embeds = self.outputs_to_use.text_embeds
        # image_embeds = self.val_outputs.image_embeds[:wandb.config['batch_size']].T
        # text_embeds = self.val_outputs.text_embeds[:wandb.config['batch_size']].T

        # embeds shape: ([batch_size, dimensionality])

        # normalize
        image_embeds = image_embeds / torch.norm(image_embeds, dim=1, keepdim=True)
        text_embeds = text_embeds / torch.norm(text_embeds, dim=1, keepdim=True)

        if linalg:
            # get pytorch rank
            full_image_rank = torch.linalg.matrix_rank(image_embeds)
            full_text_rank = torch.linalg.matrix_rank(text_embeds)

            # get rank
            U, S, Vh = torch.linalg.svd(image_embeds)

            image_S = S

            

            


            # rank is number of singular values greater than 1
            image_rank = torch.count_nonzero(S > 1)

            U, S, Vh = torch.linalg.svd(text_embeds)

            text_S = S

            text_rank = torch.count_nonzero(S > 1)

            # get first element that is less than 1
            lt1_values = S[S <= 1]

            if len(lt1_values) > 0:
                first_lt1_value = lt1_values[0] # works because S is sorted in descending order
            else:
                first_lt1_value = -1



            # get average S value
            avg_S = torch.mean(S)

            print('image_rank ', image_rank)
            print('text_rank ', text_rank)  

        image_pca = PCA(n_components=min(image_embeds.shape[0], image_embeds.shape[1]))
        image_pca.fit(image_embeds.cpu().numpy())
        image_explained_variance_ratios = image_pca.explained_variance_ratio_

        text_pca = PCA(n_components=min(text_embeds.shape[0], text_embeds.shape[1]))
        text_pca.fit(text_embeds.cpu().numpy())
        text_explained_variance_ratios = text_pca.explained_variance_ratio_



        return {
            'image_rank': image_rank,
            'text_rank': text_rank,
            'full_image_rank': full_image_rank,
            'full_text_rank': full_text_rank,
            'first_lt1_value': first_lt1_value,
            'avg_S': avg_S,
            'image_S': image_S,
            'text_S': text_S,
            'image_explained_variance_ratios': torch.tensor(image_explained_variance_ratios),
            'text_explained_variance_ratios': torch.tensor(text_explained_variance_ratios),

        }
        


    def linear_probe_cifar10(self, clip_model:HFClip):
        '''
        train a linear classifier on the image embeddings to predict the class labels
        input features are from just before the projection head
        '''

        vision_model = clip_model.get_image_encoder().image_model.vision_model
        # need this vision model to get embeddings BEFORE projection head

        # setup linear classifier
        linear_classifier = LogisticRegression(max_iter=1000)

        cifar10_image_features = torch.empty(0, vision_model.config.hidden_size) # shape: ([n, 768])
        image_labels = []



        for batch in tqdm(self.cifar_val_dataloader):
            (cifar_val_imgs, cifar_val_indices) = batch

            cifar_val_imgs = cifar_val_imgs.to(clip_model.device)

            vision_model_outputs = vision_model(pixel_values=cifar_val_imgs)

            image_features = vision_model_outputs[1] # pooled_output, since vision model outputs is a sequence. This is from https://github.dev/huggingface/transformers/blob/v4.39.0/src/transformers/models/clip/modeling_clip.py

            cifar10_image_features = torch.cat((cifar10_image_features, image_features.to(cifar10_image_features.device)), dim=0)

            # append image labels
            image_labels.extend(cifar_val_indices.tolist())


        n_train = int(0.8 * len(cifar10_image_features))
        train_features = cifar10_image_features[:n_train]
        train_labels = image_labels[:n_train]

        test_features = cifar10_image_features[n_train:]
        test_labels = image_labels[n_train:]

        # train linear classifier
        linear_classifier.fit(train_features, train_labels)

        # get accuracy
        accuracy = linear_classifier.score(test_features, test_labels)

        print('cifar10_linear_probe_accuracy ', accuracy)

        del cifar10_image_features, image_labels, linear_classifier

        return accuracy



    def get_val_image_classification_acc(self, return_all=False):

        val_logits_per_image = self.outputs_to_use.logits_per_image # shape of both: ([64, 64])

        # image embeddings
        image_embeds = self.outputs_to_use.image_embeds # normalized_encoder1 embeds. Shape: ([batch_size, 512])
        text_embeds = self.outputs_to_use.text_embeds # normalized_encoder2 embeds

        # normalize
        normalized_encoder1_embeds = image_embeds / torch.norm(image_embeds, dim=1, keepdim=True)
        normalized_encoder2_embeds = text_embeds / torch.norm(text_embeds, dim=1, keepdim=True)


        # softmax on logits_per_image
        val_image_class_probs = F.softmax(val_logits_per_image, dim=-1) # shape: ([64, 64])

        topk = [1, 3, 5, 10]
        correct = {k: 0 for k in topk}

        ranks = val_logits_per_image.topk(max(topk), 1)[1].T

        for k in topk:
            correct[k] += torch.sum(torch.any(ranks[:k] == torch.arange(ranks.shape[1], device=ranks.device).unsqueeze(0), dim = 0)).item()



       
        # calculate accuracy
        # get indices of max values
        val_image_class_preds = val_image_class_probs.argmax(dim=-1) # shape: ([64])


        val_image_class_labels = torch.arange(val_image_class_probs.shape[0], device=val_image_class_probs.device) # shape: ([64])


        # calculate accuracy
        val_image_classification_accuracy = (val_image_class_preds == val_image_class_labels).float().mean()

        print('val image class acc ', {k: round(correct[k] / val_image_class_labels.shape[0], 3) for k in topk})

        print('val_image_classification_accuracy ', val_image_classification_accuracy.item())

        if return_all:
            return {k: correct[k] / val_image_class_labels.shape[0] for k in topk}
        else:
            return val_image_classification_accuracy.item()

    def get_dataset_zero_shot_acc(self, clip_model: HFClip, dataset_processor: DatasetProcessorParent, return_all=False) -> float:

        if dataset_processor == None:
            return  0


        with torch.no_grad():

            
            cifar_classes = dataset_processor.classes

            # tokenize captions
            # cifar_tokenized_classes = clip_model.tokenize_captions(cifar_classes)

            text_embeddings = []

            templates = dataset_processor.templates

            for c in cifar_classes:
                text = [template(c) for template in templates]

                tokenized_text = clip_model.get_text_encoder().tokenize_captions(text)


                text_embedding = clip_model.encode_text(tokenized_text, output_dict=True)['embeds']
                text_embedding /= text_embedding.norm(dim = -1, keepdim = True)
                text_embedding = text_embedding.mean(dim = 0)
                text_embedding /= text_embedding.norm()
                text_embeddings.append(text_embedding)
            text_embeddings = torch.stack(text_embeddings, dim = 1).to(clip_model.device) # shape: ([dim, num_classes] == (512, 10))





            cifar10_val_image_classification_accuracy_runsum = 0

            topk = [1, 3, 5, 10]
            correct = {k: 0 for k in topk}
            for batch in tqdm(dataset_processor.val_dataloader):

                
                (cifar_val_imgs, cifar_val_indices) = batch
                
                cifar_image_embeddings = clip_model.encode_image(cifar_val_imgs, output_dict=True)['embeds']

                cifar_image_embeddings /= cifar_image_embeddings.norm(dim = -1, keepdim = True)


                # get logits per image
                # cifar_val_outputs = clip_model(cifar_val_imgs, cifar_tokenized_classes, output_loss=False, return_all=True)
                # cifar_val_outputs = clip_model(cifar_val_imgs, cifar_classes, output_loss=False, return_all=True)

                cifar_val_logits_per_image = cifar_image_embeddings @ text_embeddings # (64, 512) * (512, 10) = (64, 10)

                # cifar_val_logits_per_image = cifar_val_outputs.logits_per_image # shape of both: ([64, 10])

                ranks = cifar_val_logits_per_image.topk(max(topk), 1)[1].T

                cifar_val_indices = cifar_val_indices.to(clip_model.device)
                predictions = ranks == cifar_val_indices

                for k in topk:
                    correct[k] += torch.sum(torch.any(predictions[:k], dim = 0)).item() 

                # softmax on logits_per_image
                # cifar_val_image_class_probs = F.softmax(cifar_val_logits_per_image, dim=-1) # shape: ([batch_size, 10]). 10 is num_classes in cifar10

                # calculate accuracy
                # get indices of max values
                # cifar_val_image_class_preds = cifar_val_image_class_probs.argmax(dim=-1) # shape: ([batch_size])

                # cifar_val_indices = cifar_val_indices.to(clip_model.device)

                # cifar10_val_image_classification_accuracy_runsum += (cifar_val_image_class_preds == cifar_val_indices).float().sum()
                # print('cifar10_val_image_classification_accuracy_runsum ', cifar10_val_image_classification_accuracy_runsum)

            # cifar10_val_image_classification_accuracy = cifar10_val_image_classification_accuracy_runsum / len(dataset_processor.val_dataset)
            # print(f'{dataset_processor.name} zero shot accuracy ', cifar10_val_image_classification_accuracy.item())

            print(f'{dataset_processor.name} zero shot accuracy ', {k: correct[k] / len(dataset_processor.val_dataset) for k in topk})

        
            # return cifar10_val_image_classification_accuracy
            # return top1 acc as string

            if return_all:
                return {k: correct[k] / len(dataset_processor.val_dataset) for k in topk}
            else:

                return correct[1] / len(dataset_processor.val_dataset)


    def get_cifar10_zero_shot_acc(self, clip_model: HFClip):

        with torch.no_grad():

            if self.val_dataset_processor != None:
                cifar_classes = self.val_dataset_processor.classes

                # tokenize captions
                # cifar_tokenized_classes = clip_model.tokenize_captions(cifar_classes)



                cifar10_val_image_classification_accuracy_runsum = 0
                for batch in tqdm(self.cifar_val_dataloader):
                    (cifar_val_imgs, cifar_val_indices) = batch
                    
                    
                    # get logits per image
                    # cifar_val_outputs = clip_model(cifar_val_imgs, cifar_tokenized_classes, output_loss=False, return_all=True)
                    cifar_val_outputs = clip_model(cifar_val_imgs, cifar_classes, output_loss=False, return_all=True)

                    cifar_val_logits_per_image = cifar_val_outputs.logits_per_image # shape of both: ([64, 64])

                    # softmax on logits_per_image
                    cifar_val_image_class_probs = F.softmax(cifar_val_logits_per_image, dim=-1) # shape: ([batch_size, 10]). 10 is num_classes in cifar10

                    # calculate accuracy
                    # get indices of max values
                    cifar_val_image_class_preds = cifar_val_image_class_probs.argmax(dim=-1) # shape: ([batch_size])

                    cifar_val_indices = cifar_val_indices.to(clip_model.device)

                    cifar10_val_image_classification_accuracy_runsum += (cifar_val_image_class_preds == cifar_val_indices).float().sum()
                    # print('cifar10_val_image_classification_accuracy_runsum ', cifar10_val_image_classification_accuracy_runsum)

                cifar10_val_image_classification_accuracy = cifar10_val_image_classification_accuracy_runsum / len(self.cifar_val_dataset)
                print('cifar10_val_image_classification_accuracy ', cifar10_val_image_classification_accuracy.item())

         
                return cifar10_val_image_classification_accuracy
        
            


    def get_val_image_retrieval_acc(self, return_all=False):
        logits_per_text = self.outputs_to_use.logits_per_text # shape of both: ([64, 64])

        # softmax on logits_per_text
        text_class_probs = F.softmax(logits_per_text, dim=-1)
        
        # calculate accuracy
        # get indices of max values: These are indices of the retrieved images
        text_class_preds = text_class_probs.argmax(dim=-1)

        # get indices of correct predictions
        val_text_class_labels = torch.arange(text_class_probs.shape[0], device=text_class_probs.device) # shape: ([64])

        # calculate accuracy
        val_image_retrieval_accuracy = (text_class_preds == val_text_class_labels).float().mean()

        topk = [1, 3, 5, 10]
        correct = {k: 0 for k in topk}

        ranks = logits_per_text.topk(max(topk), 1)[1].T

        for k in topk:
            correct[k] += torch.sum(torch.any(ranks[:k] == torch.arange(ranks.shape[1], device=ranks.device).unsqueeze(0), dim = 0)).item()

        if return_all:
            return {k: correct[k] / val_text_class_labels.shape[0] for k in topk}

        else:
            return val_image_retrieval_accuracy.item()

        # print('retrieval done')
        # return val_image_retrieval_accuracy.item()
    


    def get_val_loss(self):

        loss = self.outputs_to_use.loss

        return loss

    def get_mean_cosine_similarity(self, temperature: int):

        cosine_similarities = self.outputs_to_use.logits_per_image.diag() # shape: [64]
        # get mean cosine similarity
        mean_cosine_similarity = torch.mean(cosine_similarities)

        # scale with temperature
        mean_cosine_similarity = mean_cosine_similarity * temperature

        print('mean cosine similarity ', mean_cosine_similarity.item())

        return mean_cosine_similarity.item()

    

    def non_similar_mean_cosine_similarity(self, temperature: int) -> float:

        # get mean of elements that are not on the diagonal
        non_similar_mean_cosine_similarity = self.outputs_to_use.logits_per_image[~torch.eye(self.outputs_to_use.logits_per_image.shape[0], dtype=bool)].mean()

        # scale with temperature

        non_similar_mean_cosine_similarity = non_similar_mean_cosine_similarity * temperature


        print('non_similar_mean_cosine_similarity ', non_similar_mean_cosine_similarity * temperature)

        return non_similar_mean_cosine_similarity.item()




    def get_mean_cosine_similarity_between_diff_layers(self):

        if wandb.config['encoder1_modality'] == wandb.config['encoder2_modality']:
            # can only measure modality gap if the two modalities are same



            e1_e2_mean_cosine_similarities = [] # tracks mean cosine similarity between embeddings of each layer in layers_to_use
            for e1_pool, e2_pool in zip(self.encoder1_pooled_hidden_states, self.encoder2_pooled_hidden_states):

                # cosine similarities between e1_pool and e2_pool
                e1_e2_cosine_similarities = e1_pool @ e2_pool.t()

                # ensure all cosine similarities are between -1 and 1
                # assert torch.all(e1_e2_cosine_similarities <= 1) and torch.all(e1_e2_cosine_similarities >= -1), f'e1_e2_cosine_similarities = {e1_e2_cosine_similarities}'

                # get mean of elements that are on the diagonal
                e1_e2_mean_cosine_similarity = e1_e2_cosine_similarities.diag().mean()

                # assert e1_e2_mean_cosine_similarity <= 1 and e1_e2_mean_cosine_similarity >= -1, f'e1_e2_mean_cosine_similarity = {e1_e2_mean_cosine_similarity}'

                e1_e2_mean_cosine_similarities.append(e1_e2_mean_cosine_similarity)

            return e1_e2_mean_cosine_similarities


    def get_text_text_similarity(self):

        text_embeds = self.outputs_to_use.text_embeds

        text_encoder_outputs = text_embeds # shape: ([batch_size, 512])
    

        # normalize features
        text_encoder_outputs = text_encoder_outputs / torch.norm(text_encoder_outputs, dim=1, keepdim=True)

        # cosine similarities between text-text pairs
        text_text_cosine_similarities = text_encoder_outputs @ text_encoder_outputs.t() # shape: ([batch_size, batch_size])

        # get median of elements that are in the upper triangle (excluding diagonal!!)
        mean_text_text_cosine_similarity = text_text_cosine_similarities[torch.triu(torch.ones(text_text_cosine_similarities.shape[0], text_text_cosine_similarities.shape[1]), diagonal=1).bool()].mean()

        # print('median_text_text_cosine_similarity ', median_text_text_cosine_similarity)

        return mean_text_text_cosine_similarity.item()
    
    def get_image_image_similarity(self):

        image_embeds = self.outputs_to_use.image_embeds

        image_encoder_outputs = image_embeds

        image_encoder_outputs = image_embeds # shape: ([batch_size, 512])


        # normalize features
        image_encoder_outputs = image_encoder_outputs / torch.norm(image_encoder_outputs, dim=1, keepdim=True)

        # cosine similarities between image-image pairs
        image_image_cosine_similarities = image_encoder_outputs @ image_encoder_outputs.t()

        # get median of elements that are not on the diagonal
        mean_image_image_cosine_similarity = image_image_cosine_similarities[~torch.eye(image_image_cosine_similarities.shape[0], dtype=bool)].mean()



        print('mean_image_image_cosine_similarity ', mean_image_image_cosine_similarity)

        return mean_image_image_cosine_similarity.item()






    def get_mean_pairwise_euclidean_distance(self):
        '''
        Euclidean distance between image and text pairs
        '''

        image_embeds = self.outputs_to_use.image_embeds
        text_embeds = self.outputs_to_use.text_embeds

        image_encoder_outputs = image_embeds # shape: ([batch_size, 512])
        text_encoder_outputs = text_embeds

        # normalize features
        image_encoder_outputs = image_encoder_outputs / torch.norm(image_encoder_outputs, dim=1, keepdim=True)
        text_encoder_outputs = text_encoder_outputs / torch.norm(text_encoder_outputs, dim=1, keepdim=True)


        # euclidean distance between image and text pairs
        euclidean_distances = torch.norm(image_encoder_outputs - text_encoder_outputs, dim=-1)

        assert torch.all(euclidean_distances >= 0) and torch.all(euclidean_distances <= 2), f'euclidean_distances = {euclidean_distances}'

        # get mean euclidean distance
        mean_euclidean_distance = euclidean_distances.mean()

        assert mean_euclidean_distance >= 0 and mean_euclidean_distance <= 2, f'mean_euclidean_distance = {mean_euclidean_distance}'



        print('mean_pairwise_euclidean_distance ', mean_euclidean_distance)

        return mean_euclidean_distance.item()



    def get_mean_pairwise_euclidean_distance_between_diff_layers(self):

        '''
        - Euclidean distance between centroids of hidden_states from each layer
        '''

        if wandb.config['encoder1_modality'] == wandb.config['encoder2_modality']:

            e1_e2_centroid_euclidean_distances = [] # tracks mean euclidean distance between centroids of each layer in layers_to_use


            for e1_pool, e2_pool in zip(self.encoder1_pooled_hidden_states, self.encoder2_pooled_hidden_states):

                # euclidean distance between e1_pool and e2_pool
                e1_e2_euclidean_distances = torch.norm(e1_pool - e2_pool, dim=-1)

                # ensure that all euclidean distances are between 0 and 2
                assert torch.all(e1_e2_euclidean_distances >= 0) and torch.all(e1_e2_euclidean_distances <= 2), f'e1_e2_euclidean_distances = {e1_e2_euclidean_distances}'

                # get mean euclidean distance
                e1_e2_mean_euclidean_distance = e1_e2_euclidean_distances.mean()

                assert e1_e2_mean_euclidean_distance >= 0 and e1_e2_mean_euclidean_distance <= 2, f'e1_e2_mean_euclidean_distance = {e1_e2_mean_euclidean_distance}'
                # <= 2 since the maximum euclidean distance between two normalized vectors is 2

                e1_e2_centroid_euclidean_distances.append(e1_e2_mean_euclidean_distance)

            return e1_e2_centroid_euclidean_distances
        


    def get_centroid_cosine_similarity(self):


        '''
        - Cosine similarity between image and text centroids
        '''

        image_embeds = self.outputs_to_use.image_embeds
        text_embeds = self.outputs_to_use.text_embeds

        image_encoder_outputs = image_embeds # shape: ([batch_size, 512])
        text_encoder_outputs = text_embeds

        # normalize features
        image_encoder_outputs = image_encoder_outputs / torch.norm(image_encoder_outputs, dim=1, keepdim=True)
        text_encoder_outputs = text_encoder_outputs / torch.norm(text_encoder_outputs, dim=1, keepdim=True)


        # get centroids
        text_centroid = text_encoder_outputs.mean(dim=0)
        image_centroid = image_encoder_outputs.mean(dim=0)   


        # normalize centroids
        text_centroid = text_centroid / torch.norm(text_centroid, dim=0, keepdim=True)
        image_centroid = image_centroid / torch.norm(image_centroid, dim=0, keepdim=True)

        # cosine similarity between centroids
        centroid_cosine_similarity = text_centroid @ image_centroid.t()

        print('centroid_cosine_similarity ', centroid_cosine_similarity)

        return centroid_cosine_similarity.item()




    def get_centroid_euclidean_distance(self, image_embeds=None, text_embeds=None):
            
            '''
            - Euclidean distance between image and text centroids
            '''

            if image_embeds == None:
                image_embeds = self.outputs_to_use.image_embeds
                text_embeds = self.outputs_to_use.text_embeds
            else:
                image_embeds = image_embeds
                text_embeds = text_embeds
    
            image_encoder_outputs = image_embeds # shape: ([batch_size, 512])   
            text_encoder_outputs = text_embeds

            # normalize features
            image_encoder_outputs = image_encoder_outputs / torch.norm(image_encoder_outputs, dim=1, keepdim=True)
            text_encoder_outputs = text_encoder_outputs / torch.norm(text_encoder_outputs, dim=1, keepdim=True)


            # get centroids
            text_centroid = text_encoder_outputs.mean(dim=0)
            image_centroid = image_encoder_outputs.mean(dim=0)

            # euclidean distance between centroids
            centroid_euclidean_distance = torch.norm(text_centroid - image_centroid)

            print('centroid_euclidean_distance ', centroid_euclidean_distance)

            return centroid_euclidean_distance.item()



    def get_linear_seperability(self, image_embeds=None, text_embeds=None):
        # Split validation dataset into train and test splits
        # train on 20% of the data and test on 80%

        if image_embeds == None:

            image_embeds = self.outputs_to_use.image_embeds
            text_embeds = self.outputs_to_use.text_embeds

        else:
            image_embeds = image_embeds
            text_embeds = text_embeds            


        normalized_image_embeds = image_embeds / torch.norm(image_embeds, dim=1, keepdim=True)
        normalized_text_embeds = text_embeds / torch.norm(text_embeds, dim=1, keepdim=True)

        # check if normalization happened properly as expected
        # assert torch.allclose(torch.norm(normalized_image_embeds, dim=1), torch.ones(normalized_image_embeds.shape[0], device=normalized_image_embeds.device))
        # assert torch.allclose(torch.norm(normalized_text_embeds, dim=1), torch.ones(normalized_text_embeds.shape[0], device=normalized_text_embeds.device))

        n_train = int(0.8 * len(image_embeds))
        n_test = len(image_embeds) - n_train

        # get random indices

        indices = torch.randperm(len(image_embeds))

        train_indices = indices[:n_train]
 
        test_indices = indices[n_train:]

        train_image_embeds = normalized_image_embeds[train_indices]
        test_image_embeds = normalized_image_embeds[test_indices]

        train_text_embeds = normalized_text_embeds[train_indices]
        test_text_embeds = normalized_text_embeds[test_indices]

        # Generate train dataset
        train_image_text_embeds = torch.cat((train_image_embeds, train_text_embeds), dim=0)
        # generate labels
        train_labels = torch.cat((torch.zeros(n_train), torch.ones(n_train))) # 0 for image, 1 for text

        # shuffle
        shuffle_indices = torch.randperm(2 * n_train)

        train_image_text_embeds = train_image_text_embeds[shuffle_indices]
        train_labels = train_labels[shuffle_indices]

        # Generate test dataset
        test_image_text_embeds = torch.cat((test_image_embeds, test_text_embeds), dim=0)
        # generate labels
        test_labels = torch.cat((torch.zeros(n_test), torch.ones(n_test))) # 0 for image, 1 for text

        # shuffle
        test_shuffle_indices = torch.randperm(2 * n_test)

        test_image_text_embeds = test_image_text_embeds[test_shuffle_indices]
        test_labels = test_labels[test_shuffle_indices]

        

        
        print('fitting linear classifier')
        # fit linear classifier on train set to predict text embeddings from image embeddings
        clf = LogisticRegression(random_state=wandb.config['seed']).fit(train_image_text_embeds.cpu(), train_labels.cpu())

        # get accuracy on test set
        linear_seperability_accuracy = clf.score(test_image_text_embeds.cpu(), test_labels.cpu())

        print('linear_seperability_accuracy ', linear_seperability_accuracy) 

        return linear_seperability_accuracy



    def get_linear_seperability_between_diff_layers(self):

        image_embeds = self.val_outputs.image_embeds
        
        n_train = int(0.2 * len(image_embeds))
        n_test = len(image_embeds) - n_train

        if wandb.config['encoder1_modality'] == wandb.config['encoder2_modality']:

            e1_e2_linear_seperability_accuracies = [] # tracks linear seperability accuracy of each layer in layers_to_use

            for e1_pool, e2_pool in tqdm(zip(self.encoder1_pooled_hidden_states, self.encoder2_pooled_hidden_states)):

                # generate train dataset taking first n_train elements from each pool
                train_e1_pool = e1_pool[:n_train]
                train_e2_pool = e2_pool[:n_train]

                # generate test dataset taking last n_test elements from each pool
                test_e1_pool = e1_pool[n_train:]
                test_e2_pool = e2_pool[n_train:]

                # Generate train dataset
                train_e1_e2_pool = torch.cat((train_e1_pool, train_e2_pool), dim=0)
                # generate labels
                train_e1_e2_labels = torch.cat((torch.zeros(n_train), torch.ones(n_train))) # 0 for image, 1 for text

                # generate test dataset
                test_e1_e2_pool = torch.cat((test_e1_pool, test_e2_pool), dim=0)
                # generate labels
                test_e1_e2_labels = torch.cat((torch.zeros(n_test), torch.ones(n_test))) # 0 for image, 1 for text

                clf = LogisticRegression(random_state=wandb.config['seed']).fit(train_e1_e2_pool.cpu(), train_e1_e2_labels.cpu())

                # get accuracy on test set
                e1_e2_linear_seperability_accuracy = clf.score(test_e1_e2_pool.cpu(), test_e1_e2_labels.cpu())

                e1_e2_linear_seperability_accuracies.append(e1_e2_linear_seperability_accuracy)


            return e1_e2_linear_seperability_accuracies





    def get_rsa_correlations(self, temperature: int):

        '''
         - Before interchanging
        '''

        text_embeds = self.outputs_to_use.text_embeds
        image_embeds = self.outputs_to_use.image_embeds

        text_encoder_outputs = text_embeds # shape: ([batch_size, 512])
        image_encoder_outputs = image_embeds

        val_logits_per_image = self.outputs_to_use.logits_per_image

        # normalize features
        text_encoder_outputs = text_encoder_outputs / torch.norm(text_encoder_outputs, dim=1, keepdim=True)
        image_encoder_outputs = image_encoder_outputs / torch.norm(image_encoder_outputs, dim=1, keepdim=True)

        # cosine similarities between text-text pairs
        text_text_cosine_similarities = text_encoder_outputs @ text_encoder_outputs.t() # shape: ([batch_size, batch_size])

        # cosine similarities between image-image pairs
        image_image_cosine_similarities = image_encoder_outputs @ image_encoder_outputs.t()

        text_RSM = text_text_cosine_similarities[torch.tril(torch.ones(text_text_cosine_similarities.shape[0], text_text_cosine_similarities.shape[1]), diagonal=-1).bool()]

        image_RSM = image_image_cosine_similarities[torch.tril(torch.ones(image_image_cosine_similarities.shape[0], image_image_cosine_similarities.shape[1]), diagonal=-1).bool()]

        del text_text_cosine_similarities, image_image_cosine_similarities

        result = stats.spearmanr(text_RSM.cpu(), image_RSM.cpu())
        pearson_result = stats.pearsonr(text_RSM.cpu(), image_RSM.cpu())

        print('correlation before interchanging', result.correlation)
        print('p value ', result.pvalue)

        print('pearson correlation before interchanging', pearson_result.statistic)
        print('pearson p value ', pearson_result.pvalue)

        pearson_rsa_before_interchanging = pearson_result.statistic

        rsa_before_interchanging = result.correlation

        # scale with temp
        image_text_cosine_similarities = val_logits_per_image * temperature


        image_text_RSM = image_text_cosine_similarities[torch.tril(torch.ones(image_text_cosine_similarities.shape[0], image_text_cosine_similarities.shape[1]), diagonal=-1).bool()]

        del image_text_cosine_similarities

        text_inter_result = stats.spearmanr(image_text_RSM.cpu(), text_RSM.cpu())
        image_inter_result = stats.spearmanr(image_text_RSM.cpu(), image_RSM.cpu())

        text_inter_pearson_result = stats.pearsonr(image_text_RSM.cpu(), text_RSM.cpu())
        image_inter_pearson_result = stats.pearsonr(image_text_RSM.cpu(), image_RSM.cpu())

        print('--- SPEARMAN CORRELATION ---')

        print('correlation between image-text RSM and text RSM', text_inter_result.correlation)
        print('p value between image-text RSM and text RSM ', text_inter_result.pvalue)
        print('correlation between image-text RSM and image RSM', image_inter_result.correlation)
        print('p value between image-text RSM and image RSM ', image_inter_result.pvalue)

        print('--- PEARSON CORRELATION ---')

        print('pearson correlation between image-text RSM and text RSM', text_inter_pearson_result.statistic)
        print('pearson p value between image-text RSM and text RSM ', text_inter_pearson_result.pvalue)
        print('pearson correlation between image-text RSM and image RSM', image_inter_pearson_result.statistic)
        print('pearson p value between image-text RSM and image RSM ', image_inter_pearson_result.pvalue)

        text_intermodality_rsa = text_inter_result.correlation
        image_intermodality_rsa = image_inter_result.correlation

        pearson_text_intermodality_rsa = text_inter_pearson_result.statistic
        pearson_image_intermodality_rsa = image_inter_pearson_result.statistic


        return {
            'rsa_before_interchanging': rsa_before_interchanging,
            'text_intermodality_rsa': text_intermodality_rsa,
            'image_intermodality_rsa': image_intermodality_rsa,
            'pearson_rsa_before_interchanging': pearson_rsa_before_interchanging,
            'pearson_text_intermodality_rsa': pearson_text_intermodality_rsa,
            'pearson_image_intermodality_rsa': pearson_image_intermodality_rsa
        }


    def get_rsa_correlations_between_diff_layers(self):

        if wandb.config['encoder1_modality'] == wandb.config['encoder2_modality']:


            e1_e2_inter_intra_rsas = [] # tracks RSA between inter and intra modality for each layer in layers_to_use

            e1_e2_rsas = [] # tracks RSA between e1 and e2 hidden states for each layer in layers_to_use

            

            for e1_pool, e2_pool in zip(self.encoder1_pooled_hidden_states, self.encoder2_pooled_hidden_states):

                # get RSMs
                e1_cosine_similarity_mat = e1_pool @ e1_pool.t()
                e2_cosine_similarity_mat = e2_pool @ e2_pool.t()


            
                e1_RSM = e1_cosine_similarity_mat[torch.tril(torch.ones(e1_cosine_similarity_mat.shape[0], e1_cosine_similarity_mat.shape[1]), diagonal=-1).bool()]

                e2_RSM = e2_cosine_similarity_mat[torch.tril(torch.ones(e2_cosine_similarity_mat.shape[0], e2_cosine_similarity_mat.shape[1]), diagonal=-1).bool()]

                e1_e2_cosine_similarities = e1_pool @ e2_pool.t()

                e1_e2_RSM = e1_e2_cosine_similarities[torch.tril(torch.ones(e1_e2_cosine_similarities.shape[0], e1_e2_cosine_similarities.shape[1]), diagonal=-1).bool()]

                # print('e1 rsm ', e1_RSM.cpu())
                # print('e2 rsm ', e2_RSM.cpu())

                # get RSA between e1 and e2 hidden states
                e1_e2_rsa = stats.spearmanr(e1_RSM.cpu(), e2_RSM.cpu()).correlation
                e1_inter_rsa = stats.spearmanr(e1_RSM.cpu(), e1_e2_RSM.cpu()).correlation
                e2_inter_rsa = stats.spearmanr(e2_RSM.cpu(), e1_e2_RSM.cpu()).correlation

                mean_inter_intra_rsa = (e1_inter_rsa + e2_inter_rsa) / 2

                e1_e2_rsas.append(e1_e2_rsa)
                e1_e2_inter_intra_rsas.append(mean_inter_intra_rsa)

            return {
                'e1_e2_rsas': e1_e2_rsas,
                'e1_e2_inter_intra_rsas': e1_e2_inter_intra_rsas
            }
        


    def get_intra_modality_similarities_within_diff_layers(self):

        '''
        - Intra similarities for each layer hidden states
        This is the only metric I can measure when encoders have different modalities
        '''
        e1_cosine_similarities = [] # tracks cosine similarities within e1 hidden states (intra e1 cos sim) for each layer in layers_to_use
        e2_cosine_similarities = [] # tracks cosine similarities within e2 hidden states (intra e2 cos sim) for each layer in layers_to_use

        for e1_pool, e2_pool in zip(self.encoder1_pooled_hidden_states, self.encoder2_pooled_hidden_states):

            # get RSMs
            e1_cosine_similarity_mat = e1_pool @ e1_pool.t()
            e2_cosine_similarity_mat = e2_pool @ e2_pool.t()

            mean_e1_cosine_similarity = e1_cosine_similarity_mat[torch.tril(torch.ones(e1_cosine_similarity_mat.shape[0], e1_cosine_similarity_mat.shape[1]), diagonal=-1).bool()]

            mean_e2_cosine_similarity = e2_cosine_similarity_mat[torch.tril(torch.ones(e2_cosine_similarity_mat.shape[0], e2_cosine_similarity_mat.shape[1]), diagonal=-1).bool()].mean()

            # make sure number of elements in lower triangle is correct
            assert len(mean_e1_cosine_similarity) == (e1_cosine_similarity_mat.shape[0] * (e1_cosine_similarity_mat.shape[0] - 1)) / 2

            mean_e1_cosine_similarity = mean_e1_cosine_similarity.mean() # doing assertion for one only becuase I coded the rest the same way

            e1_cosine_similarities.append(mean_e1_cosine_similarity)
            e2_cosine_similarities.append(mean_e2_cosine_similarity)

        return {
            'e1_cosine_similarities': e1_cosine_similarities,
            'e2_cosine_similarities': e2_cosine_similarities
        }




