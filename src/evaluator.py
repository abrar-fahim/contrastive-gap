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


import sys
import os

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# add sibling directory to path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))# def 

from src.utils import generate_csv_file_name, get_embeddings_path
from src.config import *
from clips.hf_clip import HFClipOutput, HFClip
from dataset_processors.mscoco_processor import MSCOCOProcessor # change this to dataset_processor_parent later, after you write the abstract functions there.

from dataset_processors.dataset_processor_parent import DatasetProcessorParent
from dataset_processors.cifar10_processor import CIFAR10Processor
from dataset_processors.food101_processor import Food101Processor
from dataset_processors.cifar100_processor import CIFAR100Processor
from dataset_processors.sun397_processor import SUN397Processor
from dataset_processors.dtd_processor import DTDProcessor
from dataset_processors.caltech101_processor import Caltech101Processor
from dataset_processors.fgvc_aircraft_processor import FGVCAircraftProcessor
from dataset_processors.stanford_cars_processor import StanfordCarsProcessor

class Evaluator():
    def __init__(self, dataset_processor: MSCOCOProcessor):
        self.dataset_processor = dataset_processor

        self.mscoco_batch_file_path = f"datasets/mscoco/val_batch_cache_{generate_csv_file_name()}.pt"

        self.mscoco_train_dataset_batch_file_path = self.mscoco_batch_file_path

        self.mscoco_val_dataloader: torch.utils.data.DataLoader = None
        self.mscoco_train_dataloader: torch.utils.data.DataLoader = None

        self.val_outputs: HFClipOutput = None


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

            self.zero_shot_datasets: list[DatasetProcessorParent] = [CIFAR10Processor()]    

        self.encoder1_pooled_hidden_states = []
        self.encoder2_pooled_hidden_states = []
        self.layers_to_use = [0, 3, 6, 9, 12] # these are the layers to use for computing mean cosine similarity


        '''
        setting dataloaders
        '''
        if not (os.path.exists(self.mscoco_batch_file_path) and wandb.config['use_cached_val_batch']):

            if wandb.config['use_train_as_val']:
                mscoco_val_dataset =  self.dataset_processor.train_dataset
                batch_size = wandb.config['small_train_loader_batch_size'] # MAYBE CHANGE LATER
            else:
                mscoco_val_dataset =  self.dataset_processor.val_dataset
                batch_size = wandb.config['validation_batch_size']

            collate_fn = self.dataset_processor.collate_fn
            mscoco_val_dataloader = torch.utils.data.DataLoader(mscoco_val_dataset, batch_size=batch_size, collate_fn=collate_fn, generator=torch.Generator().manual_seed(wandb.config['seed']))



            # for now, using val dataset as train dataset.
            self.train_dataloader = mscoco_val_dataloader


        
        '''
        Loading cached batch from file
        '''

        if os.path.exists(self.mscoco_batch_file_path) and wandb.config['use_cached_val_batch']:
            print('loading batch from cache')

            (mscoco_val_imgs, mscoco_val_captions) = torch.load(self.mscoco_batch_file_path)
            print('loading cache done')

        else:
            print('LOADING VAL BATCH')
            for batch in mscoco_val_dataloader:
                print('loading val batch')
                # (val_imgs, val_captions) = next(iter(val_dataloader))
                (mscoco_val_imgs, mscoco_val_captions) = batch
                print('val batch loading done')

            if wandb.config['use_cached_val_batch']:
                print('saving batch to cache')
                # save batch to cache
                torch.save((mscoco_val_imgs, mscoco_val_captions), self.mscoco_batch_file_path)

                del mscoco_val_dataloader

        self.mscoco_val_imgs = mscoco_val_imgs
        self.mscoco_val_captions = mscoco_val_captions


    def evaluate_model(self, clip_model: HFClip, epoch: int, index: int):

        with torch.no_grad():

            self.set_val_outputs(clip_model)
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

            rsa_correlations = self.get_rsa_correlations(clip_model.temperature)

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

            if wandb is not None:
                wandb.log(
                    data={
                        'val_image_classification_accuracy': self.get_val_image_classification_acc(),
                        'val_image_retrieval_accuracy': self.get_val_image_retrieval_acc(),
                        'train_intramodality_loss': val_loss['intra_modality'],
                        'train_intermodality_loss': val_loss['inter_modality'],
                        'train_rsa_loss': val_loss['rsa'],
                        'train_pearson_loss': val_loss['pearson_rsa'],
                        'train_total_loss':val_loss['total'],
                        'mean_pairwise_euclidean_distance':  self.get_mean_pairwise_euclidean_distance(),
                        'mean_cosine_similarity': self.get_mean_cosine_similarity(clip_model.temperature),
                        'linear_seperability_accuracy': self.get_linear_seperability(),
                        'centroid_cosine_similarity': self.get_centroid_cosine_similarity(),
                        'centroid_euclidean_distance': self.get_centroid_euclidean_distance(),

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
                        'non_similar_mean_cosine_similarity': self.non_similar_mean_cosine_similarity(clip_model.temperature),
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


                        'cifar10_val_image_classification_accuracy': self.get_dataset_zero_shot_acc(clip_model, self.zero_shot_datasets[0]),
                        
                    },
                    # step = int(epoch * (len(dataset_processor.train_dataloader) // wandb.config['batch_size']) + index) # this may not work with WIT dataset, check later
                    # step= int(epoch * 470 + index), # by 100 to maintain fair comparison with existing runs data
                    step= int(epoch * self.dataset_processor.get_num_batches() + index), # by 100 to maintain fair comparison with existing runs data
                    

                )


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


        
    def set_val_outputs(self, clip_model: HFClip):
        
        with torch.no_grad():
            clip_model.eval()
            # val_outputs = do_validation(clip_model, mscoco_val_imgs, mscoco_val_captions, device, self.wandb)
            val_outputs: HFClipOutput = clip_model(self.mscoco_val_imgs, self.mscoco_val_captions, output_loss=True, return_all=True, output_hidden_states=True, output_intra_modality_loss=True) # outputting everything all at once and storing them
            self.val_outputs = val_outputs


    def set_pooled_hidden_states(self, clip_model: HFClip):
        # both image and text encoders have 13 outputs in the hidden_states list
        # image encoder each hidden state shape = ([batch_size, 50, 768]) == (batch_size, sequence_length, hidden_size)
        # text encoder each hidden state shape = ([batch_size, 26, 512]) == (batch_size, sequence_length, hidden_size)

        # get mean cosine similarity between encoder outputs at different layers
        encoder1_hidden_states = self.val_outputs.encoder1_hidden_states
        encoder2_hidden_states = self.val_outputs.encoder2_hidden_states

        for layer in self.layers_to_use:

            # pool hidden states to convert from sequence to single value

            e1_pool = clip_model.encoder1.pool_hidden_state(encoder1_hidden_states[layer], self.val_outputs.encoder1_input_ids)

            e2_pool = clip_model.encoder2.pool_hidden_state(encoder2_hidden_states[layer], self.val_outputs.encoder2_input_ids)


            # pooled_hidden_states shape: ([batch_size, hidden_size])

            # following assertions work because in HFClipOutput, image_embeds = encoder1 and text_embeds = encoder2

            image_embeds = self.val_outputs.image_embeds
            text_embeds = self.val_outputs.text_embeds

            assert e1_pool.shape == (image_embeds.shape[0], clip_model.encoder1.hidden_size), f'e1_pool.shape = {e1_pool.shape}, expected shape = ({image_embeds.shape[0]}, {clip_model.encoder1.hidden_size})'

            assert e2_pool.shape == (text_embeds.shape[0], clip_model.encoder2.hidden_size), f"e2_pool.shape = {e2_pool.shape}, expected shape = ({text_embeds.shape[0]}, {clip_model.encoder2.hidden_size})"



            # normalize features
            e1_pool = e1_pool / torch.norm(e1_pool, dim=1, keepdim=True)
            e2_pool = e2_pool / torch.norm(e2_pool, dim=1, keepdim=True)

            self.encoder1_pooled_hidden_states.append(e1_pool)
            self.encoder2_pooled_hidden_states.append(e2_pool)

    def save_pooled_hidden_states_to_file(self, save_path:str, epoch:int, index:int):

        print('saving encoder hidden states to ', save_path)

        n_to_save = wandb.config['n_embeds_to_save']

        # take first n_to_save embeds from each layer
        encoder1_pooled_hidden_states_to_save = [e1_pool[:n_to_save] for e1_pool in self.encoder1_pooled_hidden_states]

        # e_pool shape = [batch_size, CLIPConfig.hidden_size]

        encoder2_pooled_hidden_states_to_save = [e2_pool[:n_to_save] for e2_pool in self.encoder2_pooled_hidden_states]

        image_embeds = self.val_outputs.image_embeds
        text_embeds = self.val_outputs.text_embeds

        # normalize features
        normalized_encoder1_embeds = image_embeds / torch.norm(image_embeds, dim=1, keepdim=True)
        normalized_encoder2_embeds = text_embeds / torch.norm(text_embeds, dim=1, keepdim=True)


        step_data = {
                'step': int(epoch * (len(self.dataset_processor.train_dataloader) // wandb.config['batch_size']) + index),
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



    def get_val_image_classification_acc(self):

        val_logits_per_image = self.val_outputs.logits_per_image # shape of both: ([64, 64])

        # image embeddings
        image_embeds = self.val_outputs.image_embeds # normalized_encoder1 embeds. Shape: ([batch_size, 512])
        text_embeds = self.val_outputs.text_embeds # normalized_encoder2 embeds

        # normalize
        normalized_encoder1_embeds = image_embeds / torch.norm(image_embeds, dim=1, keepdim=True)
        normalized_encoder2_embeds = text_embeds / torch.norm(text_embeds, dim=1, keepdim=True)


        # softmax on logits_per_image
        val_image_class_probs = F.softmax(val_logits_per_image, dim=-1) # shape: ([64, 64])


       
        # calculate accuracy
        # get indices of max values
        val_image_class_preds = val_image_class_probs.argmax(dim=-1) # shape: ([64])


        val_image_class_labels = torch.arange(val_image_class_probs.shape[0], device=val_image_class_probs.device) # shape: ([64])


        # calculate accuracy
        val_image_classification_accuracy = (val_image_class_preds == val_image_class_labels).float().mean()

        return val_image_classification_accuracy.item()

    def get_dataset_zero_shot_acc(self, clip_model: HFClip, dataset_processor: DatasetProcessorParent) -> float:

        if dataset_processor == None:
            return  0


        with torch.no_grad():

            
            cifar_classes = dataset_processor.classes

            # tokenize captions
            # cifar_tokenized_classes = clip_model.tokenize_captions(cifar_classes)



            cifar10_val_image_classification_accuracy_runsum = 0
            for batch in tqdm(dataset_processor.val_dataloader):
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

            cifar10_val_image_classification_accuracy = cifar10_val_image_classification_accuracy_runsum / len(dataset_processor.val_dataset)
            print(f'{dataset_processor.name} zero shot accuracy ', cifar10_val_image_classification_accuracy.item())

        
            return cifar10_val_image_classification_accuracy


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
        
            


    def get_val_image_retrieval_acc(self):
        logits_per_text = self.val_outputs.logits_per_text # shape of both: ([64, 64])

        # softmax on logits_per_text
        text_class_probs = F.softmax(logits_per_text, dim=-1)
        
        # calculate accuracy
        # get indices of max values: These are indices of the retrieved images
        text_class_preds = text_class_probs.argmax(dim=-1)

        # get indices of correct predictions
        val_text_class_labels = torch.arange(text_class_probs.shape[0], device=text_class_probs.device) # shape: ([64])

        # calculate accuracy
        val_image_retrieval_accuracy = (text_class_preds == val_text_class_labels).float().mean()

        # print('retrieval done')
        return val_image_retrieval_accuracy.item()
    


    def get_val_loss(self):

        loss = self.val_outputs.loss

        return loss

    def get_mean_cosine_similarity(self, temperature: int):

        cosine_similarities = self.val_outputs.logits_per_image.diag() # shape: [64]
        # get mean cosine similarity
        mean_cosine_similarity = torch.mean(cosine_similarities)

        # scale with temperature
        mean_cosine_similarity = mean_cosine_similarity * temperature

        print('mean cosine similarity ', mean_cosine_similarity.item())

        return mean_cosine_similarity.item()

    

    def non_similar_mean_cosine_similarity(self, temperature: int):

        # get mean of elements that are not on the diagonal
        non_similar_mean_cosine_similarity = self.val_outputs.logits_per_image[~torch.eye(self.val_outputs.logits_per_image.shape[0], dtype=bool)].mean()

        # scale with temperature

        non_similar_mean_cosine_similarity = non_similar_mean_cosine_similarity * temperature


        print('non_similar_mean_cosine_similarity ', non_similar_mean_cosine_similarity * temperature)




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

        text_embeds = self.val_outputs.text_embeds

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

        image_embeds = self.val_outputs.image_embeds

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

        image_embeds = self.val_outputs.image_embeds
        text_embeds = self.val_outputs.text_embeds

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

        image_embeds = self.val_outputs.image_embeds
        text_embeds = self.val_outputs.text_embeds

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




    def get_centroid_euclidean_distance(self):
            
            '''
            - Euclidean distance between image and text centroids
            '''
    
            image_embeds = self.val_outputs.image_embeds
            text_embeds = self.val_outputs.text_embeds
    
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



    def get_linear_seperability(self):
        # Split validation dataset into train and test splits
        # train on 20% of the data and test on 80%

        image_embeds = self.val_outputs.image_embeds
        text_embeds = self.val_outputs.text_embeds


        normalized_image_embeds = image_embeds / torch.norm(image_embeds, dim=1, keepdim=True)
        normalized_text_embeds = text_embeds / torch.norm(text_embeds, dim=1, keepdim=True)

        # check if normalization happened properly as expected
        assert torch.allclose(torch.norm(normalized_image_embeds, dim=1), torch.ones(normalized_image_embeds.shape[0], device=normalized_image_embeds.device))
        assert torch.allclose(torch.norm(normalized_text_embeds, dim=1), torch.ones(normalized_text_embeds.shape[0], device=normalized_text_embeds.device))

        n_train = int(0.2 * len(image_embeds))
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

        text_embeds = self.val_outputs.text_embeds
        image_embeds = self.val_outputs.image_embeds

        text_encoder_outputs = text_embeds # shape: ([batch_size, 512])
        image_encoder_outputs = image_embeds

        val_logits_per_image = self.val_outputs.logits_per_image

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




