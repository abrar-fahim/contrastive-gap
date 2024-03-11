import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from evaluate import load as load_evaluator
from src.config import *
import wandb
import os
from tqdm import tqdm
from scipy import stats

from torchvision.datasets import CIFAR10
from sklearn.linear_model import LogisticRegression
from typing import Any, Optional, Tuple, Union

import sys
import os
from clips.hf_clip import HFClipOutput, HFClip

from dataset_processors.mscoco_processor import MSCOCOProcessor # change this to dataset_processor_parent later, after you write the abstract functions there.

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# add sibling directory to path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))# def do_validation(val_dataset, train_dataset, clip_model, index=0, epoch=0, captioning_model=False):





def do_validation(dataset_processor: MSCOCOProcessor, clip_model: HFClip, index=0, epoch=0, captioning_model=False, wandb=wandb, val_dataset_processor = None):

    


    
    '''
    Report accuracy and mean cosine similarity on validation set
    Report text-text and image-image cosine similarities
    Dump numbers to csv file
    '''

    # print('validation started')

    # create seperate dataloaders for val and train dataset, seperate from the ones used in training, so that I get same train and val batch each time this runs

    
    mscoco_batch_file_path = f"datasets/mscoco/val_batch_cache_{training_hyperparameters['seed']}.pt"
    mscoco_train_dataset_batch_file_path = f"datasets/mscoco/train_batch_cache_{training_hyperparameters['seed']}.pt"

    if not (os.path.exists(mscoco_batch_file_path) and training_hyperparameters['use_cached_val_batch']):
        # only create dataloaders if batch is not cached

        mscoco_val_dataset = dataset_processor.val_dataset
        collate_fn = dataset_processor.collate_fn
        mscoco_val_dataloader = torch.utils.data.DataLoader(mscoco_val_dataset, batch_size=training_hyperparameters['validation_batch_size'], collate_fn=collate_fn, generator=torch.Generator().manual_seed(training_hyperparameters['seed']))

       

    if not (os.path.exists(mscoco_train_dataset_batch_file_path) and training_hyperparameters['use_cached_val_batch']):
        collate_fn = dataset_processor.collate_fn
        train_dataset = dataset_processor.train_dataset
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=training_hyperparameters['validation_batch_size'], collate_fn=collate_fn, generator=torch.Generator().manual_seed(training_hyperparameters['seed']))

    
        
        
    if val_dataset_processor != None:
        # for CIFAR10 and other zero shot datasets
        cifar_val_dataset = val_dataset_processor.val_dataset
        collate_fn = None
        # cifar_batch_file_path = f"datasets/cifar10/val_batch_cache_{training_hyperparameters['seed']}.pt"
        # batch contains (images, index of target class)
        cifar_val_dataloader = torch.utils.data.DataLoader(cifar_val_dataset, batch_size=training_hyperparameters['cifar_batch_size'], num_workers=training_hyperparameters['num_workers'])

    

    # print('defining dataloaders')



    # create dataloader for validation set
    # creating dataloader seperately here instead of using the one inside dataset_processor to set the manual seed explicitly so that I get same batch each time
    

    # print('defining dataloaders done')

    # create dataloader for train set
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=training_hyperparameters['validation_batch_size'], collate_fn=collate_fn, generator=torch.Generator().manual_seed(training_hyperparameters['seed']))



    with torch.no_grad():
        # get batch from validation set

        if os.path.exists(mscoco_batch_file_path) and training_hyperparameters['use_cached_val_batch']:
            print('loading batch from cache')

            (mscoco_val_imgs, mscoco_val_captions) = torch.load(mscoco_batch_file_path)
            print('loading cache done')

        else:
            for batch in mscoco_val_dataloader:
                # (val_imgs, val_captions) = next(iter(val_dataloader))
                (mscoco_val_imgs, mscoco_val_captions) = batch

            if training_hyperparameters['use_cached_val_batch']:
                print('saving batch to cache')
                # save batch to cache
                torch.save((mscoco_val_imgs, mscoco_val_captions), mscoco_batch_file_path)

                del mscoco_val_dataloader

        

        

        if clip_caption_model_train_hyperparameters['show_real_images']:

            # show the first 10 images from the validation set in a subplot
            fig = plt.figure()

            
                
            for i in range(10):
                ax = plt.subplot(2, 5, i + 1)
                plt.imshow(val_imgs[i + 5].permute(1, 2, 0))
                # plt.title(captions[i])
                plt.axis("off")
                
                print(val_captions[i+5])


            plt.show()


        
        
        val_outputs: HFClipOutput = clip_model(mscoco_val_imgs, mscoco_val_captions, output_loss=False, return_all=True, output_hidden_states=True)
        

        del mscoco_val_imgs, mscoco_val_captions


        '''
        1. Validation image classification accuracy
        '''
        val_logits_per_image = val_outputs.logits_per_image # shape of both: ([64, 64])

        # image embeddings
        image_embeds = val_outputs.image_embeds
        text_embeds = val_outputs.text_embeds


        # softmax on logits_per_image
        val_image_class_probs = F.softmax(val_logits_per_image, dim=-1) # shape: ([64, 64])


       
        # calculate accuracy
        # get indices of max values
        val_image_class_preds = val_image_class_probs.argmax(dim=-1) # shape: ([64])


        val_image_class_labels = torch.arange(val_image_class_probs.shape[0], device=val_image_class_probs.device) # shape: ([64])


        # calculate accuracy
        val_image_classification_accuracy = (val_image_class_preds == val_image_class_labels).float().mean()


        '''
        1.1 Validation image classification accuracy on CIFAR10
        '''

        if val_dataset_processor != None:
            cifar_classes = val_dataset_processor.classes

            # tokenize captions
            cifar_tokenized_classes = clip_model.tokenize_captions(cifar_classes)

            cifar10_val_image_classification_accuracy_runsum = 0
            for batch in tqdm(cifar_val_dataloader):
                (cifar_val_imgs, cifar_val_indices) = batch
                
                
                # get logits per image
                cifar_val_outputs = clip_model(cifar_val_imgs, cifar_tokenized_classes, output_loss=False, return_all=True)

                cifar_val_logits_per_image = cifar_val_outputs.logits_per_image # shape of both: ([64, 64])

                # softmax on logits_per_image
                cifar_val_image_class_probs = F.softmax(cifar_val_logits_per_image, dim=-1) # shape: ([batch_size, 10]). 10 is num_classes in cifar10

                # calculate accuracy
                # get indices of max values
                cifar_val_image_class_preds = cifar_val_image_class_probs.argmax(dim=-1) # shape: ([batch_size])

                cifar_val_indices = cifar_val_indices.to(clip_model.device)

                cifar10_val_image_classification_accuracy_runsum += (cifar_val_image_class_preds == cifar_val_indices).float().sum()
                # print('cifar10_val_image_classification_accuracy_runsum ', cifar10_val_image_classification_accuracy_runsum)

            cifar10_val_image_classification_accuracy = cifar10_val_image_classification_accuracy_runsum / len(cifar_val_dataset)
            print('cifar10_val_image_classification_accuracy ', cifar10_val_image_classification_accuracy.item())




        '''
        2. Validation image retrieval accuracy
        '''

        logits_per_text = val_outputs.logits_per_text # shape of both: ([64, 64])

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


        '''
        3. Training image classification accuracy
        '''

        if os.path.exists(mscoco_train_dataset_batch_file_path) and training_hyperparameters['use_cached_val_batch']:
            print('loading train batch from cache')
            (train_imgs, train_captions) = torch.load(mscoco_train_dataset_batch_file_path)
            print('loading cache done')
        else:
            print('loading train batch')
            for batch in train_dataloader:
                (train_imgs, train_captions) = batch
                break # loading a single train batch for now

            if training_hyperparameters['use_cached_val_batch']:
                print('saving train batch to cache')
                # save batch to cache
                torch.save((train_imgs, train_captions), mscoco_train_dataset_batch_file_path)

                del train_dataloader
            

        # (train_imgs, train_captions) = next(iter(train_dataloader))

        train_outputs: HFClipOutput = clip_model(train_imgs, train_captions, output_loss=True, return_all=True, output_intra_modality_loss=True) # so tha I get cosine similarities directly
        train_logits_per_image = train_outputs.logits_per_image # shape of both: ([64, 64])
        train_image_class_probs = F.softmax(train_logits_per_image, dim=-1) # shape: ([64, 64])
        train_image_class_preds = train_image_class_probs.argmax(dim=-1) # shape: ([64])
        train_image_class_labels = torch.arange(train_image_class_probs.shape[0], device=train_image_class_probs.device) # shape: ([64])
        train_image_classification_accuracy = (train_image_class_preds == train_image_class_labels).float().mean()

        # train_loss = train_outputs.loss.item()

        train_intra_loss = train_outputs.loss['intra_modality']
        train_rsa_loss = train_outputs.loss['rsa']
        train_pearson_loss = train_outputs.loss['pearson_rsa']
        train_inter_loss = train_outputs.loss['inter_modality']
        train_loss = train_outputs.loss['total']

        del train_outputs

        print('train_intermodality_loss ', train_inter_loss)


        print('--- ACCURACY STUFF --- ')

        # print('image preds ', image_class_preds)
        # print('image labels ', image_class_labels)

        print('validation image_accuracy ', val_image_classification_accuracy.item())
        # print('train image_accuracy ', train_image_classification_accuracy.item())


        print('--- IMAGE-TEXT SIMILARITIES --- ')

        # print('logits_per_image ', logits_per_image)

        # print logits per image for first 5 images
        # print('logits_per_image ', logits_per_image[:5, :5])
        cosine_similarities = val_logits_per_image.diag() # shape: [64]
        # get mean cosine similarity
        mean_cosine_similarity = torch.mean(cosine_similarities)

        # scale with temperature
        mean_cosine_similarity = mean_cosine_similarity * clip_model.temperature

        print('mean cosine similarity ', mean_cosine_similarity.item())


        # get mean of elements that are not on the diagonal
        non_similar_mean_cosine_similarity = val_logits_per_image[~torch.eye(val_logits_per_image.shape[0], dtype=bool)].mean()

        # scale with temperature

        non_similar_mean_cosine_similarity = non_similar_mean_cosine_similarity * clip_model.temperature


        print('non_similar_mean_cosine_similarity ', non_similar_mean_cosine_similarity * clip_model.temperature)

        print('encoder 1 hidden states', val_outputs.encoder1_hidden_states[0].shape)

        print('encoder 2 hidden states', val_outputs.encoder2_hidden_states[0].shape)




        print('encoder 1 num hidden states ', len(val_outputs.encoder1_hidden_states))
        print('encoder 2 num hidden states ', len(val_outputs.encoder2_hidden_states))


        '''
        - Get text-text similarities
        '''
        
        text_encoder_outputs = text_embeds # shape: ([batch_size, 512])
    

        # normalize features
        text_encoder_outputs = text_encoder_outputs / torch.norm(text_encoder_outputs, dim=1, keepdim=True)

        # cosine similarities between text-text pairs
        text_text_cosine_similarities = text_encoder_outputs @ text_encoder_outputs.t() # shape: ([batch_size, batch_size])

        # get median of elements that are in the upper triangle (excluding diagonal!!)
        mean_text_text_cosine_similarity = text_text_cosine_similarities[torch.triu(torch.ones(text_text_cosine_similarities.shape[0], text_text_cosine_similarities.shape[1]), diagonal=1).bool()].mean()

        # print('median_text_text_cosine_similarity ', median_text_text_cosine_similarity)

        '''
        - Get image-image similarities
        '''

        

        image_encoder_outputs = image_embeds # shape: ([batch_size, 512])


        # normalize features
        image_encoder_outputs = image_encoder_outputs / torch.norm(image_encoder_outputs, dim=1, keepdim=True)

        # cosine similarities between image-image pairs
        image_image_cosine_similarities = image_encoder_outputs @ image_encoder_outputs.t()

        # get median of elements that are not on the diagonal
        mean_image_image_cosine_similarity = image_image_cosine_similarities[~torch.eye(image_image_cosine_similarities.shape[0], dtype=bool)].mean()



        print('mean_image_image_cosine_similarity ', mean_image_image_cosine_similarity)


        '''
        Euclidean distance between image and text pairs
        '''

        # euclidean distance between image and text pairs
        euclidean_distances = torch.norm(image_encoder_outputs - text_encoder_outputs, dim=-1)

        # get mean euclidean distance
        mean_euclidean_distance = euclidean_distances.mean()

        print('mean_pairwise_euclidean_distance ', mean_euclidean_distance)


        '''
        - Similarity between image and text centroids
        '''

        # get centroids
        text_centroid = text_encoder_outputs.mean(dim=0)
        image_centroid = image_encoder_outputs.mean(dim=0)

        del text_encoder_outputs, image_encoder_outputs


        # euclidean distance between centroids
        centroid_euclidean_distance = torch.norm(text_centroid - image_centroid)


    


        # normalize centroids
        text_centroid = text_centroid / torch.norm(text_centroid, dim=0, keepdim=True)
        image_centroid = image_centroid / torch.norm(image_centroid, dim=0, keepdim=True)

        # cosine similarity between centroids
        centroid_cosine_similarity = text_centroid @ image_centroid.t()

        print('centroid_cosine_similarity ', centroid_cosine_similarity)


        '''
        Calculate cosine similarity quality metric
        '''

        # first, scale the cosine similarities by temperature


        '''
        Linear seperability of image and text embeddings
        '''

        # Split validation dataset into train and test splits
        # train on 20% of the data and test on 80%

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
        clf = LogisticRegression(random_state=0).fit(train_image_text_embeds.cpu(), train_labels.cpu())

        # get accuracy on test set
        linear_seperability_accuracy = clf.score(test_image_text_embeds.cpu(), test_labels.cpu())

        print('linear_seperability_accuracy ', linear_seperability_accuracy) 
        


        '''
        RSA correlations
        '''

        shuffle_ratio = 0.5 # percentage of texts and images to shuffle

        

        '''
         - Before interchanging
        '''

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
        image_text_cosine_similarities = val_logits_per_image * clip_model.temperature

        del val_logits_per_image

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


        '''
        - Slope of linear regression between image-text RSM and text RSM
        '''

        # get slope of linear regression between image-text RSM and text RSM


        text_inter_stats = stats.linregress(image_text_RSM.cpu(), text_RSM.cpu())

        image_inter_stats = stats.linregress(image_text_RSM.cpu(), image_RSM.cpu())

        image_text_stats = stats.linregress(image_RSM.cpu(), text_RSM.cpu())


        text_inter_slope, text_inter_intercept, text_inter_r_value, text_inter_p_value, text_inter_std_err = text_inter_stats

        image_inter_slope, image_inter_intercept, image_inter_r_value, image_inter_p_value, image_inter_std_err = image_inter_stats

        image_text_slope, image_text_intercept, image_text_r_value, image_text_p_value, image_text_std_err = image_text_stats

        '''
        log to wandb
        '''

        average_intra_modality_cosine_sim = (mean_text_text_cosine_similarity.item() + mean_image_image_cosine_similarity.item()) / 2

        if wandb is not None:
            wandb.log(
                data={
                    'val_image_classification_accuracy': val_image_classification_accuracy.item(),
                    'val_image_retrieval_accuracy': val_image_retrieval_accuracy.item(),
                    'train_image_accuracy': train_image_classification_accuracy.item(),
                    'train_intramodality_loss': train_intra_loss,
                    'train_intermodality_loss': train_inter_loss,
                    'train_rsa_loss': train_rsa_loss,
                    'train_pearson_loss': train_pearson_loss,
                    'train_total_loss': train_loss,
                    'mean_pairwise_euclidean_distance': mean_euclidean_distance.item(), # this is the mean of the euclidean distances between image and text pairs
                    'mean_cosine_similarity': mean_cosine_similarity.item(),
                    'linear_seperability_accuracy': linear_seperability_accuracy,
                    'centroid_cosine_similarity': centroid_cosine_similarity.item(),
                    'centroid_euclidean_distance': centroid_euclidean_distance.item(),
                    'non_similar_mean_cosine_similarity': non_similar_mean_cosine_similarity.item(),
                    'mean_text_text_cosine_similarity': mean_text_text_cosine_similarity.item(),
                    'mean_image_image_cosine_similarity': mean_image_image_cosine_similarity.item(),
                    'average_intra_modality_cosine_similarity': average_intra_modality_cosine_sim,
                    'rsa_before_interchanging': rsa_before_interchanging,
                    'text_intermodality_rsa': text_intermodality_rsa,
                    'image_intermodality_rsa': image_intermodality_rsa,
                    'cifar10_val_image_classification_accuracy': cifar10_val_image_classification_accuracy.item() if val_dataset_processor != None else 0,
                    'pearson_rsa_before_interchanging': pearson_rsa_before_interchanging,
                    'pearson_text_intermodality_rsa': pearson_text_intermodality_rsa,
                    'pearson_image_intermodality_rsa': pearson_image_intermodality_rsa,
                    'text_inter_slope': text_inter_slope,
                    'text_inter_intercept': text_inter_intercept,
                    'image_inter_slope': image_inter_slope,
                    'image_inter_intercept': image_inter_intercept,
                    'image_text_slope': image_text_slope,
                    'image_text_intercept': image_text_intercept,
                    
                },
                # step= int(epoch * (len(dataset_processor.train_dataloader) // training_hyperparameters['batch_size']) + index) # this may not work with WIT dataset, check later
                step= int(epoch * 100 + index), # by 100 to maintain fair comparison with existing runs data
                

            )
        
        # set dataprocessor caching back to off
        dataset_processor.use_cached_tokenized_captions = False

        # print('val done')

        '''
        Show real images and captions for the incorrect predictions
        '''

        if training_hyperparameters['show_incorrect_images']:
            # get real images and captions

            dataset_processor.show_real_images_captions = True
            collate_fn = dataset_processor.collate_fn

            # create dataloader for validation set
            real_images_captions_val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=training_hyperparameters['validation_batch_size'], collate_fn=collate_fn, generator=torch.Generator().manual_seed(training_hyperparameters['seed']))


            (images, true_captions) = next(iter(real_images_captions_val_dataloader))

            # get indices of incorrect predictions
            incorrect_preds_mask = (val_image_class_preds != val_image_class_labels)

            # print('incorrect preds mask ', incorrect_preds_mask)

            # print('images shape ', images.shape)

            # display real images and captions for incorrect predictions
            # incorrect_images = torch.masked_select(images, incorrect_preds_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)).reshape(-1, 3, 224, 224) # shape: ([n_wrong, 3, 224, 224]

            incorrect_images = [item for keep, item in zip(incorrect_preds_mask, images) if keep]
            label_captions = [item for keep, item in zip(incorrect_preds_mask, true_captions) if keep]


            del val_image_class_labels
            # get predicted captions
            incorrect_predicted_caption_indices = val_image_class_preds[incorrect_preds_mask] # shape: ([n_wrong])

            del val_image_class_preds

            # get predicted captions
            incorrect_predicted_captions = [item for keep, item in zip(incorrect_predicted_caption_indices, true_captions) if keep] # shape: ([n_wrong])

            

            # show the first 10 images from the validation set in a subplot

            fig = plt.figure()

            for i in range(min(10, len(incorrect_images))):
                ax = plt.subplot(2, 5, i + 1)
                # plt.imshow(incorrect_images[i].permute(1, 2, 0))
                plt.imshow(incorrect_images[i])
                # plt.title(captions[i])
                plt.axis("off")
                print('true: ')
                print(label_captions[i])
                print('predicted: ')
                print(incorrect_predicted_captions[i])
            
            plt.show()

            dataset_processor.show_real_images_captions = False

            # print('predicted captions ', incorrect_predicted_captions[:10])



        '''
        evaluating captioning model
        '''


        if captioning_model:
            from clip_caption.clip_caption_predict import Predictor as MLPPredictor
            from clip_caption.clip_caption_transformer_predict import Predictor as TransformerPredictor
            # text_embeds = outputs.text_model_output.pooler_output # shape: ([batch_size, 512]), these are before linear projection
            # image_embeds = outputs.image_embeds

            if clip_caption_model_train_hyperparameters['model_config'] == ClipCaptionModelMapping.MLP:
                predictor = MLPPredictor()
            elif clip_caption_model_train_hyperparameters['model_config'] == ClipCaptionModelMapping.TRANSFORMER:
                predictor = TransformerPredictor()

            predictor.setup()

            if selected_clip_model == ClipModels.FINETUNED_TEMP:
                # get predictions
                predicted_captions = predictor.predict(val_imgs, "finetuned_caption_temp", False)
                # predicted_captions = predictor.predict(val_imgs, "og_mscoco", False) # using the default model for now

            elif selected_clip_model == ClipModels.FINETUNED:
                # get predictions
                predicted_captions = predictor.predict(val_imgs, "finetuned_caption", False)
            elif selected_clip_model == ClipModels.DEFAULT:
                # get predictions
                predicted_captions = predictor.predict(val_imgs, "og_mscoco", False)

            # predictions is a list of strings

            # print('predictions ', predicted_captions)

            bertscore_evaluator = load_evaluator("bertscore")

            # get bertscore
            bertscores = bertscore_evaluator.compute(predictions=predicted_captions, references=val_captions, model_type="distilbert-base-uncased", lang="en", verbose=True)

            bleu_score_evaluator = load_evaluator("bleu")

            # convert val captions into a list of lists for input to bleu score
            bleu_val_captions = [[caption] for caption in val_captions]

            # get bleu score
            bleu_scores = bleu_score_evaluator.compute(predictions=predicted_captions, references=bleu_val_captions)

            # print first 10 predicted captions and ground truth captions
            print('predicted_captions ', predicted_captions[:10])
            print('val_captions ', val_captions[:10])

            print()
            print(' --- CAPTIONING METRICS --- ')
            print()

            # print('bertscore precision ', bertscores['precision'])
            # print('bertscore recall ', bertscores['recall'])
            # print('bertscore f1 ', bertscores['f1'])

            print('bleu ', bleu_scores['bleu'])
            # print('precisions ', bleu_scores['precisions'])

            # get scores
            precision = np.mean(bertscores['precision'])
            recall = np.mean(bertscores['recall'])
            f1 = 2 * (precision * recall) / (precision + recall)

            print('bertscore precision ', precision)
            print('bertscore recall ', recall)
            print('bertscore f1 ', f1)

            print()
            print(' ROUGE')
            print()

            rouge_evaluator = load_evaluator("rouge")

            # get rouge score
            rouge_scores = rouge_evaluator.compute(predictions=predicted_captions, references=val_captions)

            print('rouge1 ', rouge_scores['rouge1'])
            print('rouge2 ', rouge_scores['rouge2'])
            print('rougeL ', rouge_scores['rougeL'])
            print("rougeLsum ", rouge_scores['rougeLsum'])

