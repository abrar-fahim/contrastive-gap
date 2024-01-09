import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from evaluate import load as load_evaluator
from src.config import *
import wandb

import nltk
import random
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import pickle
from sklearn.decomposition import PCA
import os
from tqdm import tqdm
from scipy import stats

pca = None

random.seed(training_hyperparameters['seed']) # random used in evaluate_concept_arrangement




# def do_validation(val_dataset, train_dataset, clip_model, index=0, epoch=0, captioning_model=False):
def do_validation(dataset_processor, clip_model, index=0, epoch=0, captioning_model=False, wandb=wandb):

    
    '''
    Report accuracy and mean cosine similarity on validation set
    Report text-text and image-image cosine similarities
    Dump numbers to csv file
    '''

    # print('validation started')

    # create seperate dataloaders for val and train dataset, seperate from the ones used in training, so that I get same train and val batch each time this runs

    val_dataset = dataset_processor.val_dataset
    train_dataset = dataset_processor.train_dataset
    # set caching to true only for inside do_validation
    dataset_processor.use_cached_tokenized_captions = True
    collate_fn = dataset_processor.collate_fn

    # print('defining dataloaders')



    # create dataloader for validation set
    # creating dataloader seperately here instead of using the one inside dataset_processor to set the manual seed explicitly so that I get same batch each time
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=training_hyperparameters['validation_batch_size'], collate_fn=collate_fn, generator=torch.Generator().manual_seed(training_hyperparameters['seed']))

    # print('defining dataloaders done')

    # create dataloader for train set
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=training_hyperparameters['validation_batch_size'], collate_fn=collate_fn, generator=torch.Generator().manual_seed(training_hyperparameters['seed']))

    





    with torch.no_grad():
        # get batch from validation set

        # print('torch no grad')

        batch_file_path = f"datasets/mscoco/val_batch_cache_{training_hyperparameters['seed']}.pt"

        if os.path.exists(batch_file_path):
            print('loading batch from cache')
            (val_imgs, val_captions) = torch.load(batch_file_path)
            print('loading cache done')

        else:
            for batch in val_dataloader:
                # (val_imgs, val_captions) = next(iter(val_dataloader))
                (val_imgs, val_captions) = batch
            print('saving batch to cache')
            # save batch to cache
            torch.save((val_imgs, val_captions), batch_file_path)


        # print('batching')
        # (val_imgs, val_captions) = next(iter(val_dataloader))
        


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

        
        # print('clipping')
        val_outputs = clip_model(val_imgs, val_captions, output_loss=False, return_all=True) # so that I get cosine similarities directly
        # print('clipping done')
        '''
        1. Validation image classification accuracy
        '''
        val_logits_per_image = val_outputs.logits_per_image # shape of both: ([64, 64])


        # softmax on logits_per_image
        val_image_class_probs = F.softmax(val_logits_per_image, dim=-1) # shape: ([64, 64])

       
        # calculate accuracy
        # get indices of max values
        val_image_class_preds = val_image_class_probs.argmax(dim=-1) # shape: ([64])

        # print('val_image_class_preds ', val_image_class_preds)


        
        # get indices of correct predictions
        val_image_class_labels = torch.arange(val_image_class_probs.shape[0], device=val_image_class_probs.device) # shape: ([64])

        # calculate accuracy
        val_image_classification_accuracy = (val_image_class_preds == val_image_class_labels).float().mean()



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

        # (train_imgs, train_captions) = next(iter(train_dataloader))

        # train_outputs = clip_model(train_imgs, train_captions, output_loss=True, return_all=True, output_intra_modality_loss=True) # so tha I get cosine similarities directly
        # train_logits_per_image = train_outputs.logits_per_image # shape of both: ([64, 64])
        # train_image_class_probs = F.softmax(train_logits_per_image, dim=-1) # shape: ([64, 64])
        # train_image_class_preds = train_image_class_probs.argmax(dim=-1) # shape: ([64])
        # train_image_class_labels = torch.arange(train_image_class_probs.shape[0], device=train_image_class_probs.device) # shape: ([64])
        # train_image_classification_accuracy = (train_image_class_preds == train_image_class_labels).float().mean()

        # # train_loss = train_outputs.loss.item()

        # train_intra_loss = train_outputs.loss['intra_modality']
        # train_inter_loss = train_outputs.loss['inter_modality']
        # train_loss = train_outputs.loss['total']



        

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
        print('mean cosine similarity ', mean_cosine_similarity.item())


        # get mean of elements that are not on the diagonal
        non_similar_mean_cosine_similarity = val_logits_per_image[~torch.eye(val_logits_per_image.shape[0], dtype=bool)].mean()
        print('non_similar_mean_cosine_similarity ', non_similar_mean_cosine_similarity * clip_model.temperature)



        '''
        - Get text-text similarities
        '''
        # print()
        # print(' --- TEXT-TEXT SIMILARITIES --- ')
        # print()

        text_encoder_outputs = clip_model.encode_text(val_captions) # shape: ([batch_size, 512])

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

        # print()
        # print(' --- IMAGE-IMAGE SIMILARITIES --- ')
        # print()

        image_encoder_outputs = clip_model.encode_image(val_imgs) # shape: ([batch_size, 512])

        # normalize features
        image_encoder_outputs = image_encoder_outputs / torch.norm(image_encoder_outputs, dim=1, keepdim=True)

        # cosine similarities between image-image pairs
        image_image_cosine_similarities = image_encoder_outputs @ image_encoder_outputs.t()

        # get median of elements that are not on the diagonal
        mean_image_image_cosine_similarity = image_image_cosine_similarities[~torch.eye(image_image_cosine_similarities.shape[0], dtype=bool)].mean()



        print('mean_image_image_cosine_similarity ', mean_image_image_cosine_similarity)

        '''
        Calculate cosine similarity quality metric
        '''

        # first, scale the cosine similarities by temperature
        mean_cosine_similarity = mean_cosine_similarity * clip_model.temperature
        non_similar_mean_cosine_similarity = non_similar_mean_cosine_similarity * clip_model.temperature
        mean_text_text_cosine_similarity = mean_text_text_cosine_similarity
        mean_image_image_cosine_similarity = mean_image_image_cosine_similarity


        '''
        RSA correlations
        '''

        shuffle_ratio = 0.5 # percentage of texts and images to shuffle

        

        '''
         - Before interchanging
        '''

        text_RSM = text_text_cosine_similarities[torch.tril(torch.ones(text_text_cosine_similarities.shape[0], text_text_cosine_similarities.shape[1]), diagonal=-1).bool()]

        image_RSM = image_image_cosine_similarities[torch.tril(torch.ones(image_image_cosine_similarities.shape[0], image_image_cosine_similarities.shape[1]), diagonal=-1).bool()]

        result = stats.spearmanr(text_RSM.cpu(), image_RSM.cpu())

        print('correlation before interchanging', result.correlation)
        print('p value ', result.pvalue)

        rsa_before_interchanging = result.correlation

        '''
         - After interchanging
        '''

        # get random indices 
        num_elements_to_switch = int(shuffle_ratio * text_encoder_outputs.shape[0])
        random_indices = random.sample(range(text_encoder_outputs.shape[0]), num_elements_to_switch)

        interchanged_text_encoder_outputs = text_encoder_outputs.clone()
        interchanged_image_encoder_outputs = image_encoder_outputs.clone()

        # switch elements at random indices
        interchanged_text_encoder_outputs[random_indices], interchanged_image_encoder_outputs[random_indices] = interchanged_image_encoder_outputs[random_indices], interchanged_text_encoder_outputs[random_indices]

        # cosine similarities between text-text pairs
        interchanged_text_RSM = interchanged_text_encoder_outputs @ interchanged_text_encoder_outputs.t() # shape: ([batch_size, batch_size])
        # get elements in lower traingle excluding diagonal
        interchanged_text_RSM = interchanged_text_RSM[torch.tril(torch.ones(interchanged_text_RSM.shape[0], interchanged_text_RSM.shape[1]), diagonal=-1).bool()]

        # cosine similarities between image-image pairs
        interchanged_image_RSM = interchanged_image_encoder_outputs @ interchanged_image_encoder_outputs.t() # shape: ([batch_size, batch_size])


        # get elements in lower traingle excluding diagonal
        interchanged_image_RSM = interchanged_image_RSM[torch.tril(torch.ones(interchanged_image_RSM.shape[0], interchanged_image_RSM.shape[1]), diagonal=-1).bool()]



        rho_interchanged_image_RSM_image_RSM = stats.spearmanr(interchanged_image_RSM.cpu(), image_RSM.cpu())

        print('correlation between interchanged image RSM and original image RSM', rho_interchanged_image_RSM_image_RSM.correlation)

        rho_interchanged_text_RSM_text_RSM = stats.spearmanr(interchanged_text_RSM.cpu(), text_RSM.cpu())

        print('correlation between interchanged text RSM and original text RSM', rho_interchanged_text_RSM_text_RSM.correlation)


        # result_after_shuffling = stats.spearmanr(interchanged_text_RSM.cpu(), interchanged_image_RSM.cpu())

        image_rsa_after_interchanging = rho_interchanged_image_RSM_image_RSM.correlation

        text_rsa_after_interchanging = rho_interchanged_text_RSM_text_RSM.correlation



        '''
        Compute consistency metric
        '''

        # average_linearity_cosine_sim = evaluate_linearity(clip_model)
        average_consistency_accuracy, average_consistency_cosine_similarity = evaluate_consistency(image_encoder_outputs, text_encoder_outputs)

        print('average_cosistency ', average_consistency_accuracy)

        '''
        dump numbers to csv file
        '''



        if training_hyperparameters['save_losses']:

            csv_name = generate_csv_file_name(clip_model)

            
            # save to csv file
            with open(training_hyperparameters['csv_path'] + f'{csv_name}.csv', 'a') as f:
                # f.write(str(epoch) + ',' + str(index) + ',' + str(val_image_classification_accuracy.item()) + ',' + str(train_image_classification_accuracy.item()) + ',' + str(cosine_sim_metric.item()) + ',' + str(train_loss) + ',' + str(mean_cosine_similarity.item()) + ',' + str(non_similar_mean_cosine_similarity.item()) + ',' + str(mean_text_text_cosine_similarity.item()) + ',' + str(mean_image_image_cosine_similarity.item()) + '\n')
                f.write(str(epoch) + ',' + str(index) + ',' + str(val_image_classification_accuracy.item()) + ',' + str(0) + ',' + str(0) + ',' + str(0) + ',' + str(mean_cosine_similarity.item()) + ',' + str(non_similar_mean_cosine_similarity.item()) + ',' + str(mean_text_text_cosine_similarity.item()) + ',' + str(mean_image_image_cosine_similarity.item()) + '\n')

        '''
        log to wandb
        '''

        average_intra_modality_cosine_sim = (mean_text_text_cosine_similarity.item() + mean_image_image_cosine_similarity.item()) / 2

        if wandb is not None:
            wandb.log(
                data={
                    'val_image_classification_accuracy': val_image_classification_accuracy.item(),
                    'val_image_retrieval_accuracy': val_image_retrieval_accuracy.item(),
                    # 'train_image_accuracy': train_image_classification_accuracy.item(),
                    # 'cosine_sim_metric': cosine_sim_metric.item(),
                    # 'train_intramodality_loss': train_intra_loss,
                    # 'train_intermodality_loss': train_inter_loss,
                    'mean_cosine_similarity': mean_cosine_similarity.item(),
                    'non_similar_mean_cosine_similarity': non_similar_mean_cosine_similarity.item(),
                    'mean_text_text_cosine_similarity': mean_text_text_cosine_similarity.item(),
                    'mean_image_image_cosine_similarity': mean_image_image_cosine_similarity.item(),
                    'average_intra_modality_cosine_similarity': average_intra_modality_cosine_sim,
                    'average_consistency_accuracy': average_consistency_accuracy,
                    'average_consistency_cosine_sim': average_consistency_cosine_similarity,
                    # 'average_linearity_cosine_similarity': average_linearity_cosine_sim,     
                    'rsa_before_interchanging': image_rsa_after_interchanging,
                    'rsa_after_interchanging': text_rsa_after_interchanging       
                    
                },
                # step= int(epoch * (len(dataset_processor.train_dataloader) // training_hyperparameters['batch_size']) + index) # this may not work with WIT dataset, check later
                step= int(epoch * 100 + index) # by 100 to maintain fair comparison with existing runs data

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

            # get predicted captions
            incorrect_predicted_caption_indices = val_image_class_preds[incorrect_preds_mask] # shape: ([n_wrong])

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


def evaluate_consistency(norm_image_embeddings, norm_caption_embeddings):
    '''
    Input embeddings should be normalized
    '''

    num_corrects = 0
    total_count = 0

    n_targets_to_try = 5

    average_cosine_similarity = 0 # this is the final output metric, should be high

    for cap_i, cap_emb in tqdm(enumerate(norm_caption_embeddings)):
        # randomly sample another 
        for i in range(n_targets_to_try):
            source_img_emb = norm_image_embeddings[cap_i]

            target_cap_index = random.randint(0, len(norm_caption_embeddings) - 1)
            target_cap_emb = norm_caption_embeddings[target_cap_index]

            target_img_emb = norm_image_embeddings[target_cap_index]

            cap_direction = target_cap_emb - cap_emb

            target_img_emb_hat = source_img_emb + cap_direction

            # normalize
            norm_target_img_emb_hat = target_img_emb_hat / torch.norm(target_img_emb_hat, dim=-1, keepdim=True)

            # check if closest image to norm_target_img_emb_hat is target_img_emb
            cosine_similarity = norm_image_embeddings @ norm_target_img_emb_hat.t()
            # shape: ([n_images])
            # do softmax
            logits = F.softmax(cosine_similarity, dim=0)
            closest_image_index = torch.argmax(logits)
            # cosine similarity between target_img_emb and closest_image
            closest_image = norm_image_embeddings[closest_image_index]
            closest_image_cosine_similarity = closest_image @ target_img_emb.t()
            average_cosine_similarity += closest_image_cosine_similarity.item()

            if closest_image_index == target_cap_index:
                num_corrects += 1
            
            total_count += 1

    average_cosine_similarity = average_cosine_similarity / total_count
    print('average_cosine_similarity ', average_cosine_similarity)


    return num_corrects / total_count, average_cosine_similarity




def old_evaluate_concept_arrangement(dataset_processor, clip_model, all_subjects, wandb=wandb):
    '''
    - "Pizza on a table" (img) + []"Dog" (txt) - "Pizza" (txt)] = "Dog on a table" (img)
    - Measuring how similar distance and direction between "Pizza on..." (img) and "Dog on..." (img) is compared to that between "Pizza" (txt) and "Dog" (txt)
    '''

    dataset_processor.return_org_imgs_collate_fn = True
    val_dataset = dataset_processor.val_dataset
    collate_fn = dataset_processor.collate_fn

    # create dataloader for validation set
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=training_hyperparameters['validation_batch_size'], collate_fn=collate_fn, generator=torch.Generator().manual_seed(training_hyperparameters['seed']))

    n_subjects_to_try = 5
    pos_to_get = ['NN', 'NNP']

    with torch.no_grad():
        preprocessed_imgs, tokenized_captions, original_imgs, original_captions = next(iter(val_dataloader))

        outputs = clip_model(preprocessed_imgs, tokenized_captions, output_loss=False, return_all=True)

        image_embeds = outputs.image_embeds # shape: ([batch_size, 512])
        text_embeds = outputs.text_embeds # shape: ([batch_size, 512])

        # normalize
        image_embeds = image_embeds / torch.norm(image_embeds, dim=1, keepdim=True)
        text_embeds = text_embeds / torch.norm(text_embeds, dim=1, keepdim=True)

        '''
        - Adding a text concept to image embeddings
        - Doing this n_subjects_to_try times for each image
        '''

        for (image_embed, text_embed, original_img, original_caption) in zip(image_embeds, text_embeds, original_imgs, original_captions):

            pos_tags = pos_tag(word_tokenize(original_caption))

            caption_subject = [word for word, pos in pos_tags if pos in pos_to_get]

            if len(caption_subject) == 0:
                continue

            for i in range(n_subjects_to_try):

                # get random subject
                subject = random.choice(all_subjects)
                subject = subject.strip()
                
                # get subject embedding
                tokenized_subject = clip_model.tokenize_captions([subject])
                subject_embed = clip_model.encode_text(tokenized_subject)
                # normalize subject embedding
                subject_embed = subject_embed / torch.norm(subject_embed, dim=1, keepdim=True)

                # caption subject
                tokenized_caption_subject = clip_model.tokenize_captions([caption_subject[0]])
                caption_subject_embed = clip_model.encode_text(tokenized_caption_subject)
                caption_subject_embed = caption_subject_embed / torch.norm(caption_subject_embed, dim=1, keepdim=True)

                # math with all normalized embeddings
                new_image_embed = image_embed + subject_embed - caption_subject_embed
                # normalize
                new_image_embed = new_image_embed / torch.norm(new_image_embed, dim=1, keepdim=True)

                # generate new caption by replacing caption subject with subject
                new_caption = original_caption.replace(caption_subject[0], subject, 1)
                # this will change when I'm searching whole dataset LATER

                # get closest image to new_caption
                tokenized_new_caption = clip_model.tokenize_captions([new_caption])
                new_caption_embed = clip_model.encode_text(tokenized_new_caption)
                new_caption_embed = new_caption_embed / torch.norm(new_caption_embed, dim=1, keepdim=True)
                # cosine similarity between image_embeds and new_caption_embed
                cosine_similarity = image_embeds @ new_caption_embed.t()
                # get closest image
                closest_image_index = torch.argmax(cosine_similarity)
                closest_image_embed = image_embeds[closest_image_index]
                closest_image = original_imgs[closest_image_index]
                closest_image_caption = original_captions[closest_image_index]

                # cosine similarity between new_image_embed and closest_image_embed
                cosine_similarity = new_image_embed @ closest_image_embed.t()

                sim_new_image_text = closest_image_embed @ new_caption_embed.t()

                sim_new_image_old_image = closest_image_embed @ image_embed.t()

                # this cosine similarity should be high

                # print everything
                print('original caption ', original_caption)
                print('randomly chosen subject ', subject)
                print('caption subject ', caption_subject[0])
                print('new caption ', new_caption)
                print('closest image caption ', closest_image_caption)
                print('cosine similarity ', cosine_similarity.item())
                print('target_cosine_sim', sim_new_image_text.item())
                print('sim between new image embed and same image ', sim_new_image_old_image.item())

                # show images
                fig = plt.figure()
                
                ax = plt.subplot(1, 3, 1)
                plt.imshow(original_img)
                plt.axis("off")

                ax = plt.subplot(1, 3, 2)
                plt.imshow(closest_image)
                plt.axis("off")

                plt.show()

def plot_embeddings(clip_model, dataloader):

    with torch.no_grad():
        for batch in dataloader:
            (imgs, captions) = batch

        outputs = clip_model(imgs, captions, output_loss=False, return_all=True)

        text_embeds = outputs.text_embeds # shape: ([batch_size, 512]), these are normalized
        image_embeds = outputs.image_embeds # normalized also

        pca = PCA(n_components=2)
        plot_captions = text_embeds.detach().cpu().numpy()
        plot_images = image_embeds.detach().cpu().numpy()

        all_embeddings = np.concatenate((plot_captions, plot_images), axis=0)

        pca.fit(all_embeddings)

        pca_coordinates = pca.transform(all_embeddings)
        
        caption_coordinates = pca_coordinates[:plot_captions.shape[0]]
        image_coordinates = pca_coordinates[plot_captions.shape[0]:]

        plt.title(f'PCA of image (red) and text (blue) projections.')
        plt.scatter(caption_coordinates[:, 0], caption_coordinates[:, 1], c='b')
        plt.scatter(image_coordinates[:, 0], image_coordinates[:, 1], c='r')

        # set axes limits
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)

        plt.show()









def evaluate_linearity(clip_model, evaluate_just_text=False, wandb=wandb, plot=False): # evaluate = 'linearity' or 'concept_arrangement'
    '''
    I'll only evaluate on the dataset I prepared and pickled in 'datasets/mscoco/linear_eval_dataset' file
    '''


    # filename = 'datasets/mscoco/linear_eval_dataset/linearity_dataset'
    filename = 'datasets/mscoco/linear_eval_dataset/consistency_dataset'

    average_cosine_similarity = 0 # this is the final output metric, should be high

    average_counter = 0

    plot_captions = []
    plot_images = [] 
    plot_input_image = []
    plot_input_caption = None
    plot_new_images = [] # this is after arithmetic
    plot_target_image = None # this is the image I want to get close to with the new_image_embedding.
    plot_literal_input_caption = None
    plot_literal_target_captions = []
    plot_literal_input_subjects = []
    plot_literal_target_subjects = [] # list of lists
    # not comparing with captions, because they don't contain all info in the image. 

    plot_n = 20

    plot_i = 0

    input_image_index = 5 # 5 was bad

    # load dataset
    with open(filename, 'rb') as fr:
        try:
            while True:
                # data.append(pickle.load(fr))
                input_caption_data = pickle.load(fr) # data for one input caption

                with torch.no_grad():

                    # create custom embedding to retrieve target image
                    input_preprocessed_image = input_caption_data['input_preprocessed_image']
                    input_caption = input_caption_data['input_caption']

                    if plot:
                        print('input caption ', input_caption)
                        print('input caption subjects ', input_caption_data['input_caption_subjects'])

                    # get input image embedding

                    outputs = clip_model(input_preprocessed_image.unsqueeze(0), clip_model.tokenize_captions([input_caption]), output_loss=False, return_all=True)



                    input_image_embedding = outputs.image_embeds.squeeze(0)

                    # normalize
                    norm_input_image_embedding = input_image_embedding / torch.norm(input_image_embedding, dim=-1, keepdim=True)

                    input_caption_embedding = outputs.text_embeds.squeeze(0)

                    # normalize
                    norm_input_caption_embedding = input_caption_embedding / torch.norm(input_caption_embedding, dim=-1, keepdim=True)


                    # input_image_embedding = input_image_embedding.to(clip_model.device)

                    input_subjects = input_caption_data['input_caption_subjects']

                    # embed the input subjects
                    input_subject_embeddings = clip_model.encode_text(clip_model.tokenize_captions(input_subjects))
                    # shape: ([n_subjects, 512])

                    # normalize
                    norm_input_subject_embeddings = input_subject_embeddings / torch.norm(input_subject_embeddings, dim=1, keepdim=True)

                    # calculate cosine similarity between sum of input subjects and input sentence

                    # sum_input_subject_embeddings = torch.sum(norm_input_subject_embeddings, dim=0)

                    # sum_input_subject_embeddings = torch.zeros_like(norm_input_subject_embeddings[0])

                    # for embedding in norm_input_subject_embeddings:
                    # for embedding in input_subject_embeddings:
                    #     sum_input_subject_embeddings += embedding

                    #     # normalize
                    #     # sum_input_subject_embeddings = sum_input_subject_embeddings / torch.norm(sum_input_subject_embeddings, dim=-1, keepdim=True)
                    # # sum_input_subject_embeddings = torch.sum(input_subject_embeddings, dim=0)

                    # # normalize
                    # norm_sum_input_subject_embeddings = sum_input_subject_embeddings / torch.norm(sum_input_subject_embeddings, dim=-1, keepdim=True)

                    # plot_new_images.append(norm_sum_input_subject_embeddings) # this is after arithmetic





                    # cosine_similarity = torch.dot(norm_input_caption_embedding, norm_sum_input_subject_embeddings) # should be high

                    

                    # print('cosine similarity between sum of input subjects and input sentence ', cosine_similarity.item())
                    


                        

                    for i, _ in enumerate(input_caption_data['top_captions']):
                    # for i, _ in enumerate(range(0)):

                        target_preprocessed_image = input_caption_data['top_preprocessed_images'][i]
                        target_preprocessed_image = target_preprocessed_image.to(clip_model.device)
                        # target_tokenized_caption = input_caption_data['top_tokenized_captions'][i]
                        # target_tokenized_caption = target_tokenized_caption.to(clip_model.device)

                        target_caption = input_caption_data['top_captions'][i]

                        outputs = clip_model(target_preprocessed_image.unsqueeze(0), clip_model.tokenize_captions([target_caption]), output_loss=False, return_all=True)


                        # get target image embedding
                        target_image_embedding = outputs.image_embeds.squeeze(0)
                        target_caption_embedding = outputs.text_embeds.squeeze(0)

                        # normalize
                        norm_target_image_embedding = target_image_embedding / torch.norm(target_image_embedding, dim=-1, keepdim=True)
                        norm_target_caption_embedding = target_caption_embedding / torch.norm(target_caption_embedding, dim=-1, keepdim=True)

                        # get target caption subjects
                        target_subjects = input_caption_data['top_subjects'][i]

                        # embed the target subjects
                        target_subject_embeddings = clip_model.encode_text(clip_model.tokenize_captions(target_subjects)) # shape: ([n_subjects, 512])

                        # normalize
                        norm_target_subject_embeddings = target_subject_embeddings / torch.norm(target_subject_embeddings, dim=1, keepdim=True)




                        if evaluate_just_text:
                            # new_caption_embedding = norm_input_caption_embedding.detach().clone()
                            new_embedding = input_caption_embedding.detach().clone()
                        else:
                            # new_image_embedding = norm_input_image_embedding.detach().clone()
                            new_embedding = input_image_embedding.detach().clone()
                            
                        new_embedding = new_embedding.to(clip_model.device)


                        #do the math

                        # for embedding in norm_target_subject_embeddings:
                        for embedding in target_subject_embeddings:
                            new_embedding += embedding 
                            # normalize
                            # new_caption_embedding = new_caption_embedding / torch.norm(new_caption_embedding, dim=-1, keepdim=True)

                        # print('new_caption_embedding after ', new_caption_embedding[:100])
                            
                        # print('new caption embedding before ', new_caption_embedding[:100])
                        # for embedding in norm_input_subject_embeddings:
                        for embedding in input_subject_embeddings:
                            new_embedding -= embedding
                            # normalize
                            # new_caption_embedding = new_caption_embedding / torch.norm(new_caption_embedding, dim=-1, keepdim=True)

                        # plot new caption embedding before and after
                        if plot:
                            # normalize
                            norm_new_embedding = new_embedding / torch.norm(new_embedding, dim=-1, keepdim=True)
                              
                            
                            predicted_target_embedding = new_embedding
                        

                        # normalize
                        norm_predicted_target_embedding = predicted_target_embedding / torch.norm(predicted_target_embedding , dim=-1, keepdim=True)


                        if plot and i < plot_n:
                            plot_images.append(norm_target_image_embedding)
                            plot_captions.append(norm_target_caption_embedding)

                          


                            if plot_i == input_image_index:

                                plot_new_images.append(norm_predicted_target_embedding) # this is after arithmetic
                                plot_literal_target_captions.append(target_caption)
                                plot_literal_target_subjects.append(target_subjects)

                                
                        # cosine similarity between new_image_embedding and target_image_embedding
                        cosine_similarity = torch.dot(norm_predicted_target_embedding, norm_target_image_embedding) # should be high

                        # original_cosine_similarity = torch.dot(target_image_embedding, target_caption_embedding) # is probably high

                        # print magnitudes of embeddings

                        new_embedding_original_image_similarity = torch.dot(norm_predicted_target_embedding, norm_input_image_embedding) # should be low, but default CLIP has this as super high, so it always selects the original image

                        average_cosine_similarity += cosine_similarity.item()
                        average_counter += 1

                        


                        # print()

                        # print('input caption ', input_caption_data['input_caption'])
                        # print('input caption subjects ', input_caption_data['input_caption_subjects'])
                        # print('target caption ', input_caption_data['top_captions'][i])
                        # print('target caption subjects ', input_caption_data['top_subjects'][i])
                        # print('cosine similarity ', cosine_similarity.item())
                        # print('original cosine similarity ', original_cosine_similarity.item())
                        # print('new_embedding_original_image_similarity ', new_embedding_original_image_similarity.item())
                        # print()
                    


                if plot: 



                    if input_image_index == plot_i:

                        if evaluate_just_text:
                            plot_input_image.append(norm_input_caption_embedding)
                        else:
                            plot_input_image.append(norm_input_image_embedding)
                        plot_literal_input_caption = input_caption
                        plot_literal_input_subjects = input_subjects
                    else:
                        plot_images.append(norm_input_image_embedding)
                    plot_captions.append(norm_input_caption_embedding)
                    plot_i += 1

        except EOFError:
            print('done')
            pass


    
    average_cosine_similarity = average_cosine_similarity / average_counter

    if plot:

        # print input caption, input subjects, target captions, and target subjects for each caption

        print(' -- INPUT -- ')
        print('input caption ', plot_literal_input_caption)
        print('input subjects ', plot_literal_input_subjects)
        for i in range(len(plot_literal_target_captions)):
            
            print(f' -- TARGET {i+1}-- ')
            print('target caption ', plot_literal_target_captions[i])
            print('target subjects ', plot_literal_target_subjects[i])
            print()
        # do pca 
        pca = PCA(n_components=2)
        plot_captions = torch.stack(plot_captions).detach().cpu().numpy()
        plot_images = torch.stack(plot_images).detach().cpu().numpy()
        plot_new_images = torch.stack(plot_new_images).detach().cpu().numpy()
        plot_input_image = torch.stack(plot_input_image).detach().cpu().numpy()
        # plot_target_subjects = torch.stack(plot_target_subjects).detach().cpu().numpy()
        # plot_input_subjects = torch.stack(plot_input_subjects).detach().cpu().numpy()

        print('plot new images shape ', plot_new_images.shape)

        all_embeddings = np.concatenate((plot_captions, plot_images, plot_input_image), axis=0)

        # all_embeddings = np.concatenate((plot_captions, plot_images, plot_new_images, plot_input_image), axis=0)
        pca.fit(all_embeddings)

        all_embeddings = np.concatenate((plot_captions, plot_images, plot_new_images, plot_input_image), axis=0)

        

        # get PCA coordinates
        pca_coordinates = pca.transform(all_embeddings)

        # get image and text coordinates
        caption_coordinates = pca_coordinates[:len(plot_captions), :] # shape: [batch_size, 2]
        image_coordinates = pca_coordinates[len(plot_captions):len(plot_captions) + len(plot_images), :] # shape: [batch_size, 2]
        # new_image_coordinates = pca_coordinates[len(plot_captions) + len(plot_images):, :] # shape: [batch_size, 2]
        new_image_coordinates = pca_coordinates[len(plot_captions) + len(plot_images):len(plot_captions) + len(plot_images) + len(plot_new_images), :] # shape: [batch_size, 2]
        input_image_coordinates = pca_coordinates[len(plot_captions) + len(plot_images) + len(plot_new_images):, :] # shape: [batch_size, 2]

        # plot
        plt.title(f'PCA of image (red) and text (blue) projections. Transformed image coordinates (green). Input image coordinates (yellow).')
        plt.scatter(caption_coordinates[:, 0], caption_coordinates[:, 1], c='b')
        plt.scatter(image_coordinates[:, 0], image_coordinates[:, 1], c='r')

        for i, coord in enumerate(new_image_coordinates):
            # plt.scatter(caption_coordinates[i, 0], caption_coordinates[i, 1], c='b', marker=f'${i}$', s=100)
            plt.scatter(coord[0], coord[1], c='g', marker=f'${len(plot_literal_target_subjects[i])}$')
            # plt.scatter(coord[0], coord[1], c='g', marker=f'${i}$', s=100)
            # plt.annotate(plot_target_literal_captions[i], (coord[0], coord[1]))
        # plt.scatter(new_image_coordinates[:, 0], new_image_coordinates[:, 1], c='g', marker=f'${len(plot_target_subjects) - len(plot_input_subjects)}$')
        plt.scatter(input_image_coordinates[:, 0], input_image_coordinates[:, 1], c='y')

        plt.show()



    return average_cosine_similarity



def generate_csv_file_name(clip_model):
    # create csv file name
    name_template = training_hyperparameters['loss_file_name_template']

    name_parts = name_template.split('_')

    csv_name = ''

    temperature = training_hyperparameters['temperature']
    intra_modality_temperature = training_hyperparameters['intra_modality_temperature']

    seed = training_hyperparameters['seed']

    for i, part in enumerate(name_parts):
        # replace temp with temperature
        if 'temp' in part:

            new_part = part.replace('temp', str(temperature))

            if training_hyperparameters['intra_modality_loss']:
                new_part += '_' + str(intra_modality_temperature)



        elif 'name' in part:
            new_part = part.replace('name', str(selected_clip_model.value))
        elif 'iweight' in part:
            new_part = part.replace('iweight', str(training_hyperparameters['loss_weights']['image_to_text_weight']))
        elif 'tweight' in part:
            new_part = part.replace('tweight', str(training_hyperparameters['loss_weights']['text_to_image_weight']))
        elif 'loss' in part:

            if training_hyperparameters['intra_modality_loss']:
                new_part = part.replace('loss', 'Lit_ii_tt')
            else:
                new_part = part.replace('loss', 'Lit')
        elif 'seed' in part:
            new_part = part.replace('seed', str(seed))
        else:
            new_part = part


        if i == len(name_parts) - 1:
            csv_name += new_part
        else:
            csv_name += new_part + '_'

    return csv_name


def init_stats_csv_file(clip_model):
    '''
    Initialize csv file for storing stats
    '''

    csv_name = generate_csv_file_name(clip_model)
    if training_hyperparameters['save_losses']:
        # save to csv file
        with open(training_hyperparameters['csv_path'] + f'{csv_name}.csv', 'w') as f:
            # write the training hyperparameters 
            f.write('training_hyperparameters\n')
            for key, value in training_hyperparameters.items():
                f.write(f'{key}: {value}\n')
            f.write('\n')
            f.write('epoch,index,val_image_accuracy,train_image_accuracy, cosine_similarity_metric, train_loss, mean_cosine_similarity,non_similar_mean_cosine_similarity,mean_text_text_cosine_similarity,mean_image_image_cosine_similarity\n')

def get_checkpoint_path():
    '''
    Get path of model to load
    '''

    return 'checkpoints/' + generate_csv_file_name(None) + '.pt'
    if selected_clip_model == ClipModels.FINETUNED_TEMP:
        return 'checkpoints/my_clip_checkpoint_finetuned_temp.pt'
    elif selected_clip_model == ClipModels.FINETUNED:
        return 'checkpoints/my_clip_checkpoint_finetuned.pt'
    elif selected_clip_model == ClipModels.DEFAULT:
        return 'checkpoints/my_clip_checkpoint_default.pt'
    elif selected_clip_model == ClipModels.WARM:
        return 'checkpoints/my_clip_checkpoint_warm.pt'
    
    # training_hyperparameters['model_path'] = 'checkpoints/my_clip_checkpoint_' + "_".join(selected_clip_model.value.split("_")[1:]) + '.pt'


def write_pca_plots_to_file(image_projections, text_projections, index, output_dir):
    '''
    write PCA plot coordinates of image and text projections, AFTER the linear projection, to file in output_dir
    image projections shape: [batch_size, 512]
    text projections shape: [batch_size, 512]
    '''

    global pca

    # stack image and text projections
    stacked_projections = torch.cat((image_projections, text_projections), dim=0) # shape: [2*batch_size, 512]

    if pca is None:
        # get PCA
        pca = PCA(n_components=2)
        pca.fit(stacked_projections.cpu().numpy())


    # # get PCA
    # pca = PCA(n_components=2)
    # pca.fit(stacked_projections.cpu().numpy())

    # get PCA coordinates
    pca_coordinates = pca.transform(stacked_projections.cpu().numpy()) # shape: [2*batch_size, 2]

    # get image and text coordinates
    image_coordinates = pca_coordinates[:image_projections.shape[0], :] # shape: [batch_size, 2]
    text_coordinates = pca_coordinates[image_projections.shape[0]:, :] # shape: [batch_size, 2]

    # write to file
    np.save(output_dir + 'image_coordinates_' + str(index) + '.npy', image_coordinates)
    np.save(output_dir + 'text_coordinates_' + str(index) + '.npy', text_coordinates)





def plot_pca_from_file(image_coordinates_file, text_coordinates_file):
    '''
    Plot PCA of image and text projections, AFTER the linear projection
    image projections shape: [batch_size, 512]
    text projections shape: [batch_size, 512]
    '''

    # get image and text coordinates
    image_coordinates = np.load(image_coordinates_file) # shape: [batch_size, 2]
    text_coordinates = np.load(text_coordinates_file) # shape: [batch_size, 2]

    # 

    # plot
    plt.title(f'PCA of image (red) and text (blue) projections, after {image_coordinates_file.split("_")[3].split(".")[0]} pass(es)')
    plt.scatter(image_coordinates[:, 0], image_coordinates[:, 1], c='r')
    plt.scatter(text_coordinates[:, 0], text_coordinates[:, 1], c='b')
    plt.show()
    

def plot_pca_subplots_from_file(dir, start, stop, step):

    # calculate number of subplots
    num_subplots = (stop - start) // step

    # create subplots
    fig, axs = plt.subplots(1, num_subplots)

    # set title
    fig.suptitle('stacked subplots of image (red) and text (blue) projections')

    for i in range(start, stop, step):
        # get image and text coordinates
        image_coordinates = np.load(dir + 'image_coordinates_' + str(i) + '.npy') # shape: [batch_size, 2]
        text_coordinates = np.load(dir + 'text_coordinates_' + str(i) + '.npy') # shape: [batch_size, 2]

        # keep axes fixed for all subplots
        axs[i//step].set_xlim(-0.75, 0.75)
        axs[i//step].set_ylim(-0.75, 0.75)

        # plot
        axs[i//step].set_title(f'after {i} pass(es)')
        axs[i//step].scatter(image_coordinates[:, 0], image_coordinates[:, 1], c='r')
        axs[i//step].scatter(text_coordinates[:, 0], text_coordinates[:, 1], c='b')
    
    plt.show()


