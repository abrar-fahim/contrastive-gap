import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from evaluate import load as load_evaluator
from src.config import *
import wandb


pca = None




# def do_validation(val_dataset, train_dataset, clip_model, index=0, epoch=0, captioning_model=False):
def do_validation(dataset_processor, clip_model, index=0, epoch=0, captioning_model=False, wandb=wandb):

    
    '''
    Report accuracy and median cosine similarity on validation set
    Report text-text and image-image cosine similarities
    Dump numbers to csv file
    '''

    # create seperate dataloaders for val and train dataset, seperate from the ones used in training, so that I get same train and val batch each time this runs

    val_dataset = dataset_processor.val_dataset
    train_dataset = dataset_processor.train_dataset
    collate_fn = dataset_processor.collate_fn



    # create dataloader for validation set
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=training_hyperparameters['validation_batch_size'], collate_fn=collate_fn, generator=torch.Generator().manual_seed(42))

    # create dataloader for train set
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=training_hyperparameters['validation_batch_size'], collate_fn=collate_fn, generator=torch.Generator().manual_seed(42))

    with torch.no_grad():
        # get batch from validation set
        (val_imgs, val_captions) = next(iter(val_dataloader))

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


        val_outputs = clip_model(val_imgs, val_captions, output_loss=False, return_all=True) # so that I get cosine similarities directly

        '''
        1. Validation image classification accuracy
        '''
        val_logits_per_image = val_outputs.logits_per_image # shape of both: ([64, 64])


        # softmax on logits_per_image
        val_image_class_probs = F.softmax(val_logits_per_image, dim=-1) # shape: ([64, 64])

       
        # calculate accuracy
        # get indices of max values
        val_image_class_preds = val_image_class_probs.argmax(dim=-1) # shape: ([64])

        
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



        '''
        3. Training image classification accuracy
        '''

        (train_imgs, train_captions) = next(iter(train_dataloader))

        train_outputs = clip_model(train_imgs, train_captions, output_loss=True, return_all=True, output_intra_modality_loss=True) # so tha I get cosine similarities directly
        train_logits_per_image = train_outputs.logits_per_image # shape of both: ([64, 64])
        train_image_class_probs = F.softmax(train_logits_per_image, dim=-1) # shape: ([64, 64])
        train_image_class_preds = train_image_class_probs.argmax(dim=-1) # shape: ([64])
        train_image_class_labels = torch.arange(train_image_class_probs.shape[0], device=train_image_class_probs.device) # shape: ([64])
        train_image_classification_accuracy = (train_image_class_preds == train_image_class_labels).float().mean()

        # train_loss = train_outputs.loss.item()

        train_intra_loss = train_outputs.loss['intra_modality']
        train_inter_loss = train_outputs.loss['inter_modality']
        train_loss = train_outputs.loss['total']



        

        print('--- ACCURACY STUFF --- ')

        # print('image preds ', image_class_preds)
        # print('image labels ', image_class_labels)

        print('validation image_accuracy ', val_image_classification_accuracy.item())
        print('train image_accuracy ', train_image_classification_accuracy.item())


        print('--- IMAGE-TEXT SIMILARITIES --- ')



        # print('logits_per_image ', logits_per_image)

        # print logits per image for first 5 images
        # print('logits_per_image ', logits_per_image[:5, :5])
        cosine_similarities = val_logits_per_image.diag() # shape: [64]
        # get median cosine similarity
        median_cosine_similarity = torch.median(cosine_similarities)
        print('median cosine similarity ', median_cosine_similarity.item())


        # Get 2nd highest cosine similarity for each image
        top2_cosine_similarities = torch.topk(val_logits_per_image, k=2, dim=-1).values # shape: [batch_size, 2]
        # print('top2_cosine_similarities ', top2_cosine_similarities.shape)
        # get median of 2nd highest cosine similarity for each image
        median_top2_cosine_similarity = torch.median(top2_cosine_similarities[:, 1])

        # print('median_top2_cosine_similarity ', median_top2_cosine_similarity)

        # get median of elements that are not on the diagonal
        non_similar_median_cosine_similarity = val_logits_per_image[~torch.eye(val_logits_per_image.shape[0], dtype=bool)].median()
        # print('non_similar_median_cosine_similarity ', non_similar_median_cosine_similarity)

        # print temperature
        # print('clip_model.logit_scale ', clip_model.model.logit_scale)

        '''
        Check if model predictions are exploding
        (Do this check without the temperature param)
        '''

        # doing it just for images for now
        # image_embeds = outputs.vision_model_output.pooler_output # shape: ([batch_size, 512]), these are before linear projection
        # image_embeds = val_outputs.image_embeds # shape: ([batch_size, 512]), these are after linear projection




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
        median_text_text_cosine_similarity = text_text_cosine_similarities[torch.triu(torch.ones(text_text_cosine_similarities.shape[0], text_text_cosine_similarities.shape[1]), diagonal=1).bool()].median()

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
        median_image_image_cosine_similarity = image_image_cosine_similarities[~torch.eye(image_image_cosine_similarities.shape[0], dtype=bool)].median()

        # print('median_image_image_cosine_similarity ', median_image_image_cosine_similarity)

        '''
        Calculate cosine similarity quality metric
        '''

        # first, scale the cosine similarities by temperature
        median_cosine_similarity = median_cosine_similarity * clip_model.temperature
        non_similar_median_cosine_similarity = non_similar_median_cosine_similarity * clip_model.temperature
        median_text_text_cosine_similarity = median_text_text_cosine_similarity
        median_image_image_cosine_similarity = median_image_image_cosine_similarity
        

        cosine_sim_metric = (median_cosine_similarity+1) / (((non_similar_median_cosine_similarity+1) ** 2 * (median_text_text_cosine_similarity+1) * (median_image_image_cosine_similarity+1)) + (median_cosine_similarity+1))
        # adding +1 to handle negative cosine sims more easily
        # adding median cosine sim in denominator prevent divide by zero
        # seems like it varies from 0 (worst) to 1 (best) for now.

        print()
        print('cosine_sim_metric ', cosine_sim_metric.item())
        print()

        '''
        dump numbers to csv file
        '''



        if training_hyperparameters['save_losses']:

            csv_name = generate_csv_file_name(clip_model)

            
            # save to csv file
            with open(training_hyperparameters['csv_path'] + f'{csv_name}.csv', 'a') as f:
                f.write(str(epoch) + ',' + str(index) + ',' + str(val_image_classification_accuracy.item()) + ',' + str(train_image_classification_accuracy.item()) + ',' + str(cosine_sim_metric.item()) + ',' + str(train_loss) + ',' + str(median_cosine_similarity.item()) + ',' + str(non_similar_median_cosine_similarity.item()) + ',' + str(median_text_text_cosine_similarity.item()) + ',' + str(median_image_image_cosine_similarity.item()) + '\n')

        '''
        log to wandb
        '''

        average_intra_modality_cosine_sim = (median_text_text_cosine_similarity.item() + median_image_image_cosine_similarity.item()) / 2

        if wandb is not None:
            wandb.log(
                data={
                    'val_image_classification_accuracy': val_image_classification_accuracy.item(),
                    'val_image_retrieval_accuracy': val_image_retrieval_accuracy.item(),
                    'train_image_accuracy': train_image_classification_accuracy.item(),
                    'cosine_sim_metric': cosine_sim_metric.item(),
                    'train_intramodality_loss': train_intra_loss,
                    'train_intermodality_loss': train_inter_loss,
                    'median_cosine_similarity': median_cosine_similarity.item(),
                    'non_similar_median_cosine_similarity': non_similar_median_cosine_similarity.item(),
                    'median_text_text_cosine_similarity': median_text_text_cosine_similarity.item(),
                    'median_image_image_cosine_similarity': median_image_image_cosine_similarity.item(),
                    'average_intra_modality_cosine_similairity': average_intra_modality_cosine_sim
                    
                },
                # step= int(epoch * (len(dataset_processor.train_dataloader) // training_hyperparameters['batch_size']) + index) # this may not work with WIT dataset, check later
                step= int(epoch * 100 + index) # by 100 to maintain fair comparison with existing runs data

            )




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

def generate_csv_file_name(clip_model):
    # create csv file name
    name_template = training_hyperparameters['loss_file_name_template']

    name_parts = name_template.split('_')

    csv_name = ''

    for i, part in enumerate(name_parts):
        # replace temp with temperature
        if 'temp' in part:
            new_part = part.replace('temp', str(clip_model.temperature))

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
            f.write('epoch,index,val_image_accuracy,train_image_accuracy, cosine_similarity_metric, train_loss, median_cosine_similarity,non_similar_median_cosine_similarity,median_text_text_cosine_similarity,median_image_image_cosine_similarity\n')    

def get_checkpoint_path():
    '''
    Get path of model to load
    '''
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


