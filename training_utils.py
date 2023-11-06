import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from clip_caption_predict import Predictor as MLPPredictor
from clip_caption_transformer_predict import Predictor as TransformerPredictor
from evaluate import load as load_evaluator
from config import selected_clip_model, ClipModels, ClipCaptionModelMapping, clip_caption_model_weight_paths, clip_caption_model_train_hyperparameters


pca = None




def do_validation(val_dataloader, clip_model, index=0, captioning_model=False):
    with torch.no_grad():
        
        # get batch from validation set
        (val_imgs, val_captions) = next(iter(val_dataloader))

        # print('val caps ', val_captions[:15])

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


        outputs = clip_model(val_imgs, val_captions, output_loss=False, return_all=True) # so tha I get cosine similarities directly
        logits_per_image = outputs.logits_per_image # shape of both: ([64, 64])

        logits_per_text = outputs.logits_per_text # shape of both: ([64, 64])

        # softmax on logits_per_image
        image_class_probs = F.softmax(logits_per_image, dim=-1) # shape: ([64, 64])

        # print('image_class_probs ', image_class_probs)

        

        # calculate accuracy
        # get indices of max values
        image_class_preds = image_class_probs.argmax(dim=-1) # shape: ([64])

        
        # get indices of correct predictions
        image_class_labels = torch.arange(image_class_probs.shape[0], device=image_class_probs.device) # shape: ([64])

        # calculate accuracy
        image_accuracy = (image_class_preds == image_class_labels).float().mean()

        print('--- ACCURACY STUFF --- ')

        # print('image preds ', image_class_preds)
        # print('image labels ', image_class_labels)

        print('image_accuracy ', image_accuracy)

        print('--- IMAGE-TEXT SIMILARITIES --- ')



        # print('logits_per_image ', logits_per_image)

        # print logits per image for first 5 images
        # print('logits_per_image ', logits_per_image[:5, :5])
        cosine_similarities = logits_per_image.diag() # shape: [64]
        # get median cosine similarity
        median_cosine_similarity = torch.median(cosine_similarities)
        print('median cosine similarity ', median_cosine_similarity)


        # Get 2nd highest cosine similarity for each image
        top2_cosine_similarities = torch.topk(logits_per_image, k=2, dim=-1).values # shape: [batch_size, 2]
        # print('top2_cosine_similarities ', top2_cosine_similarities.shape)
        # get median of 2nd highest cosine similarity for each image
        median_top2_cosine_similarity = torch.median(top2_cosine_similarities[:, 1])

        print('median_top2_cosine_similarity ', median_top2_cosine_similarity)

        # get median of elements that are not on the diagonal
        non_similar_median_cosine_similarity = logits_per_image[~torch.eye(logits_per_image.shape[0], dtype=bool)].median()
        print('non_similar_median_cosine_similarity ', non_similar_median_cosine_similarity)

        # print temperature
        print('clip_model.logit_scale ', clip_model.model.logit_scale)

        '''
        Check if model predictions are exploding
        (Do this check without the temperature param)
        '''

        # doing it just for images for now
        # image_embeds = outputs.vision_model_output.pooler_output # shape: ([batch_size, 512]), these are before linear projection
        image_embeds = outputs.image_embeds # shape: ([batch_size, 512]), these are after linear projection




        '''
        - Get text-text similarities
        '''
        print()
        print(' --- TEXT-TEXT SIMILARITIES --- ')
        print()

        text_encoder_outputs = clip_model.encode_text(val_captions) # shape: ([batch_size, 512])

        # normalize features
        text_encoder_outputs = text_encoder_outputs / torch.norm(text_encoder_outputs, dim=1, keepdim=True)

        # cosine similarities between text-text pairs
        text_text_cosine_similarities = text_encoder_outputs @ text_encoder_outputs.t() # shape: ([batch_size, batch_size])

        # get median of elements that are in the upper triangle (excluding diagonal!!)
        median_text_text_cosine_similarity = text_text_cosine_similarities[torch.triu(torch.ones(text_text_cosine_similarities.shape[0], text_text_cosine_similarities.shape[1]), diagonal=1).bool()].median()

        print('median_text_text_cosine_similarity ', median_text_text_cosine_similarity)

        '''
        - Get image-image similarities
        '''

        print()
        print(' --- IMAGE-IMAGE SIMILARITIES --- ')
        print()

        image_encoder_outputs = clip_model.encode_image(val_imgs) # shape: ([batch_size, 512])

        # normalize features
        image_encoder_outputs = image_encoder_outputs / torch.norm(image_encoder_outputs, dim=1, keepdim=True)

        # cosine similarities between image-image pairs
        image_image_cosine_similarities = image_encoder_outputs @ image_encoder_outputs.t()

        # get median of elements that are not on the diagonal
        median_image_image_cosine_similarity = image_image_cosine_similarities[~torch.eye(image_image_cosine_similarities.shape[0], dtype=bool)].median()

        print('median_image_image_cosine_similarity ', median_image_image_cosine_similarity)


        '''
        evaluating captioning model
        '''


        if captioning_model:
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
    if clip_caption_model_train_hyperparameters['show_real_images']:
        # return (torch.stack(imgs), captions)
        return (imgs, captions)     
    return (torch.stack(imgs), captions)

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


