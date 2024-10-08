import torch
import numpy as np
from clips.clip_parent import ClipParent
from transformers import CLIPModel, AutoTokenizer, CLIPConfig, CLIPTextConfig, CLIPTextModelWithProjection, GPT2Tokenizer
from src.utils import get_checkpoint_path
from torch.functional import F

from transformers.models.clip.modeling_clip import CLIPOutput

from src.config import *
import os
from clips.image_encoder import ImageEncoder
from clips.text_encoder import TextEncoder
from clips.encoder import Encoder
from clips.projection_layer import ProjectionLayer
from collections import OrderedDict
from typing import Any, Optional, Tuple, Union
from dataclasses import dataclass
from src.my_ce_loss import MyCrossEntropyLoss, MyCEAlignmentLoss

@dataclass
class HFClipOutput(OrderedDict):

    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`CLIPTextModel`].
        image_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`CLIPVisionModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPTextModel`].
        vision_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPVisionModel`].
    """


    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    encoder1_hidden_states: Tuple[torch.FloatTensor] = None
    encoder2_hidden_states: Tuple[torch.FloatTensor] = None
    encoder1_input_ids: torch.LongTensor = None
    encoder2_input_ids: torch.LongTensor = None



 

class HFClip(ClipParent):


    def __init__(self, encoder1: Encoder, encoder2: Encoder, common_projection_layer: ProjectionLayer = None):
        super().__init__()
        self.device = torch.device(config_cuda_device if torch.cuda.is_available() else "cpu")


        '''
        Set config variables to self
        '''

        self.encoder1_modality = wandb.config['encoder1_modality']
        self.encoder2_modality = wandb.config
        self.same_inputs = wandb.config['same_inputs']
        self.same_encoder = wandb.config['same_encoder']
        self.second_caption_offset = wandb.config['second_caption_offset']
        self.one_encoder = wandb.config['one_encoder']
        self.mismatched_pairs = wandb.config['mismatched_pairs']

        self.common_projection_layer = common_projection_layer
        self.shared_transformer_layers = wandb.config['shared_transformer_layers']

        

        '''
        Set encoders
        1 is image (or text if text_only is True)
        2 is text
        '''

        self.encoder1 = encoder1
        self.encoder2 = encoder2

        print('CLIP device ', self.device)
    
        self.temperature: int = wandb.config['temperature']
        self.set_temperature()

        '''
        load CLIP from respective checkpoint regardless of training mode
        clip training toy and training loop will handle loading from scratch or loading from checkpoint
        '''

        checkpoint_path = get_checkpoint_path()

        print('check point path for CLIP model ', checkpoint_path)

        loaded_checkpoint = None

        # check if checkpoint path exists
        if os.path.exists(checkpoint_path) and wandb.config['continue_from_checkpoint']:
            loaded_checkpoint = torch.load(checkpoint_path, map_location=self.device)

        
            # this only makes sense if we're loading from a checkpoint
            if not selected_clip_model == ClipModels.DEFAULT.value:
                self.load_state_dict(loaded_checkpoint['model_state_dict'])
                print('loaded clip model from checkpoint ', checkpoint_path)

            else:
                print('CLIP model not loaded from checkpoint')

        else:
            print('CLIP model not loaded from checkpoint')

        # if path doesnt exist, it means we're starting from pretrained model anyway

        # self.loss = torch.nn.CrossEntropyLoss()

        self.loss = MyCrossEntropyLoss()
        # self.loss = MyCEAlignmentLoss()



        print()
        print('--- HF CLIP MODEL ---')
        print()

        print('selected clip model ', selected_clip_model.name)


        if loaded_checkpoint == None:
            # temperature will NOT be close to logit scale temp if we're loading from checkpoint
            assert np.isclose(self.temperature, 1 / self.logit_scale.exp().item())

        # print('logit scale: ', self.model.logit_scale)
        print('temperature (T): ', self.temperature)

        # no need to load state dict for default, since it's the same as the pretrained model




    def set_temperature(self):
        

        self.temperature = wandb.config['temperature']
        self.intra_modality_temperature = wandb.config['intra_modality_temperature']

        self.intra_modality_logit_scale = torch.nn.Parameter(torch.tensor(np.log(1 / self.intra_modality_temperature), requires_grad=False, device=self.device)) # not self.model since clip_model doesn't have intra_modality_logit_scale

        self.intra_modality_logit_scale.requires_grad = False
        
    
        self.logit_scale = torch.nn.Parameter(torch.tensor(np.log(1 / self.temperature), requires_grad=True, device=self.device))

        if wandb.config['learnable_temperature']:
            self.logit_scale.requires_grad = True
        else:
            self.logit_scale.requires_grad = False
      
        
        self.to(self.device)

    def get_temperature(self) -> float:
        # calculate current temp from logit scale
        return 1 / self.logit_scale.exp().item()

    def setW(self, W: torch.FloatTensor):

        # assert isinstance(self.encoder2, TextEncoder), 'Encoder 2 is not text encoder'

        self.encoder2.setW(W)





    def get_image_encoder(self):
        if isinstance(self.encoder1, ImageEncoder):
            return self.encoder1
        elif isinstance(self.encoder2, ImageEncoder):
            return self.encoder2
        else:
            raise ValueError('No image encoder found')
        
    def get_text_encoder(self):
        if isinstance(self.encoder1, TextEncoder):
            return self.encoder1
        elif isinstance(self.encoder2, TextEncoder):
            return self.encoder2
        else:
            raise ValueError('No text encoder found')
        

    def encode_image(self, images, output_dict=False):
        '''
        Find which encoder is image
        '''

        if isinstance(self.encoder1, ImageEncoder) and isinstance(self.encoder2, ImageEncoder):
            raise ValueError('Ambigious! Both encoders are image encoders')
        
        if self.one_encoder:
            return self.encoder1(images, output_dict=output_dict)

        if isinstance(self.encoder1, ImageEncoder):
            image_encoder = self.encoder1
        elif isinstance(self.encoder2, ImageEncoder):
            image_encoder = self.encoder2
        else:
            raise ValueError('No image encoder found')

        # preprocessed_images = image_encoder.preprocess_images(images)

        image_features = image_encoder(images, output_dict=output_dict)

        # return pooled_output AFTER projection
        return image_features

    
    def encode_text(self, tokenized_captions: torch.Tensor, output_dict=False):
        '''
        Returns pooled_output AFTER projection

        captions = tokenized captions
        '''



        if isinstance(self.encoder1, TextEncoder) and isinstance(self.encoder2, TextEncoder):
            raise ValueError('Ambigious! Both encoders are text encoders')

        if self.one_encoder:

            # tokenized_captions = self.encoder1.tokenize_captions(captions)
            return self.encoder1(tokenized_captions, output_dict=output_dict, input_tokenized_captions=True)

        # find which encoder is text
        if isinstance(self.encoder1, TextEncoder):
            text_encoder = self.encoder1
        elif isinstance(self.encoder2, TextEncoder):
            text_encoder = self.encoder2
        else:
            raise ValueError('No text encoder found')




        # tokenized_captions = text_encoder.tokenize_captions(captions)
        text_features = text_encoder(tokenized_captions, output_dict=output_dict, input_tokenized_captions=True)

        return text_features
    

    def encoder1_features(self, inputs):

        return self.encoder1(inputs, output_dict=True)
    


    def encoder2_features(self, inputs):

        if self.one_encoder:
            return self.encoder1(inputs, output_dict=True)

        return self.encoder2(inputs, output_dict=True)

    
    def pool_hidden_states(self, hidden_state: torch.FloatTensor):
        '''
        `hidden_state` is `torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`
        `LayerNorm` without trainable params before pooling
        Need to normalize since we'll eventually compute cosine similarity, but maybe dont do that here?
        '''


    def clamp_logit_scale(self):
        self.logit_scale.data.clamp_(min=0, max=np.log(100))


    def forward(self, encoder1_inputs, encoder2_inputs, output_loss=True, return_all=False, output_intra_modality_loss=False, output_hidden_states=False):

        
        '''
        outputs = HFClipOutput(
            loss=loss,
            logits_per_image = logits_per_encoder1_embeds,
            logits_per_text = logits_per_encoder2_embeds,
            text_embeds = normalized_encoder2_embeds,
            image_embeds = normalized_encoder1_embeds,
            encoder1_hidden_states=encoder1_hidden_states,
            encoder2_hidden_states=encoder2_hidden_states,
            encoder1_input_ids = encoder1_input_ids,
            encoder2_input_ids = encoder2_input_ids
        )
        '''

        encoder1_outputs = self.encoder1(encoder1_inputs, output_hidden_states, output_dict=True)

        encoder1_hidden_states  = encoder1_outputs['hidden_states']
        encoder1_input_ids = encoder1_outputs['input_ids']
        encoder1_outputs = encoder1_outputs['embeds']

        # print('encoder 1 outputs ', encoder1_outputs)


        if self.common_projection_layer:
            encoder1_outputs = self.common_projection_layer(encoder1_outputs)

        assert encoder1_outputs.shape[1] == wandb.config['clip_projection_dim'], f'encoder1 output shape != clip_projection_dim {encoder1_outputs.shape[1]} != {wandb.config["clip_projection_dim"]}'

        


        if self.one_encoder:
            encoder2_outputs = self.encoder1(encoder2_inputs, output_hidden_states, output_dict=True)
        else:
            encoder2_outputs = self.encoder2(encoder2_inputs, output_hidden_states, output_dict=True)

        encoder2_hidden_states  = encoder2_outputs['hidden_states']
        encoder2_input_ids = encoder2_outputs['input_ids']
        encoder2_outputs = encoder2_outputs['embeds']

        # print('encoder 1 outputs ', encoder1_outputs)

        if self.common_projection_layer:
            encoder2_outputs = self.common_projection_layer(encoder2_outputs)

        assert encoder2_outputs.shape[1] == wandb.config['clip_projection_dim'], 'encoder2 output shape is not clip_projection_dim'

        # normalize features
        normalized_encoder1_embeds = encoder1_outputs / encoder1_outputs.norm(p=2, dim=-1, keepdim=True)
        normalized_encoder2_embeds = encoder2_outputs / encoder2_outputs.norm(p=2, dim=-1, keepdim=True)


        # print('asserting')

        # check if embeds are normalized as expected
        # assert torch.allclose(normalized_encoder1_embeds.norm(p=2, dim=-1), torch.tensor(1.0).to(self.device)), 'encoder1 embeds are not normalized'
        # assert torch.allclose(normalized_encoder2_embeds.norm(p=2, dim=-1), torch.tensor(1.0).to(self.device)), 'encoder2 embeds are not normalized'
        
        if self.same_inputs and not self.mismatched_pairs:
              

            if self.encoder1_modality == 'text':
                assert (encoder1_inputs == encoder2_inputs), 'inputs are not same'

            elif self.encoder1_modality == 'image':
                assert torch.eq(encoder1_inputs, encoder2_inputs).all(), 'inputs are not same'

        if self.same_encoder and self.same_inputs and not self.second_caption_offset:

            assert torch.eq(normalized_encoder1_embeds, normalized_encoder2_embeds).all(), 'embeddings are not same'

        # print('asserting done')



        logits_per_encoder1_embeds = normalized_encoder1_embeds @ normalized_encoder2_embeds.t() * self.logit_scale.exp() # logit_scale.exp() is 1 / temperature, so 100 for 0.01
        logits_per_encoder2_embeds = normalized_encoder2_embeds @ normalized_encoder1_embeds.t() * self.logit_scale.exp()

        # print('logits per encoder 1 embeds ', logits_per_encoder1_embeds)

        labels = torch.arange(normalized_encoder1_embeds.shape[0]).to(self.device)

        encoder1_weight = wandb.config['i_t_loss_weight']
        encoder2_weight = wandb.config['t_i_loss_weight']

        loss = 0 

        if output_loss == True:

            if wandb.config['uniformity_loss'] or output_intra_modality_loss or wandb.config['cross_uniformity_loss'] or wandb.config['alignment_loss']: # always output uniformity loss, when evaluating
                # uniformity loss

                if not wandb.config['use_train_as_val']:
                    encoder1_sq_pdist = torch.pdist(normalized_encoder1_embeds, p=2).pow(2)
                    encoder2_sq_pdist = torch.pdist(normalized_encoder2_embeds, p=2).pow(2)

                    encoder1_uniformity_loss = encoder1_sq_pdist.mul(-2).exp().mean().log()
                    encoder2_uniformity_loss = encoder2_sq_pdist.mul(-2).exp().mean().log()

                    uniformity_loss = 0.5 * encoder1_uniformity_loss + 0.5 * encoder2_uniformity_loss

                    off_diagonal_ones = torch.ones((len(normalized_encoder1_embeds), len(normalized_encoder2_embeds))).to(self.device).tril(diagonal = -1) 

                    off_diagonal_ones += torch.ones((len(normalized_encoder2_embeds), len(normalized_encoder1_embeds))).to(self.device).triu(diagonal = 1)

                    lower_diagonal_ones = torch.ones((len(normalized_encoder1_embeds), len(normalized_encoder2_embeds))).to(self.device).tril(diagonal = -1)

                    off_diagonal_ones = off_diagonal_ones.to(self.device)

                    cross_encoder_uniform_loss  = torch.masked_select(torch.cdist(normalized_encoder1_embeds.unsqueeze(0), normalized_encoder2_embeds.unsqueeze(0))[0], off_diagonal_ones == 1).square().mul(-2).exp().mean().log()

                    uniform_cyclic_loss = (encoder1_sq_pdist - encoder2_sq_pdist).square().mean()



                    intra_image_directions = normalized_encoder1_embeds.unsqueeze(1) - normalized_encoder1_embeds

                    intra_text_directions = normalized_encoder2_embeds.unsqueeze(1) - normalized_encoder2_embeds

                    # TAKE ONLY lower triangular part LATER

                    cyclic_direction_loss = (intra_image_directions - intra_text_directions).square().mean()

                    alignment_loss = (normalized_encoder1_embeds - normalized_encoder2_embeds).norm(dim=1).pow(2).mean()




                    batch_size = normalized_encoder1_embeds.shape[0]


                    logits_encoder1_per_encoder1_embeds = normalized_encoder1_embeds @ normalized_encoder1_embeds.t() * self.logit_scale.exp()
                    logits_encoder2_per_encoder2_embeds = normalized_encoder2_embeds @ normalized_encoder2_embeds.t() * self.logit_scale.exp()


                    inmodal_cyclic_loss = (logits_encoder1_per_encoder1_embeds - logits_encoder2_per_encoder2_embeds).square().mean() / (self.logit_scale.exp() * self.logit_scale.exp()) * batch_size
                    

                    crossmodal_cyclic_loss = (logits_per_encoder1_embeds - logits_per_encoder2_embeds).square().mean() / (self.logit_scale.exp() * self.logit_scale.exp()) * batch_size

                    cyclic_loss = 0.25 * inmodal_cyclic_loss + 0.25 * crossmodal_cyclic_loss
                    
                    # crossmodal_cyclic_loss = (normalized_encoder1_embeds - normalized_encoder2_embeds).square().mean() / (self.logit_scale.exp() * self.logit_scale.exp()) * batch_size

                else:
                    # uniformity_loss = 0
                    # cross_encoder_uniform_loss = 0
                    # uniform_cyclic_loss = 0
                    # cyclic_direction_loss = 0
                    # alignment_loss = 0
                    # cyclic_loss = 0

                    uniformity_loss = torch.tensor(0)
                    cross_encoder_uniform_loss = torch.tensor(0)
                    uniform_cyclic_loss = torch.tensor(0)
                    cyclic_direction_loss = torch.tensor(0)
                    alignment_loss = torch.tensor(0)
                    cyclic_loss = torch.tensor(0)

            

            if wandb.config['cyclip_loss']:

                pass

            if wandb.config['simclr_loss']:
                # concat image and text embeddings
                image_text_embeddings = torch.cat([normalized_encoder1_embeds, normalized_encoder2_embeds], dim=0) # shape: (2 * batch_size, embed_dim)


                self_similarity = image_text_embeddings @ image_text_embeddings.t() * self.logit_scale.exp() # shape: (2 * batch_size, 2 * batch_size)

                # print('self similarity ', self_similarity.shape) 



                batch_size = normalized_encoder1_embeds.shape[0]

                labels_image_to_text = torch.arange(start=batch_size, end=2 * batch_size).to(self.device)


                labels_text_to_image = torch.arange(start=0, end=batch_size).to(self.device)

                simclr_labels = torch.cat([labels_image_to_text, labels_text_to_image], dim=0) #shape: (2 * batch_size)

                # print('labels ', labels.shape)

                simclr_loss = self.loss(self_similarity, simclr_labels) # automatically divides by 2N

                # print('simclr loss ', simclr_loss)

            else:
                simclr_loss = torch.tensor(0)




                

                


            if wandb.config['svd_loss']:
                encoder1_svd_vals = torch.linalg.svdvals(normalized_encoder1_embeds)
                encoder2_svd_vals = torch.linalg.svdvals(normalized_encoder2_embeds)

                # difference between highest and lowest singular values
                encover1_svd_diff = encoder1_svd_vals.max() - encoder1_svd_vals.min()
                encoder2_svd_diff = encoder2_svd_vals.max() - encoder2_svd_vals.min()

                svd_loss = 0.5 * encover1_svd_diff ** 2 + 0.5 *  encoder2_svd_diff ** 2

                # compute average of singular values
                # encover1_svd_avg = torch.mean(encoder1_svd_vals)
                # encoder2_svd_avg = torch.mean(encoder2_svd_vals)

                # svd_loss = 

            if wandb.config['intra_modality_loss']:
                # find cosine similarities between image embeddings themselves
                scaled_image_image_similarity = normalized_encoder1_embeds @ normalized_encoder1_embeds.t() * self.intra_modality_logit_scale.exp()

                # find cosine similarities between text embeddings themselves
                scaled_text_text_similarity = normalized_encoder2_embeds @ normalized_encoder2_embeds.t() * self.intra_modality_logit_scale.exp()

                intra_modality_loss = self.loss(scaled_image_image_similarity, labels) * encoder1_weight + self.loss(scaled_text_text_similarity, labels) * encoder2_weight

                # print('intra loss: ,', intra_modality_loss)
            if wandb.config['rsa_loss']:
                
                text_text_cosine_similarities = normalized_encoder2_embeds @ normalized_encoder2_embeds.t()
                image_image_cosine_similarities = normalized_encoder1_embeds @ normalized_encoder1_embeds.t()

                # i can make intra-modality cosine sims and inter modality cosine sims as similiar as possible

                inter_image_rsa = F.cosine_similarity(logits_per_encoder1_embeds.reshape(1, -1), image_image_cosine_similarities.reshape(1, -1))
                inter_text_rsa = F.cosine_similarity(logits_per_text.reshape(1, -1), text_text_cosine_similarities.reshape(1, -1))

                rsa_loss = -(inter_image_rsa + inter_text_rsa) / 2

            if wandb.config['pearson_loss']:

                logits_per_image = logits_per_encoder1_embeds
                logits_per_text = logits_per_encoder2_embeds

                '''
                ACTUAL PEARSON CORRELATION IN RSA LOSS
                '''

                text_text_cosine_similarities = normalized_encoder2_embeds @ normalized_encoder2_embeds.t()
                image_image_cosine_similarities = normalized_encoder1_embeds @ normalized_encoder1_embeds.t()

                image_text_RSM = logits_per_image[torch.tril(torch.ones(logits_per_image.shape[0], logits_per_image.shape[1]), diagonal=-1).bool()] # shape: (k)

                text_RSM = text_text_cosine_similarities[torch.tril(torch.ones(text_text_cosine_similarities.shape[0], text_text_cosine_similarities.shape[1]), diagonal=-1).bool()]    # shape: (k)

                image_RSM = image_image_cosine_similarities[torch.tril(torch.ones(image_image_cosine_similarities.shape[0], image_image_cosine_similarities.shape[1]), diagonal=-1).bool()]   # shape: (k)
                
                # stack image_text RSM and text RSM 
                # and then calculate pearson correlation
                # then calculate loss
                stacked_RSM_text = torch.stack([image_text_RSM, text_RSM], dim=0) # shape: (2, k)
                stacked_RSM_image = torch.stack([image_text_RSM, image_RSM], dim=0) # shape: (2, k)

                text_corr = torch.corrcoef(stacked_RSM_text) # shape: (2, 2)
                image_corr = torch.corrcoef(stacked_RSM_image) # shape: (2, 2)

                pearson_rsa_loss = -(text_corr[0, 1] + image_corr[0, 1]) / 2


            inter_modality_loss: torch.Tensor = self.loss(logits_per_encoder1_embeds, labels) * encoder1_weight + self.loss(logits_per_encoder2_embeds, labels) * encoder2_weight 

            



            if wandb.config['scaled_denominator']:
                inter_modality_loss = inter_modality_loss - torch.log(torch.tensor(logits_per_encoder1_embeds.shape[0]).to(self.device))

            del labels

            if wandb.config['remove_contrastive_loss']:
                loss = 0
            else:

                loss = inter_modality_loss.clone() # so that I can track inter_modality_loss separately and return it

            if wandb.config['intra_modality_loss']:

                loss += intra_modality_loss
                loss = loss / 2
                # loss = (intra_modality_loss + inter_modality_loss) / 2

            if wandb.config['cosine_uniformity_loss']:
                logits_images_per_images = normalized_encoder1_embeds @ normalized_encoder1_embeds.t()

                logits_texts_per_texts = normalized_encoder2_embeds @ normalized_encoder2_embeds.t()

                logits_images_per_texts = normalized_encoder1_embeds @ normalized_encoder2_embeds.t()

                # image_text_uniformity = ((-logits_images_per_texts).exp().mean() - (-logits_images_per_texts.diag()).exp().mean()).log()

                # image_uniformity = ((-logits_images_per_images).exp().mean() - (-logits_images_per_images.diag()).exp().mean()).log()

                # text_uniformity = ((-logits_texts_per_texts).exp().mean() - (-logits_texts_per_texts.diag()).exp().mean()).log()

                
                lower_diagonal_ones = torch.ones((len(normalized_encoder1_embeds), len(normalized_encoder2_embeds))).to(self.device).tril(diagonal = -1) 

                upper_diagonal_ones = torch.ones((len(normalized_encoder2_embeds), len(normalized_encoder1_embeds))).to(self.device).triu(diagonal = 1)

                off_diagonal_ones = lower_diagonal_ones + upper_diagonal_ones

                off_diagonal_ones = off_diagonal_ones.to(self.device)

                image_text_uniformity = torch.masked_select(logits_images_per_texts, off_diagonal_ones == 1).exp().mean().log()

                
                # image_text_uniformity = ((-logits_images_per_texts).exp().mean()).log()

                image_uniformity = torch.masked_select(logits_images_per_images, lower_diagonal_ones == 1).exp().mean().log()

                # image_uniformity = ((-logits_images_per_images).exp().mean()).log()

                text_uniformity = torch.masked_select(logits_texts_per_texts, lower_diagonal_ones == 1).exp().mean().log()

                # text_uniformity = ((-logits_texts_per_texts).exp().mean()).log()



                cosine_uniformity = 0.5 * image_text_uniformity + 0.25 * image_uniformity + 0.
                25 * text_uniformity

                loss += cosine_uniformity

            if wandb.config['cosine_align_loss']:

                logits_images_per_texts = normalized_encoder1_embeds @ normalized_encoder2_embeds.t()

                cosine_align_loss = (-logits_images_per_texts).diag().mean()

                loss += cosine_align_loss


            if wandb.config['rsa_loss']:

                loss += rsa_loss
                # loss = inter_modality_loss + rsa_loss
            if wandb.config['pearson_loss']:
                loss += pearson_rsa_loss
                # loss = inter_modality_loss + pearson_rsa_loss
            if wandb.config['svd_loss']:
                loss += svd_loss
                # loss = inter_modality_loss + svd_loss

            if wandb.config['cyclip_loss']:
                loss += cyclic_loss
                # loss = inter_modality_loss + cyclic_loss
            if wandb.config['uniformity_loss']:
                loss += 1 * uniformity_loss
                # loss = inter_modality_loss + uniformity_loss

            if wandb.config['cross_uniformity_loss']:

                loss += 1 * cross_encoder_uniform_loss

            if wandb.config['simclr_loss']:
                loss += simclr_loss

            


            

            if wandb.config['alignment_loss']:

                

                loss += 1 * alignment_loss

                # if wandb.config['alignment_loss']:
                #     alignment_loss = (normalized_encoder1_embeds - normalized_encoder2_embeds).norm(dim=1).pow(2).mean()

                #     loss = inter_modality_loss + 0.5 * alignment_loss + 0.5 * (uniformity_loss) 

            if wandb.config['uniform_cyclic_loss']:
                loss += uniform_cyclic_loss

           

                # if wandb.config['remove_contrastive_loss']:
                #     loss = 0.5 * alignment_loss + 0.5 * (0.33 * encoder1_uniformity_loss + 0.33 * encoder2_uniformity_loss+ 0.33 * cross_encoder_uniform_loss) 
                # else:

                #     loss = 0.33 * inter_modality_loss + 0.33 * alignment_loss + 0.33 * (0.33 * encoder1_uniformity_loss + 0.33 * encoder2_uniformity_loss+ 0.33 * cross_encoder_uniform_loss) 

            # if wandb.config['remove_contrastive_loss']:
            #     loss = loss - inter_modality_loss

            # else:
            #     loss = inter_modality_loss
                

            

            if wandb.config['cyclic_direction_loss']:
                loss += 1 * cyclic_direction_loss

            if output_intra_modality_loss:
                loss = {
                    'inter_modality': inter_modality_loss.item(),
                    'simclr': simclr_loss.item(),
                    'rsa': rsa_loss.item() if wandb.config['rsa_loss'] else -100,
                    'intra_modality': intra_modality_loss.item() if wandb.config['intra_modality_loss'] else -100,
                    'pearson_rsa': pearson_rsa_loss.item() if wandb.config['pearson_loss'] else -100,
                    'cyclic_direction': cyclic_direction_loss.item(),
                    'svd': svd_loss.item() if wandb.config['svd_loss'] else -100,
                    'uniformity': uniformity_loss.item(),
                    'alignment': alignment_loss.item(),
                    'cyclic': cyclic_loss.item(),
                    'uniform_cyclic': uniform_cyclic_loss.item(),
                    'cross_uniformity': cross_encoder_uniform_loss.item(),
                    
                    'total': loss.item(),
                }



        # outputs = CLIPOutput(
        #     loss=loss,
        #     logits_per_image = logits_per_encoder1_embeds,
        #     logits_per_text = logits_per_encoder2_embeds,
        #     image_embeds = normalized_encoder1_embeds,
        #     text_embeds = normalized_encoder2_embeds,
        #     vision_model_output=encoder1_outputs,
        #     text_model_output=encoder2_outputs
        # )
            
        outputs = HFClipOutput(
            loss=loss,
            logits_per_image = logits_per_encoder1_embeds,
            logits_per_text = logits_per_encoder2_embeds,
            text_embeds = normalized_encoder2_embeds,
            image_embeds = normalized_encoder1_embeds,
            encoder1_hidden_states=encoder1_hidden_states,
            encoder2_hidden_states=encoder2_hidden_states,
            encoder1_input_ids = encoder1_input_ids,
            encoder2_input_ids = encoder2_input_ids
        )


        if return_all:
            return outputs
        
        del outputs, normalized_encoder1_embeds, normalized_encoder2_embeds
        
        logits_per_image = logits_per_encoder1_embeds
        logits_per_text = logits_per_encoder2_embeds
        if output_loss:
            
            return logits_per_image, logits_per_text, loss
        else:
            return logits_per_image, logits_per_text



    
  
    def forward_old(self, preprocessed_images, captions, output_loss=True, return_all=False, output_intra_modality_loss=False):

        '''
        outputs = CLIPOutput(
            loss=loss,
            logits_per_image= logits_per_image,
            logits_per_text= logits_per_text,
            text_embeds= text_embeds,
            image_embeds= image_embeds,
        )
        '''

        # inputs = self.processor(text=['captions', 'hello'], images=image, return_tensors="pt", padding=True)

        # tokenized_captions = self.tokenize_captions(captions)

        if not self.text_only:


            tokenized_captions = captions.to(self.device)
            preprocessed_images = preprocessed_images.to(self.device)

            outputs = self.model(input_ids=tokenized_captions['input_ids'], attention_mask=tokenized_captions['attention_mask'], pixel_values=preprocessed_images, return_loss=output_loss)
            
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            # normalized features
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text
        else:
            # in this case, "preprocessed_images" are actually captions

            tokenized_captions1 = captions.to(self.device)
            tokenized_captions2 = preprocessed_images.to(self.device)
            

            outputs1 = self.text_model1(**tokenized_captions1)
            outputs2 = self.text_model2(**tokenized_captions2)



            text_embeds = outputs1.text_embeds
            image_embeds = outputs2.text_embeds
            

            # normalized features
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

            logits_per_image = image_embeds @ text_embeds.t() * self.logit_scale.exp() # logit_scale.exp() is 1 / temperature, so 100 for 0.01
            logits_per_text = text_embeds @ image_embeds.t() * self.logit_scale.exp()

        # this is exactly the same code (although I wrote it) as huggingface clip's loss as in https://github.dev/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py

        

        labels = torch.arange(logits_per_image.shape[0]).to(self.device)

        image_weight = wandb.config['i_t_loss_weight']
        text_weight = wandb.config['t_i_loss_weight']

        loss = 0

        if output_loss == True:

            if wandb.config['intra_modality_loss']:
                # find cosine similarities between image embeddings themselves
                scaled_image_image_similarity = image_embeds @ image_embeds.t() * self.intra_modality_logit_scale.exp()

                # find cosine similarities between text embeddings themselves
                scaled_text_text_similarity = text_embeds @ text_embeds.t() * self.intra_modality_logit_scale.exp()

                intra_modality_loss = self.loss(scaled_image_image_similarity, labels) * image_weight + self.loss(scaled_text_text_similarity, labels) * text_weight

                # print('intra loss: ,', intra_modality_loss)
            if wandb.config['rsa_loss']:
                
                text_text_cosine_similarities = text_embeds @ text_embeds.t()
                image_image_cosine_similarities = image_embeds @ image_embeds.t()

                # i can make intra-modality cosine sims and inter modality cosine sims as similiar as possible

                inter_image_rsa = F.cosine_similarity(logits_per_image.reshape(1, -1), image_image_cosine_similarities.reshape(1, -1))
                inter_text_rsa = F.cosine_similarity(logits_per_text.reshape(1, -1), text_text_cosine_similarities.reshape(1, -1))

                rsa_loss = -(inter_image_rsa + inter_text_rsa) / 2

            if wandb.config['pearson_loss']:

                '''
                ACTUAL PEARSON CORRELATION IN RSA LOSS
                '''

                text_text_cosine_similarities = text_embeds @ text_embeds.t()
                image_image_cosine_similarities = image_embeds @ image_embeds.t()

                image_text_RSM = logits_per_image[torch.tril(torch.ones(logits_per_image.shape[0], logits_per_image.shape[1]), diagonal=-1).bool()] # shape: (k)

                text_RSM = text_text_cosine_similarities[torch.tril(torch.ones(text_text_cosine_similarities.shape[0], text_text_cosine_similarities.shape[1]), diagonal=-1).bool()]    # shape: (k)

                image_RSM = image_image_cosine_similarities[torch.tril(torch.ones(image_image_cosine_similarities.shape[0], image_image_cosine_similarities.shape[1]), diagonal=-1).bool()]   # shape: (k)
                
                # stack image_text RSM and text RSM 
                # and then calculate pearson correlation
                # then calculate loss
                stacked_RSM_text = torch.stack([image_text_RSM, text_RSM], dim=0) # shape: (2, k)
                stacked_RSM_image = torch.stack([image_text_RSM, image_RSM], dim=0) # shape: (2, k)

                text_corr = torch.corrcoef(stacked_RSM_text) # shape: (2, 2)
                image_corr = torch.corrcoef(stacked_RSM_image) # shape: (2, 2)

                pearson_rsa_loss = -(text_corr[0, 1] + image_corr[0, 1]) / 2


            inter_modality_loss = self.loss(logits_per_image, labels) * image_weight + self.loss(logits_per_text, labels) * text_weight


            if wandb.config['intra_modality_loss']:
                loss = (intra_modality_loss + inter_modality_loss) / 2
            elif wandb.config['rsa_loss']:
                loss = inter_modality_loss + rsa_loss
            elif wandb.config['pearson_loss']:
                loss = inter_modality_loss + pearson_rsa_loss
                
            else:
                loss = inter_modality_loss

            if output_intra_modality_loss:
                loss = {
                    'inter_modality': inter_modality_loss.item(),
                    'rsa': rsa_loss.item() if wandb.config['rsa_loss'] else -100,
                    'intra_modality': intra_modality_loss.item() if wandb.config['intra_modality_loss'] else -100,
                    'pearson_rsa': pearson_rsa_loss.item() if wandb.config['pearson_loss'] else -100,
                    'total': loss.item(),
                }

        outputs = CLIPOutput(
            loss=loss,
            logits_per_image= logits_per_image,
            logits_per_text= logits_per_text,
            text_embeds= text_embeds,
            image_embeds= image_embeds,
        )


        if return_all:
            return outputs
        
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text
        if output_loss:
            
            return logits_per_image, logits_per_text, loss
        else:
            return logits_per_image, logits_per_text

