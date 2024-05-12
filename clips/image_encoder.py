import torch
from transformers import CLIPTextConfig, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPVisionConfig

from transformers.models.clip.modeling_clip import CLIPOutput

from src.config import *

from clips.encoder import Encoder

from clips.rn50 import Rn50ModelWithProjection
from clips.projection_layer import MultiLayerProjection





class ImageEncoder(Encoder):


    # init
    def __init__(self, preprocessor, CLIPVisionConfig: CLIPVisionConfig, from_pretrained=False, name='Untitled Image Encoder'):
        '''
        Set CLIPImageConfig with appropriate image size if using diff image sizes
        '''
        super().__init__()

        self.device = torch.device(config_cuda_device if torch.cuda.is_available() else "cpu")

        self.preprocessor = preprocessor
        self.CLIPVisionConfig = CLIPVisionConfig
        self.hidden_size = CLIPVisionConfig.hidden_size

        self.added_projection_layer = None

        self.pooler_layer_norm = torch.nn.LayerNorm(CLIPVisionConfig.hidden_size, eps=CLIPVisionConfig.layer_norm_eps, elementwise_affine=False) # no trainable params

        self.vision_model = wandb.config['vision_model']

        if from_pretrained:
        # if False:
            print()
            print(f" --- Initializing {name} from pretrained model ---")
            print()
            self.image_model = CLIPVisionModelWithProjection.from_pretrained(wandb.config['hf_clip_model']).to(self.device)

        else:
            print()
            print(f" --- Initializing {name}: {self.vision_model} from scratch --- ")
            print()

            if self.vision_model == 'RN50':
                self.image_model = Rn50ModelWithProjection(CLIPVisionConfig).to(self.device)
            elif self.vision_model == 'VIT':
                self.image_model = CLIPVisionModelWithProjection(CLIPVisionConfig).to(self.device)

            self.image_model.init_weights()

        for param in self.image_model.parameters():
            param.requires_grad = True


        if wandb.config['finetune_multi_layer_projection']:

            print()
            print(f" --- Adding multi layer projection layer to {name}: {self.vision_model}  --- ")
            print()
            self.added_projection_layer = MultiLayerProjection()

            
            # freeze CLIP model
            for param in self.image_model.parameters():
                param.requires_grad = False

            # unfreese CLIP's projection layer
            for param in self.image_model.visual_projection.parameters():
                param.requires_grad = True

            # unfreeze projection layer
            for param in self.added_projection_layer.parameters():
                param.requires_grad = True


            # requires grad stuff LATER


    def forward(self, images, output_hidden_states=False):

        # preprocessed_images = self.preprocess_images(images)


        images = images.to(self.device)

        image_features = self.image_model(pixel_values=images, output_hidden_states=output_hidden_states)
        # del preprocessed_images

        if self.W is not None and self.W_set:
            image_features.image_embeds = self.align_embeddings(image_features.image_embeds)

        if self.added_projection_layer is not None:
            image_features.image_embeds = self.added_projection_layer(image_features.image_embeds)

        return {
            'embeds': image_features.image_embeds,
            'hidden_states': image_features.hidden_states if output_hidden_states else None,
            'input_ids': None
        }
    


    def preprocess_images(self, images):

        print('preprocessing images')

        preprocessed_images = tuple(self.preprocessor(img) for img in images)

        preprocessed_images = torch.stack(preprocessed_images).to(self.device)

        print('preprocessing done')
        return preprocessed_images
    
    def pool_hidden_state(self, hidden_state: torch.FloatTensor, input_ids: torch.Tensor):

        '''
        `hidden_state` is `torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`
        `LayerNorm` without trainable params before pooling
        Need to normalize since we'll eventually compute cosine similarity, but maybe dont do that here?
        '''

        assert hidden_state.shape[2] == self.CLIPVisionConfig.hidden_size

        pooled_output = hidden_state[:, 0, :]
        pooled_output = self.pooler_layer_norm(pooled_output)

        assert pooled_output.shape == (hidden_state.shape[0], hidden_state.shape[2]), f"pooled_output.shape = {pooled_output.shape}, hidden_state.shape = {hidden_state.shape}"

        

        return pooled_output


