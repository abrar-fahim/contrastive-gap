import torch
from transformers import CLIPTextConfig, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPVisionConfig

from transformers.models.clip.modeling_clip import CLIPOutput

from src.config import *

from clips.encoder import Encoder



class ImageEncoder(Encoder):


    # init
    def __init__(self, preprocessor, CLIPVisionConfig: CLIPVisionConfig, from_pretrained=False, name='Untitled Image Encoder'):
        '''
        Set CLIPImageConfig with appropriate image size if using diff image sizes
        '''
        super().__init__()

        self.device = torch.device(training_hyperparameters['cuda_device'] if torch.cuda.is_available() else "cpu")

        self.preprocessor = preprocessor
        self.CLIPVisionConfig = CLIPVisionConfig
        self.hidden_size = CLIPVisionConfig.hidden_size

        self.pooler_layer_norm = torch.nn.LayerNorm(CLIPVisionConfig.hidden_size, eps=CLIPVisionConfig.layer_norm_eps, elementwise_affine=False) # no trainable params

        if from_pretrained:
            print()
            print(f" --- Initializing {name} from pretrained model ---")
            print()
            self.image_model = CLIPVisionModelWithProjection.from_pretrained(training_hyperparameters['hf_clip_model']).to(self.device)

        else:
            print()
            print(f" --- Initializing {name} from scratch --- ")
            print()

            self.image_model = CLIPVisionModelWithProjection(CLIPVisionConfig).to(self.device)

            self.image_model.init_weights()

        for param in self.image_model.parameters():
            param.requires_grad = True


    def forward(self, images, output_hidden_states=False):

        preprocessed_images = self.preprocess_images(images)

        image_features = self.image_model(pixel_values=preprocessed_images, output_hidden_states=output_hidden_states)
        del preprocessed_images

        return {
            'embeds': image_features.image_embeds,
            'hidden_states': image_features.hidden_states if output_hidden_states else None,
            'input_ids': None
        }
    


    def preprocess_images(self, images):

        preprocessed_images = tuple(self.preprocessor(img) for img in images)

        preprocessed_images = torch.stack(preprocessed_images).to(self.device)
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


