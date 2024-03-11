import torch
from transformers import CLIPTextConfig, CLIPTextModelWithProjection, CLIPVisionModelWithProjection

from transformers.models.clip.modeling_clip import CLIPOutput

from src.config import *

from clips.encoder import Encoder



class ImageEncoder(Encoder):


    # init
    def __init__(self, preprocessor, CLIPVisionConfig, from_pretrained=False, name='Untitled Image Encoder'):
        '''
        Set CLIPImageConfig with appropriate image size if using diff image sizes
        '''
        super().__init__()

        self.device = torch.device(training_hyperparameters['cuda_device'] if torch.cuda.is_available() else "cpu")

        self.preprocessor = preprocessor

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
            'hidden_states': image_features.hidden_states if output_hidden_states else None
        }
    


    def preprocess_images(self, images):

        preprocessed_images = tuple(self.preprocessor(img) for img in images)

        preprocessed_images = torch.stack(preprocessed_images).to(self.device)
        return preprocessed_images