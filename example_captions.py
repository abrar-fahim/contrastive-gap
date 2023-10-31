from clip_caption_predict import Predictor
import torchvision.datasets as dset
import clip
import torch


torch.manual_seed(42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, preprocess = clip.load("ViT-B/32", device=device)

train_dataset = dset.CocoCaptions(root = './datasets/mscoco/val2014',
        annFile = 'datasets/mscoco/annotations/captions_val2014.json',
        # transform=[transforms.PILToTensor()])
        transform=preprocess,
)


predictor = Predictor()
predictor.setup()

# pass an image


