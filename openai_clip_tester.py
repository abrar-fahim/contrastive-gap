from PIL import Image
import requests
import clip
import json
import random

import torch

from matplotlib import pyplot as plt


# image1 = Image.open(requests.get('https://github.com/rmokady/CLIP_prefix_caption/raw/main/Images/COCO_val2014_000000165547.jpg', stream=True).raw) # DINING TABLE


# image1 = Image.open(requests.get('https://github.com/rmokady/CLIP_prefix_caption/raw/main/Images/COCO_val2014_000000579664.jpg', stream=True).raw) # BANANAS


# image1 = Image.open(requests.get('https://github.com/rmokady/CLIP_prefix_caption/raw/main/Images/CONCEPTUAL_04.jpg', stream=True).raw) # BANANAS
# image1 = Image.open(requests.get('https://m.media-amazon.com/images/I/61U7gUPxlvL.jpg', stream=True).raw) # BANANAS
image1 = Image.open(requests.get('https://media.istockphoto.com/id/996168058/vector/black-led-tv-television-screen-blank-on-white-wall-background.jpg?s=612x612&w=0&k=20&c=MQI3naQQ7dteEbt8xmiE97OEPy2UA30OAh0pwtyJt9I=', stream=True).raw) # BANANAS



# image2 = Image.open(requests.get('https://images.fineartamerica.com/images/artworkimages/mediumlarge/2/zebra-in-living-room-smelling-rug-side-matthias-clamer.jpg', stream=True).raw)
# image2 = Image.open(requests.get('https://static.uwalls.com/products/9000/9004/u07312pik1m_1200.webp', stream=True).raw)
# image2 = Image.open(requests.get('https://foter.com/photos/title/animal-print-dining-room-chairs.jpg', stream=True).raw) # this works, but is techincally also a dining table 
# image2 = Image.open(requests.get('https://previews.123rf.com/images/visible3dscience/visible3dscience1603/visible3dscience160314859/53977058-3d-rendered-illustration-of-zebra-cartoon-character-with-table-and-chair.jpg', stream=True).raw) # this is close

# image2 = Image.open(requests.get('https://pbxt.replicate.delivery/E60XW3jm65YtEtl8pihvhqBhIWEYCQEg29qSY3KGOBVPqFZE/tempfile.png', stream=True).raw)
# image2 = Image.open(requests.get('https://i.redd.it/y85hj65gk5e31.jpg', stream=True).raw) # THIS WORKS
# image2 = Image.open(requests.get('https://platthillnursery.com/wp-content/uploads/2020/02/maximize-space-window-houseplants-room-with-houseplants-shelves-beside-window.jpg', stream=True).raw) # THIS WORKS
# image2 = Image.open(requests.get('https://i.etsystatic.com/32710857/r/il/6b4d5b/5157493828/il_fullxfull.5157493828_722c.jpg', stream=True).raw)  # THIS WORKS

# image2 = Image.open(requests.get('https://media.gettyimages.com/id/533113012/it/foto/aynhoe-park.jpg?s=612x612&w=gi&k=20&c=sIx9JV5YzG_tnQTwgcvZsDPokmTm1cyPkD-8EblCW40=', stream=True).raw)  # THIS WORKS BETTER
image2 = Image.open(requests.get('https://i.etsystatic.com/27160608/r/il/ce5755/3499068372/il_570xN.3499068372_nky9.jpg', stream=True).raw) 










captions = ['A TV in a room', 'A a room TV in']

# captions = ['A wooden table sitting in front of a window']

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

captions = torch.cat([clip.tokenize(c) for c in captions]).to(device)

images = torch.cat([preprocess(i).unsqueeze(0) for i in [image1, image2]], dim=0)

# images = [image1, image2] # since processor expects a list of images




with torch.no_grad():
    image_features = model.encode_image(images)
    text_features = model.encode_text(captions)
    print('text_features ', text_features.shape)

    # find cosine similarity between text features
    cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
    similarity = cosine_similarity(text_features[0], text_features[1]).cpu().numpy()
    print('similarity ', similarity)
    
    logits_per_image, logits_per_text = model(images, captions)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
print('logits_per_image ', logits_per_image)
print('logits_per_text ', logits_per_text)

