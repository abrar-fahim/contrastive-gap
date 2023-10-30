url = "https://canary.contestimg.wish.com/api/webimage/61b241a3a4ee2ecaf2f63c77-large.jpg?cache_buster=bbeee1fdb460a1d12bc266824914e030"

# get HF image fearures
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import torch
import clip
from PIL import Image
import numpy as np
from hf_clip import HFClip

device = "cuda" if torch.cuda.is_available() else "cpu"




model, preprocess = clip.load("ViT-B/32", device=device)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

model = HFClip()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open(requests.get(url, stream=True).raw)

image = preprocess(Image.open(requests.get(url, stream=True).raw)).unsqueeze(0).to(device)

# inputs = processor(images=image, return_tensors="pt")



# outputs = model.get_image_features(pixel_values=image)
outputs = model.encode_image(image)
# outputs = model.get_image_features(**inputs)
pooled_output_hf = outputs.detach().cpu().numpy()

# get OpenAI image features



model, preprocess = clip.load("ViT-B/32", device=device)
image = preprocess(Image.open(requests.get(url, stream=True).raw)).unsqueeze(0).to(device)


with torch.no_grad():
   image_features = model.encode_image(image)
pooled_output_clip = image_features.detach().cpu().numpy()

# check difference


# assert np.allclose(pooled_output_hf, pooled_output_clip, atol=0.1), "hf and clip too different"

if np.allclose(pooled_output_hf, pooled_output_clip, atol=0.01):
    print("hf and clip similar YAAAYYYY")
else:
    print("hf and clip too different :(")

