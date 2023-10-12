from PIL import Image
import requests
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import seaborn as sns

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

classes = ["a photo of a cat", "a photo of a dog"]

inputs = processor(text=classes, images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)

print('outputs ', outputs)

logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
print('logits_per_image ', logits_per_image)
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

print("Label probs:", probs)