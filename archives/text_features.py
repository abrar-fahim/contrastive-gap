from transformers import AutoTokenizer, CLIPModel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy.linalg import norm

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

classes = ['a photo of something made of metal', 'food', 'animal', 'clothing', 'furniture', 'etar mane ki?']
# classes = ["a photo of a cat", "a photo of a dog"]

# classes = ["white dog on brown couch", "brown dog on white couch"]

inputs = tokenizer(classes, padding=True, return_tensors="pt")
text_features = model.get_text_features(**inputs) # shape: (5, 512)

# plot cosine similarity of text features


plt.figure(figsize=(10, 10))

print('text features ', text_features)

text_features = text_features.detach().cpu()

# dummy vector for text features
# text_features = np.array([[1, 2, 3, 4, 5], [-1, -2, -3, -4, -5]])

norms = np.linalg.norm(text_features, axis=1)

# add dimension to norms 
norms = np.expand_dims(norms, axis=1) # shape: (5,1)

# norms matrix
norms_matrix = np.matmul(norms, norms.T) # shape: (5,5)

print('norms ', norms.shape)

cosine_similarities = np.matmul(text_features, text_features.T) / norms_matrix

print('cosine_similarities ', np.matmul(text_features, text_features.T))

# sns.heatmap((text_features @ text_features.T).numpy(), annot=True, fmt=".1f", linewidths=.5, square=True, xticklabels=classes, yticklabels=classes, cbar=False)
sns.heatmap(cosine_similarities, annot=True, fmt=".2f", linewidths=.5, square=True, xticklabels=classes, yticklabels=classes, cbar=False)
plt.title("Cosine Similarity of Text Features")
plt.show()


# print("Text features:", text_features.shape)