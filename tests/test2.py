import torch

import sys
import os

print(torch.exp(torch.tensor(1e-3)))

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# add sibling directory to path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.my_ce_loss import MyCrossEntropyLoss

myloss = MyCrossEntropyLoss()

# set random seed
torch.manual_seed(9999999)

loss = torch.nn.CrossEntropyLoss()

temperature = 0.01

n = 10

# images = torch.randint(0, 4, (n, 3), dtype=torch.float32)
images = torch.randn(n, 512)

# captions = torch.randint(0, 4, (n, 3), dtype=torch.float32)

captions = torch.randn(n, 512)

# normalize
normalized_images = images / torch.norm(images, p=2, dim=1, keepdim=True)
normalized_captions = captions / torch.norm(captions, p=2, dim=1, keepdim=True)

logits_per_image = normalized_images @ normalized_captions.T / temperature
logits_per_text = normalized_captions @ normalized_images.T / temperature

labels = torch.arange(n)

image_loss = loss(logits_per_image, labels)

text_loss = loss(logits_per_text, labels)

total_loss = 0.5 * image_loss + 0.5 * text_loss

print("torch image loss ", image_loss)
print("torch text loss ", text_loss)

print('torch contrastive loss ', total_loss)



'''
my loss with torch, vectorized
'''

print('my loss with torch, vectorized ', 0.5 * myloss(logits_per_image, labels) + 0.5 * myloss(logits_per_text, labels))

exit()


scaled_logits_per_image = logits_per_image - logits_per_image.max(dim=1).values.view(-1, 1)
scaled_logits_per_text = logits_per_text - logits_per_text.max(dim=1).values.view(-1, 1)

image_diagonals = torch.gather(scaled_logits_per_image, 1, labels.view(-1, 1))
text_diagonals = torch.gather(scaled_logits_per_text, 1, labels.view(-1, 1))



image_loss = torch.sum(torch.log(torch.exp(image_diagonals) / (torch.exp(scaled_logits_per_image).sum(dim=1).view(-1, 1))))

text_loss = torch.sum(torch.log(torch.exp(text_diagonals) / (torch.exp(scaled_logits_per_text).sum(dim=1).view(-1, 1))))

my_total_loss = (-1 / (2 * n)) * (image_loss + text_loss)
print()

print('my image loss ', -image_loss / n)
print('my text loss ', -text_loss / n)

print('my contrastive loss torch vectorized ', my_total_loss)
print()



# implement my own loss





image_loss = 0
for i in range(n):
    # loss = torch.exp(normalized_images[i] @ normalized_captions[i].T / temperature)
    # loss = torch.exp(scaled_logits_per_image[i, i])
    loss = torch.exp(image_diagonals[i])

    denominator = 0
    denominator = torch.exp(scaled_logits_per_image).sum(dim=1)[i]
    # for k in range(n):
    #     # denominator += torch.exp(normalized_images[i] @ normalized_captions[k].T / temperature)
    #     denominator += torch.exp(scaled_logits_per_image[i, k])

    loss = loss / denominator

    log_loss = torch.log(loss)
    image_loss += log_loss

text_loss = 0
for i in range(n):
    # loss = torch.exp(normalized_captions[i] @ normalized_images[i].T / temperature)
    # loss = torch.exp(scaled_logits_per_text[i, i])
    loss = torch.exp(text_diagonals[i])

    denominator = 0
    denominator = torch.exp(scaled_logits_per_text).sum(dim=1)[i]
    # for k in range(n):
    #     # denominator += torch.exp(normalized_captions[i] @ normalized_images[k].T / temperature)
    #     denominator += torch.exp(scaled_logits_per_text[i, k])

    loss = loss / denominator

    log_loss = torch.log(loss)
    text_loss += log_loss

my_total_loss = (-1 / (2 * n)) * (image_loss + text_loss)
print()

print('my contrastive loss ', my_total_loss)

exit()

'''
My contrastive loss with 1/n in denominator
'''
image_loss = 0
for i in range(n):
    loss = torch.exp(normalized_images[i] @ normalized_captions[i].T / temperature)

    denominator = 0
    for k in range(n):
        denominator += torch.exp(normalized_images[i] @ normalized_captions[k].T / temperature)
    denominator = denominator / n
    loss = loss / denominator

    log_loss = torch.log(loss)
    image_loss += log_loss

text_loss = 0
for i in range(n):
    loss = torch.exp(normalized_captions[i] @ normalized_images[i].T / temperature)

    denominator = 0
    for k in range(n):
        denominator += torch.exp(normalized_captions[i] @ normalized_images[k].T / temperature)
    denominator = denominator / n
    loss = loss / denominator

    log_loss = torch.log(loss)
    text_loss += log_loss

my_total_denom_n_loss = (-1 / (2 * n)) * (image_loss + text_loss)

print('my contrastive loss with 1/n in denominator ', my_total_denom_n_loss)

'''
contrastive loss with added factor 
'''

print('torch loss with lgn added ', total_loss - torch.log(torch.tensor(n)))  