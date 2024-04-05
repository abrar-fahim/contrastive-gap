import torch

# set random seed
torch.manual_seed(9999999)

loss = torch.nn.CrossEntropyLoss()

temperature = 0.01

n = 32

images = torch.randint(0, 5, (n, 512), dtype=torch.float32)

captions = torch.randint(0, 5, (n, 512), dtype=torch.float32)

# normalize
normalized_images = images / torch.norm(images, p=2, dim=1, keepdim=True)
normalized_captions = captions / torch.norm(captions, p=2, dim=1, keepdim=True)

logits_per_image = normalized_images @ normalized_captions.T / temperature
logits_per_text = normalized_captions @ normalized_images.T / temperature

labels = torch.arange(n)

image_loss = loss(logits_per_image, labels)

text_loss = loss(logits_per_text, labels)

total_loss = 0.5 * image_loss + 0.5 * text_loss

print('torch contrastive loss ', total_loss)

# implement my own loss


image_loss = 0
for i in range(n):
    loss = torch.exp(normalized_images[i] @ normalized_captions[i].T / temperature)

    denominator = 0
    for k in range(n):
        denominator += torch.exp(normalized_images[i] @ normalized_captions[k].T / temperature)

    loss = loss / denominator

    log_loss = torch.log(loss)
    image_loss += log_loss

text_loss = 0
for i in range(n):
    loss = torch.exp(normalized_captions[i] @ normalized_images[i].T / temperature)

    denominator = 0
    for k in range(n):
        denominator += torch.exp(normalized_captions[i] @ normalized_images[k].T / temperature)

    loss = loss / denominator

    log_loss = torch.log(loss)
    text_loss += log_loss

my_total_loss = (-1 / (2 * n)) * (image_loss + text_loss)

print('my contrastive loss ', my_total_loss)

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