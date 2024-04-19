import sys
import os

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# add sibling directory to path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



import torch

from torch.optim import SGD

from torch.distributions.multivariate_normal import MultivariateNormal

from src.my_ce_loss import MyCrossEntropyLoss

import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.linear_model import LogisticRegression

from src.scheduler import cosine_scheduler

from sklearn.decomposition import PCA

from dataset import SyntheticDataset

# generate cluster of points around a point in unit sphere

def plot(a_points, b_points):

    # normalize
    a_points = a_points / a_points.norm(dim=1).view(-1, 1)
    b_points = b_points / b_points.norm(dim=1).view(-1, 1)

    pca = PCA(n_components=3)

    ab = torch.cat((a_points, b_points), dim=0)

    ab = pca.fit_transform(ab)

    a_points = ab[:n_visualize]
    b_points = ab[n_visualize:]
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    for i in range(n_visualize):
        ax.scatter(a_points[i, 0], a_points[i, 1], a_points[i, 2], c='r')
        ax.scatter(b_points[i, 0], b_points[i, 1], b_points[i, 2], c='b')

    # use fixed scale

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.show()


def linear_seperability(a_points, b_points):

    # normalize
    a_points = a_points / a_points.norm(dim=1).view(-1, 1)
    b_points = b_points / b_points.norm(dim=1).view(-1, 1)

    n_train = int(0.2 * len(a_points))
    n_test = len(a_points) - n_train

    # get random indices

    indices = torch.randperm(len(a_points), device=a_points.device)

    train_indices = indices[:n_train]

    test_indices = indices[n_train:]

    train_image_embeds = a_points[train_indices]
    test_image_embeds = a_points[test_indices]

    train_text_embeds = b_points[train_indices]
    test_text_embeds = b_points[test_indices]

    # Generate train dataset
    train_image_text_embeds = torch.cat((train_image_embeds, train_text_embeds), dim=0)
    # generate labels
    train_labels = torch.cat((torch.zeros(n_train), torch.ones(n_train))) # 0 for image, 1 for text

    # shuffle
    shuffle_indices = torch.randperm(2 * n_train)

    train_image_text_embeds = train_image_text_embeds[shuffle_indices]
    train_labels = train_labels[shuffle_indices]

    # Generate test dataset
    test_image_text_embeds = torch.cat((test_image_embeds, test_text_embeds), dim=0)
    # generate labels
    test_labels = torch.cat((torch.zeros(n_test), torch.ones(n_test))) # 0 for image, 1 for text

    # shuffle
    test_shuffle_indices = torch.randperm(2 * n_test)

    test_image_text_embeds = test_image_text_embeds[test_shuffle_indices]
    test_labels = test_labels[test_shuffle_indices]

    

    
    # fit linear classifier on train set to predict text embeddings from image embeddings
    clf = LogisticRegression(random_state=0).fit(train_image_text_embeds.cpu(), train_labels.cpu())

    # get accuracy on test set
    linear_seperability_accuracy = clf.score(test_image_text_embeds.cpu(), test_labels.cpu())

    return linear_seperability_accuracy


def classification_acc(a_points, b_points):
    
        # normalize
        a_points = a_points / a_points.norm(dim=1).view(-1, 1)
        b_points = b_points / b_points.norm(dim=1).view(-1, 1)

        # get logits
        logits = torch.matmul(a_points, b_points.t())

        scaled_logits = logits / T

        class_probs = torch.nn.functional.softmax(logits, dim=1)

        preds = torch.argmax(class_probs, dim=1)

        labels = torch.arange(a_points.shape[0], device=a_points.device)

        acc = torch.sum(preds == labels).item() / n

        return acc

d = 3

n = 2048

n_visualize = 30

T = 0.01

n_epochs = 10000

batch_size = 256

lr = 0.01

evaluate_every = 100

torch.manual_seed(0)

dataset = SyntheticDataset(n=n, d=d)




# select n_visualize points to visualize from a and b
plot(dataset[:n_visualize][0].detach().cpu(), dataset[:n_visualize][1].detach().cpu())



loss = MyCrossEntropyLoss()


# - optimizer -
sgd = SGD([dataset.ab], lr=lr)

n_steps = n_epochs * (n // batch_size)

# - scheduler -
scheduler = cosine_scheduler(sgd, lr, 100, n_steps)




epochs = tqdm(range(n_epochs))

loss_value = torch.tensor(0.0)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


for epoch in epochs:

    i = 0


    epochs.set_description(f'Epoch {epoch}, loss: {loss_value.item()}')

    for a_batch, b_batch in dataloader:

        a_batch = a_batch.to(device)
        b_batch = b_batch.to(device)

        step = epoch * (n // batch_size) + i

        # scheduler(step)

        sgd.zero_grad()

        

        # select batch_size points randomly from a and b
        indices = torch.randint(0, n, (batch_size,), device=device)

        # normalize
        a_batch = a_batch / a_batch.norm(dim=1).view(-1, 1)
        b_batch = b_batch / b_batch.norm(dim=1).view(-1, 1)
        
        # - loss -

        # find similarity between a_batch and b_batch
        # the similarity is the dot product of a_batch and b_batch

        logits = torch.matmul(a_batch, b_batch.t()) # shape (batch_size, batch_size)

        # scale with T
        scaled_logits = logits / T

        # labels are the diagonal of the matrix
        labels = torch.arange(batch_size, device=device)

        # compute loss
        loss_value = loss(scaled_logits, labels)

        # print('loss', loss_value.item())

        # - backward -
        loss_value.backward()

        # - step -
        sgd.step()


        if epoch % evaluate_every == 0 and i == 0:
            print('loss', loss_value.item())
            print('linear seperability', linear_seperability(a_batch.detach(), b_batch.detach()))
            print('classification accuracy', classification_acc(a_batch.detach(), b_batch.detach()))

        i += 1


# visualize the points
        
plot(dataset[:n_visualize][0].detach().cpu(), dataset[:n_visualize][1].detach().cpu())


