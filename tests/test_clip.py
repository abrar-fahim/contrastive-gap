import sys
import os

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# add sibling directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from clips.hf_clip import HFClip
import torch
import random
from dataset_processors.mscoco_processor import MSCOCOProcessor
from tqdm import tqdm


configs = {
    'val_batch_size': 64,
    'val_dataset_size': 512,
    'n': int(1e7),
    'checkpoint_path': 'checkpoints/T0.01_W0.5_0.5_Lit.pt'
    # no need to set temps here, because temp matters only when training
}

# if main


def calculate_coverage(clip_model: HFClip, dataset_processor):
    '''
    Calculates how much of unit sphere is covered by the image (for now) embeddings of the validation dataset

    1. Generate random point on unit sphere
    2. Find closest embedding
    3. Find cosine distance between random point and closest embedding
    4. Repeat 1-3 for n times
    5. Calculate average distance
    '''

    device = torch.device(training_hyperparameters['cuda_device'] if torch.cuda.is_available() else "cpu")

    n = configs['n']
    batch_size = configs['val_batch_size']

    # generate random points on unit sphere
    random_points = torch.randn(n, 512).to(device)
    random_points = random_points / random_points.norm(dim=-1, keepdim=True)
    # all random points have magnitude 1

    # get embeddings of validation dataset
    val_dataset = dataset_processor.val_dataset

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, collate_fn=dataset_processor.collate_fn, generator=torch.Generator().manual_seed(42))

    embeddings = []

    
    for batch in tqdm(val_dataloader):
        images, tokenized_captions = batch
        image_features = clip_model(images, tokenized_captions, return_all=True).image_embeds
        # # generate random image_features
        image_features = torch.randn_like(image_features)
        # normalize image features
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        embeddings.append(image_features)
    
    embeddings = torch.cat(embeddings, dim=0) # shape [dataset_size, 512]


    # find closest embedding for each random point
    closest_embeddings = []
    cosine_similarities = []
    for i in tqdm(range(n)):
        random_point = random_points[i]
        # closest_embedding = embeddings[torch.argmin(torch.cdist(embeddings, random_point.unsqueeze(0)), dim=0)]
        # find closest embeddings according to cosine similarity
        cosine_similarity = torch.matmul(embeddings, random_point.unsqueeze(-1)) # shape [dataset_size, 1]
        closest_embedding = embeddings[torch.argmax(cosine_similarity, dim=0)]
        closest_embeddings.append(closest_embedding)
        cosine_similarities.append(torch.max(cosine_similarity))
    
    closest_embeddings = torch.stack(closest_embeddings, dim=0) # shape [n, 512]
    cosine_similarities = torch.stack(cosine_similarities, dim=0) # shape [n, 1]

    # find distance between random point and closest embedding
    # distances = torch.cdist(random_points, closest_embeddings) # shape [n, 1]

    # find cosine similarity between random point and closest embedding
    # cosine_similarities = torch.matmul(random_points, closest_embeddings.unsqueeze(-1)) # shape [n, 1]

    # calculate average distance
    average_cosine_sim = torch.mean(cosine_similarities)

    max_cosine_sim = torch.max(cosine_similarities)
    min_cosine_sim = torch.min(cosine_similarities)

    print('max cosine sim ', max_cosine_sim)
    print('min cosine sim ', min_cosine_sim)

    return average_cosine_sim







def main():

    '''
    Init setups
    '''

    torch.manual_seed(42)
    random.seed(42)
    device = torch.device(training_hyperparameters['cuda_device'] if torch.cuda.is_available() else "cpu")
    dataset_processor = MSCOCOProcessor()

    '''
    Setting up CLIP model
    '''
    clip_model = HFClip().to(device)
    clip_model.eval()


    '''
    Loading clip checkpoint
    '''
    checkpoint_path = configs['checkpoint_path']
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # clip_model.load_state_dict(checkpoint['model_state_dict'])


    dataset_processor.print_dataset_stats()

    average_cosine_similarity = calculate_coverage(clip_model, dataset_processor)

    print('average cosine similarity ', average_cosine_similarity)




if __name__ == '__main__':
    main()