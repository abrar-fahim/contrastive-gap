'''
- Setup image encoder
- Setup text encoder
- use nonlinear projection layer
- Setup toy dataset
- Setup loss fn
- Setup training loop

- Train model on toy dataset using minibatches
- Test to see if model can get high cosine similarities between images and captions of same concept
- This works as sanity check to test algorithm in general

'''

'''
- Setup toy dataset, MSCOCO for now
'''

import sys
import os

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# add sibling directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clips.hf_clip import HFClip
from grad_cache_wrapper import GradCacheWrapper
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch
from src.utils import do_validation, collate_fn, get_checkpoint_path, init_stats_csv_file
import clip
import os
import torchvision.datasets as dset
import webdataset as wds
import random

from src.config import *
from dataset_processors.mscoco_processor import MSCOCOProcessor





def main():

    # set seed
    torch.manual_seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('device ', device)

    model, preprocess = clip.load(training_hyperparameters['openai_clip_model'], device=device)
    # model, preprocess = clip.load("ViT-B/16", device=device)

    dataset_processor = None

    if training_hyperparameters['dataset'] == ClipDatasets.MSCOCO:
        dataset_processor = MSCOCOProcessor()
    elif training_hyperparameters['dataset'] == ClipDatasets.WIT400:
        # dataset_processor = WIT400Processor()
        pass

    

    
    if training_hyperparameters['dataset'] == ClipDatasets.WIT400:
        def identity(x):
            return x['caption']
        
        # root_dir = '/Volumes/SanDisk Extreme SSD Media/UofA/Research/dataset/400m/laion400m-data/'
        root_dir = './datasets/400m/laion400m-data/'
        
        tar_count = 0
        corrupt_files = ['00000.tar','00002.tar', '00004.tar', '00007.tar', '00006.tar', '00008.tar', '00009.tar', '00010.tar', '00011.tar', '00012.tar', '00013.tar',  '00014.tar', '00015.tar']


        # count number of .tar files in root_dir
        for root, dirs, files in os.walk(root_dir):
            for filename in files:
                if filename.endswith('.tar') and filename not in corrupt_files:
                    tar_count += 1


        # setup 80/20 split
        train_tar_count = int(0.8 * tar_count)
        val_tar_count = tar_count - train_tar_count

        train_paths = []
        val_paths = []

        tar_index = 0

        for root, dirs, files in os.walk(root_dir): 
            for filename in files:
                if filename.endswith('.tar') and filename not in corrupt_files:
                    if tar_index < train_tar_count:
                        train_paths.append(os.path.join(root, filename))
                    else:
                        val_paths.append(os.path.join(root, filename))
                    tar_index += 1

        train_dataset = wds.WebDataset(train_paths).shuffle(1000, rng=random).decode("pill").to_tuple("jpg;png", "json").map_tuple(preprocess, identity).with_length(9000 * len(train_paths))

        val_dataset = wds.WebDataset(val_paths).shuffle(1000, rng=random).decode("pill").to_tuple("jpg;png", "json").map_tuple(preprocess, identity).with_length(9000 * len(val_paths))

        print()
        print('--- TRAIN DATASET STATS ---')
        print()


        print('Number of train tar files: ', len(train_paths))
        print('no of train samples: ', len(train_paths) * 9000)

        print()
        print('--- VAL DATASET STATS ---')
        print()


        print('Number of val tar files: ', len(val_paths))
        print('no of val samples: ', len(val_paths) * 9000)
        # train_dataset = wds.WebDataset('/Volumes/SanDisk Extreme SSD Media/UofA/Research/dataset/400m/laion400m-data/00001.tar').shuffle(1000).to_tuple("jpg;png", "json").map_tuple(preprocess, identity)


    # clip_model = MyClip().to(device)
    # clip_model = OpenAIClip().to(device)

    clip_model = HFClip().to(device)


    '''
    - Setup training loop
    '''

    # setup adamW optimizer

    optimizer = optim.AdamW(clip_model.parameters(), lr=training_hyperparameters['lr'], weight_decay=training_hyperparameters['weight_decay'])


    n_epochs = training_hyperparameters['n_epochs']

    clip_model.train()


    '''
    checkpointing stuff
    '''

    print('continuting from checkpoint ', training_hyperparameters['continue_from_checkpoint'])

    print('training from scratch ', training_hyperparameters['train_from_scratch'])

    checkpoint_path = get_checkpoint_path()


    i_loaded_from_checkpoint = False

    if training_hyperparameters['train_from_scratch']:
        '''
        By default clip model is initialized depending on selected_clip_model
        '''

        print()
        print('--- TRAINING FROM SCRATCH ---')
        print()
        clip_model.reset_weights_to_random()

    if os.path.exists(checkpoint_path) and training_hyperparameters['continue_from_checkpoint'] and training_hyperparameters['do_checkpointing']:

        print()
        print('--- CONTINUING FROM CHECKPOINT ---')
        print()


        # load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        clip_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        losses = checkpoint['losses']
        i = checkpoint['dataloader_enumerator_index']
        median_cosine_similarities = checkpoint['median_cosine_similarities']
        i_loaded_from_checkpoint = True

    else:
        '''
        Goes here if I dont need to load from checkpoint
        '''

        print()
        print('--- NOT LOADING FROM CHECKPOINT---')
        print()
        epoch = 0
        i = 0
        losses = []
        median_cosine_similarities = []

        clip_model.reset_weights_to_default() # because CLIP loads from latest checkpoint in init for inference

        if training_hyperparameters['use_small_trainloader']:
 

            if training_hyperparameters['dataset'] == ClipDatasets.WIT400:
                train_dataloader = DataLoader(train_dataset, batch_size=training_hyperparameters['small_train_loader_batch_size'], collate_fn=collate_fn, num_workers=0)

            '''
            display all subset indices images as small tiles
            '''
        else:

            if training_hyperparameters['dataset'] == ClipDatasets.WIT400:
                train_dataloader = DataLoader(train_dataset, batch_size=training_hyperparameters['batch_size'], collate_fn=collate_fn, num_workers=0)


    if training_hyperparameters['dataset'] == ClipDatasets.WIT400:

        # use val dataset defined at the beginning
        val_dataloader = DataLoader(val_dataset, batch_size=training_hyperparameters['validation_batch_size'], collate_fn=collate_fn, num_workers=0)
        


    dataset_processor.print_dataset_stats()

    '''
    create csv file
    '''

    if training_hyperparameters['save_losses'] and not training_hyperparameters['continue_from_checkpoint']:
        # only create new csv file if we're not continuing from checkpoint
        init_stats_csv_file(clip_model)


    # training loop
    while epoch < n_epochs:

        print(f'--- VALIDATION AT START OF EPOCH {epoch} ---')

        clip_model.eval()

        if training_hyperparameters['dataset'] == ClipDatasets.MSCOCO:
            do_validation(dataset_processor.val_dataset, dataset_processor.train_dataset, clip_model, index=i, epoch=epoch, captioning_model=False)
        elif training_hyperparameters['dataset'] == ClipDatasets.WIT400:
            do_validation(val_dataset, train_dataset, clip_model, index=i, epoch=epoch, captioning_model=False)
        # do_validation(train_val_dataloader, clip_model, index=i, captioning_model=False) # using training dataset for validation
        clip_model.train()


        running_loss = 0.0

        if not i_loaded_from_checkpoint:
            i = 0

        if training_hyperparameters['train_only_one_batch']:
            torch.random.manual_seed(42) # reset seed so that same batch is output everytime

        if training_hyperparameters['grad_cache']:
            clip_model_grad_cache = GradCacheWrapper(clip_model)

            clip_model_grad_cache.clip_model.train()

            cache_x = []
            cache_y = []
            closures_x = []
            closures_y = []

            for step, sub_batch in enumerate(dataset_processor.train_dataloader):  
                imgs, captions = sub_batch
                r_imgs, c_imgs = clip_model_grad_cache.get_image_projections(imgs)

                r_txts, c_txts = clip_model_grad_cache.get_text_projections(captions)

                # print progress in place
                # print('\rstep: ' + str(step), end='')

                # print progress every 5 steps
                if step % 5 == 0:
                    print('step: ', step)

                cache_x.append(r_imgs)
                cache_y.append(r_txts)
                closures_x.append(c_imgs)
                closures_y.append(c_txts)

                # print size of cache x
                # print('len(cache_x) ', len(cache_x))
                
                if (step + 1) % training_hyperparameters['grad_cache_multiplier'] == 0:

                    # print('cache x ', cache_x)
                    # print('cache y ', cache_y)

                    loss = clip_model_grad_cache.contrastive_loss(cache_x, cache_y)
                    # print loss
                    # print('loss ', loss)
                    
                    loss.backward()
                
                    # TEST THESE FOR LOOPS LATER 
                    for f, r in zip(closures_x, cache_x):
                        f(r)
                    for f, r in zip(closures_y, cache_y):
                        f(r)

                    cache_x = []
                    cache_y = []
                    closures_x = []
                    closures_y = []
                
                    optimizer.step()
                    # scaler.update()
                    optimizer.zero_grad()

                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, loss.item()))
                    if i % 10 == 0:
                        clip_model_grad_cache.clip_model.eval()
                        # do_validation(val_dataloader, clip_model_grad_cache.clip_model)
                        clip_model_grad_cache.clip_model.train()


                    

                    if i % 100 == 0 and training_hyperparameters['do_checkpointing']:
                        checkpoint_to_save = {
                            'epoch': epoch,
                            'model_state_dict': clip_model_grad_cache.clip_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'losses': losses,
                            # 'train_dataloader': dataloader,
                            'dataloader_enumerator_index': i,
                            'median_cosine_similarities': median_cosine_similarities
                            }
                        
                        print()
                        print('saving checkpoint')
                        print()
                        torch.save(checkpoint_to_save, get_checkpoint_path())
                    
                    i += 1

                    if training_hyperparameters['train_only_one_batch']:
                        break

        else:

            for (imgs, captions) in dataset_processor.train_dataloader:

                if training_hyperparameters['max_steps'] is not None and i + 1 >= training_hyperparameters['max_steps']:
                    break

                # print('img ', img)
                # print('caption ', caption)

                # evaluate model
                clip_model.eval()
                
                
                if i % 10 == 0:
                    if training_hyperparameters['dataset'] == ClipDatasets.MSCOCO:
                        do_validation(dataset_processor.val_dataset, dataset_processor.train_dataset, clip_model, index=i, epoch=epoch, captioning_model=False)
                    elif training_hyperparameters['dataset'] == ClipDatasets.WIT400:
                        do_validation(val_dataset, train_dataset, clip_model, index=i, epoch=epoch, captioning_model=False)
                    pass
                    

                clip_model.train()  

                # zero the parameter gradients
                optimizer.zero_grad()

                # caption WAS a list of tuples, where first tuple corresponds to first captions of all the images in the batch

                # caption is now a list of 64 strings 
                # forward + backward + optimize
                _, _, loss = clip_model(imgs, captions, output_loss=True)
                # loss = clip_loss(*outputs)
                print('loss ', loss )
                loss.backward()

                optimizer.step()

                # print statistics
                running_loss += loss.item()

                losses.append(loss.item())
                # if i % 2 == 1:    # print every 2 mini-batches
                print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 1))
                running_loss = 0.0


                # save model 
                if i % 10 == 0 and training_hyperparameters['do_checkpointing']:
                    checkpoint_to_save = {
                        'epoch': epoch,
                        'model_state_dict': clip_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'losses': losses,
                        # 'train_dataloader': dataloader,
                        'dataloader_enumerator_index': i,
                        'median_cosine_similarities': median_cosine_similarities
                        }
                    # torch.save(checkpoint_to_save, training_hyperparameters['model_path'].split(".")[0] + str(epoch) + '_' + str(i) + '.pt')
                    print()
                    print('saving checkpoint')
                    print()
                    torch.save(checkpoint_to_save, get_checkpoint_path())
                i += 1

                if training_hyperparameters['train_only_one_batch']:
                    break
        
        i_loaded_from_checkpoint = False
        epoch +=1

if __name__ == '__main__':
    main()