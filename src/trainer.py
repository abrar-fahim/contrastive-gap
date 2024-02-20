'''
Class to wrap one training step
Mainly designed to abstract away grad cache mechanics from main training loop
'''
from grad_cache_wrapper import GradCacheWrapper
from config import *
import torch
from utils import get_checkpoint_path, do_validation
from abc import ABC, abstractmethod

# import torch.multiprocessing as mp
# import pathos.multiprocessing as mp

# def top_validate(trainer_parent, clip_model, train_dataloader, optimizer, epoch, i):
#     trainer_parent.save_checkpoint_and_validate(clip_model, train_dataloader, optimizer, epoch, i)






class TrainerParent(ABC):
    def __init__(self, train_dataset, val_dataset, wandb) -> None:
        '''
        train_dataset and val_dataset needed to do validation
        '''
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.wandb = wandb

        self.val_processes = []
        pass

    def __init__(self, dataset_processor, wandb) -> None:
        '''
        dataset_processor needed to do validation
        '''
        self.train_dataset = dataset_processor.train_dataset
        self.val_dataset = dataset_processor.val_dataset
        self.dataset_processor = dataset_processor
        self.wandb = wandb
        self.val_processes = []
        # mp.set_start_method('forkserver')
        pass

    @abstractmethod
    def train_one_epoch(self, clip_model, train_dataloader, optimizer, i=0, epoch=0, save_every=10):
        pass

    def save_checkpoint_and_validate(self, clip_model, train_dataloader, optimizer, epoch, i, val_dataset_processor=None):

        clip_model.eval()
        print()
        print('--- VALIDATING ---')
        print()

        do_validation(self.dataset_processor, clip_model, i, epoch, captioning_model=False, wandb=self.wandb, val_dataset_processor=val_dataset_processor)

        clip_model.train()


        checkpoint_to_save = {
            'epoch': epoch,
            'model_state_dict': clip_model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            # 'train_dataloader': train_dataloader,
            'dataloader_enumerator_index': i,
            }
        print()
        print('saving checkpoint')
        print()
        torch.save(checkpoint_to_save, get_checkpoint_path())


class GradCacheTrainer(TrainerParent):
    '''
    This handles operations of one epoch
    This class does NOT care about maintaining states of dataloders, optimizers, schedulers, etc.
    '''
    def __init__(self, train_dataset, val_dataset, wandb) -> None:
        super().__init__(train_dataset, val_dataset, wandb)

    def __init__(self, dataset_processor, wandb) -> None:
        super().__init__(dataset_processor, wandb)
    

    def train_one_epoch(self, clip_model, train_dataloader, optimizer, i=0, epoch=0, save_every=10):
        '''
        i is parameter because we might be starting in the middle of an epoch from a checkpoint
        epoch is a parameter as we dont know this, since this class doesnt maintain global training state
        '''

        if training_hyperparameters['train_only_one_batch']:
            torch.random.manual_seed(42) # reset seed so that same batch is output everytime

        clip_model.train()
        
        clip_model_grad_cache = GradCacheWrapper(clip_model)

        cache_x = []
        cache_y = []
        closures_x = []
        closures_y = []

        step = i # this is the number of times we've called optimizer.step()  

        optimizer.zero_grad()

        for substep, sub_batch in enumerate(train_dataloader):
            imgs, captions = sub_batch

            if self.use_grad_cache:
                r_imgs, c_imgs = clip_model_grad_cache.get_image_projections(imgs)
                r_txts, c_txts = clip_model_grad_cache.get_text_projections(captions)

                # print progress every 5 steps
                if substep % 5 == 0:
                    print('substep: ', substep)

                cache_x.append(r_imgs)
                cache_y.append(r_txts)
                closures_x.append(c_imgs)
                closures_y.append(c_txts)

                if (substep + 1) % training_hyperparameters['grad_cache_multiplier'] == 0:


                    loss = clip_model_grad_cache.contrastive_loss(cache_x, cache_y)
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
                    step += 1
                    # scaler.update()
                    optimizer.zero_grad()

                    print('[%d, %5d] loss: %.3f' % (epoch + 1, step + 1, loss.item()))

                    if step % save_every == 0 and training_hyperparameters['do_checkpointing'] and epoch % 5 == 0: # save every 5 epochs only

                        # for old_p in self.val_processes:
                        #     old_p.join()

                        # p = mp.Process(target=self.save_checkpoint_and_validate, args=(clip_model, train_dataloader, optimizer, epoch, step))
                        self.save_checkpoint_and_validate(clip_model, train_dataloader, optimizer, epoch, step)
                        # p.start()
                        # self.val_processes.append(p)


                    if training_hyperparameters['train_only_one_batch']:
                        break

                    if training_hyperparameters['max_steps'] is not None and step >= training_hyperparameters['max_steps']:
                        break

                    step += 1


class Trainer(TrainerParent):

    def __init__(self, train_dataset, val_dataset, wandb) -> None:
        super().__init__(train_dataset, val_dataset, wandb)

    def __init__(self, dataset_processor, wandb) -> None:
        super().__init__(dataset_processor, wandb)

    def train_one_epoch(self, clip_model, train_dataloader, optimizer, i=0, epoch=0, save_every=10, val_dataset_processor=None):
        '''
        i is parameter because we might be starting in the middle of an epoch from a checkpoint
        epoch is a parameter as we dont know this, since this class doesnt maintain global training state
        '''

        if training_hyperparameters['train_only_one_batch']:
            torch.random.manual_seed(42) # reset seed so that same batch is output everytime

        clip_model.train()

        for (imgs, captions) in train_dataloader:

            optimizer.zero_grad()

            # captions is a list of batch_size strings 

            _, _, loss = clip_model(imgs, captions, output_loss=True)

            loss.backward()

            
            optimizer.step()

            # print statistics
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))

            del loss


            

            if i % save_every == 0 and training_hyperparameters['do_checkpointing']:
                # do this in another thread

                # for old_p in self.val_processes:
                #     print('waiting')
                #     old_p.join()

                # print('waiting done. clearing val processes')
                # self.val_processes = []

                # p = mp.Process(target=self.save_checkpoint_and_validate, args=(clip_model, train_dataloader, optimizer, epoch, i))
                # # p = mp.Process(target=top_validate, args=(self, clip_model, train_dataloader, optimizer, epoch, i))
                # # self.save_checkpoint_and_validate(clip_model, train_dataloader, optimizer, epoch, step)
                # p.start()
                # self.val_processes.append(p)
                # print('joining')
                # p.join()
                # print('joining done')



                self.save_checkpoint_and_validate(clip_model, train_dataloader, optimizer, epoch, i, val_dataset_processor=val_dataset_processor)
                pass
            
            i += 1

            # if training_hyperparameters['train_only_one_batch']:
            #     break

            if training_hyperparameters['max_steps'] is not None and i >= training_hyperparameters['max_steps']:
                break

            




        



