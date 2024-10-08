'''
Class to wrap one training step
Mainly designed to abstract away grad cache mechanics from main training loop
'''
from grad_cache_wrapper import GradCacheWrapper

import config as config
import torch
from utils import get_checkpoint_path
# from validate import do_validation
from evaluator import Evaluator
from abc import ABC, abstractmethod
from torch.autograd.profiler import record_function
import importlib
from torch.cuda.amp import GradScaler


from itertools import repeat

import wandb


from tqdm import tqdm

import sys
import os

from dataset_processors.mscoco_processor import MSCOCOProcessor

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# add sibling directory to path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # def

from dataset_processors.mscoco_processor import RepeatSampler

from clips.hf_clip import HFClipOutput, HFClip
from config import config_cuda_device

# import torch.multiprocessing as mp
# import pathos.multiprocessing as mp

# def top_validate(trainer_parent, clip_model, train_dataloader, optimizer, epoch, i):
#     trainer_parent.save_checkpoint_and_validate(clip_model, train_dataloader, optimizer, epoch, i)






class TrainerParent(ABC):
    def __init__(self, train_dataset, val_dataset) -> None:

        '''
        train_dataset and val_dataset needed to do validation
        '''
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = config_cuda_device if torch.cuda.is_available() else "cpu"

        self.val_processes = []
        pass

    def __init__(self, dataset_processor: MSCOCOProcessor, evaluator: Evaluator) -> None:
        '''
        dataset_processor needed to do validation
        '''

        self.dataset_processor: MSCOCOProcessor = dataset_processor
        self.val_processes = []
        self.evaluator = evaluator

        self.device = config_cuda_device if torch.cuda.is_available() else "cpu"
        # mp.set_start_method('forkserver')
        pass

    def calculateW(self, clip_model: HFClip):
        '''
        Calculate W matrix that aligns two modalities at init
        will align ENCODER2 modalities to encoder1 modalities
        Iterates through train_dataset
        '''
        

        # if wandb.config['use_small_trainloader'] and wandb.config['small_train_loader_dataset_size'] <= 2048:
        #     n = 2048
        n = 2048

        e1_embeds = torch.empty((0, wandb.config['clip_projection_dim']), device=self.device) # hardcoding this for now FIX LATER MAYBE
        e2_embeds = torch.empty((0, wandb.config['clip_projection_dim']), device=self.device)

        with torch.no_grad():
            clip_model.eval()

            for (e1_inputs, e2_inputs) in tqdm(self.dataset_processor.train_dataloader):
                outputs: HFClipOutput = clip_model(e1_inputs, e2_inputs, output_hidden_states=False, output_loss=False, return_all=True)

                e1_embeds = torch.cat((e1_embeds, outputs.image_embeds), dim=0) # normalized 
                e2_embeds = torch.cat((e2_embeds, outputs.text_embeds), dim=0) # normalized

                del outputs

                if e1_embeds.shape[0] >= n:
                    break

            '''
            translate e1 embeds depending on gap required at init
            '''

            e1_to_e2_vector = e2_embeds.mean(dim=0) - e1_embeds.mean(dim=0)

            # print('e1 to e2 vector ', e1_to_e2_vector)

            # gap_direction = gap_direction / gap_direction.norm()

            print('W gap in trainer ', wandb.config['W_layer_gap'])

            e1_embeds_phantom = e1_embeds + e1_to_e2_vector * wandb.config['W_layer_gap'] # translating e1 to be CLOSER TO E2
            # so that, when I map e2 to phantom e1, W maps e2 to be FURTHER AWAY from original e1, so modality gap is higher

            e1_embeds_phantom = e1_embeds_phantom / e1_embeds_phantom.norm(dim=-1, keepdim=True)



            # calculate W
            '''
            def findW(x, y):
                yx = y.T @ x

                u, s, v = torch.svd(yx)

                w = u @ v.T

                return w

            ...
            x' = x @ W.T
            '''
            e2e1 = torch.matmul(e1_embeds_phantom.T, e2_embeds) # Here, e2 is x, e1 is y

            u, s, v = torch.svd(e2e1)

            W = torch.matmul(u, v.T)

            # I'll be aligning e2 

            del e1_embeds, e2_embeds, e1_embeds_phantom

            return W

    @abstractmethod
    def train_one_epoch(self, clip_model, train_dataloader, optimizer, i=0, epoch=0, save_every=10):
        pass

    def save_checkpoint_and_validate(self, clip_model, epoch, i):

        clip_model.eval()
        print()
        print('--- VALIDATING ---')
        print()


        # run evaluate_model twice, one for train data another for val data
        self.evaluator.evaluate_model(clip_model, epoch=epoch, index=i, is_train_data=False)
        self.evaluator.evaluate_model(clip_model, epoch=epoch, index=i, is_train_data=True)


        # do_validation(self.dataset_processor, clip_model, i, epoch, captioning_model=False, wandb=self.wandb, val_dataset_processor=val_dataset_processor)

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
    def __init__(self, train_dataset, val_dataset) -> None:
        super().__init__(train_dataset, val_dataset)

    def __init__(self, dataset_processor, evaluator:Evaluator) -> None:
        super().__init__(dataset_processor, evaluator)
    

    def train_one_epoch(self, clip_model, optimizer, scaler: GradScaler=None, scheduler =None,i=0, epoch=0, save_every=10) -> int:
        '''
        i is parameter because we might be starting in the middle of an epoch from a checkpoint
        epoch is a parameter as we dont know this, since this class doesnt maintain global training state
        '''

        if wandb.config['train_only_one_batch']:
            torch.random.manual_seed(42) # reset seed so that same batch is output everytime

        clip_model.train()
        
        clip_model_grad_cache = GradCacheWrapper(clip_model)

        cache_x = []
        cache_y = []
        closures_x = []
        closures_y = []

        step = i # this is the number of times we've called optimizer.step()  

        optimizer.zero_grad()

        for substep, sub_batch in enumerate(self.dataset_processor.train_dataloader):
            imgs, captions = sub_batch

            if imgs == None:
                # happens when OSError in conceptual captions dataloader
                continue

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

                if (substep + 1) % wandb.config['grad_cache_multiplier'] == 0:


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

                    if step % save_every == 0 and wandb.config['do_checkpointing']: 

                        # for old_p in self.val_processes:
                        #     old_p.join()

                        # p = mp.Process(target=self.save_checkpoint_and_validate, args=(clip_model, train_dataloader, optimizer, epoch, step))
                        self.save_checkpoint_and_validate(clip_model, epoch, step)
                        # p.start()
                        # self.val_processes.append(p)


                    if wandb.config['train_only_one_batch']:
                        break

                    if wandb.config['max_steps'] > 0 and step >= wandb.config['max_steps']:
                        break

                    step += 1

        return epoch


class Trainer(TrainerParent):

    def __init__(self, train_dataset, val_dataset) -> None:
        super().__init__(train_dataset, val_dataset)

    def __init__(self, dataset_processor, evaluator: Evaluator) -> None:
        super().__init__(dataset_processor, evaluator)

    
    def train_one_epoch(self, clip_model, optimizer, scaler: GradScaler=None, scheduler =None, i=0, epoch=0, save_every=10) -> int:
        '''
        i is parameter because we might be starting in the middle of an epoch from a checkpoint
        epoch is a parameter as we dont know this, since this class doesnt maintain global training state
        '''
   

        if wandb.config['train_only_one_batch']:
            torch.random.manual_seed(42) # reset seed so that same batch is output everytime

        clip_model.train()

        
        for (imgs, captions) in self.dataset_processor.train_dataloader:

            if imgs == None:
                # happens when OSError in conceptual captions dataloader
                continue

            with torch.no_grad():
                clip_model.clamp_logit_scale()



            step = self.dataset_processor.get_num_batches() * epoch + i


            



            optimizer.zero_grad()

            if step % save_every == 0 and wandb.config['do_checkpointing']:
                self.save_checkpoint_and_validate(clip_model, epoch, i)
                pass

            # captions is a list of batch_size strings 
            logits_per_image, logits_per_text, loss = clip_model(imgs, captions, output_loss=True)
            del logits_per_image
            del logits_per_text


            scaler.scale(loss).backward()
            # loss.backward()
            scaler.step(optimizer)
            # optimizer.step()
            scaler.update()

            

            # print statistics
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))

            del loss

            # del imgs, captions
            
            i += 1

            # if wandb.config['train_only_one_batch']:
            #     break

            if wandb.config['max_steps'] > 0 and i >= wandb.config['max_steps']:
                break

            if step % wandb.config['schedule_every'] == 0:
                if wandb.config['use_scheduler'] == 'COSINE':
                    scheduler(step)
                elif wandb.config['use_scheduler'] == 'EXP':
                    scheduler.step()

            if type(self.dataset_processor.train_dataloader.batch_sampler) == RepeatSampler and i >= self.dataset_processor.get_num_batches():
                i = 0
                epoch += 1

                # if wandb.config['use_scheduler'] == 'COSINE':

                #     scheduler(step)
                # elif wandb.config['use_scheduler'] == 'EXP':

                #     scheduler.step()

                    
                
                if epoch >= wandb.config['n_epochs']:
                    break

            



            torch.cuda.empty_cache()

        return epoch



            




        



