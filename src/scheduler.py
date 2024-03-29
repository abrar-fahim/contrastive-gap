import numpy as np    



def cosine_scheduler(optimizer, base_lr, num_warmup_steps, total_steps):
    '''
    from https://github.dev/goel-shashank/CyCLIP/tree/main/src
    Equivalent to cosine annealing lr in pytorch (https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)
    
    '''
    def _scheduler(current_step):
        if(current_step < num_warmup_steps):
            lr = base_lr * (current_step + 1) / num_warmup_steps
        else:
            n = current_step - num_warmup_steps
            d = total_steps - num_warmup_steps
            lr = 0.5 * (1 + np.cos(np.pi * n / d)) * base_lr

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
            
    return _scheduler