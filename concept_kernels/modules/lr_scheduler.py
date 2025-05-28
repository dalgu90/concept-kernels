import numpy as np

# Source: https://github.com/jhaochenz96/spectral_contrastive_learning/blob/main/optimizers/lr_scheduler.py

class LR_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch, constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr+0.5*(base_lr-final_lr)*(1+np.cos(np.pi*np.arange(decay_iter)/decay_iter))

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0
    def step(self):
        for param_group in self.optimizer.param_groups:
            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]
        self.iter += 1
        self.current_lr = lr
        return lr
    def get_lr(self):
        return self.current_lr

class LR_Scheduler_CosineLog4(object):
    def __init__(self, optimizer, num_epochs, base_lr, iter_per_epoch):
        self.base_lr = base_lr
        t = np.arange(num_epochs * iter_per_epoch) / (num_epochs * iter_per_epoch)
        lr_sched_value = 0.5 - 0.5 * np.cos(2 * np.pi * np.log2(1 + 15 * t))
        self.lr_schedule = base_lr * lr_sched_value
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0
    def step(self):
        self.current_lr = self.lr_schedule[self.iter]
        for param_group in self.optimizer.param_groups:
            lr = self.lr_schedule[self.iter]
            if 'lr_mult' in param_group:
                lr *= param_group['lr_mult']
            param_group['lr'] = lr
        self.iter += 1
        return lr
    def get_lr(self):
        return self.current_lr

