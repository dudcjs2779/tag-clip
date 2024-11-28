import numpy as np

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def group_cosine_lr(optimizer, base_lrs, warmup_length, steps):
        def _lr_adjuster(step):
            lrs = [0] * len(base_lrs)
            for i, group in enumerate(optimizer.param_groups):
                base_lr = base_lrs[i]
                if step < warmup_length:
                    lr = _warmup_lr(base_lr, warmup_length, step)
                else:
                    e = step - warmup_length
                    es = steps - warmup_length
                    lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
                group['lr'] = lr
                lrs[i] = lr
            return lrs

        return _lr_adjuster

class GroupCosineLR():
    def __init__(self, optimizer, default_lr, warmup_length, steps):
        self.base_lrs =  [group.get("lr", default_lr) for group in optimizer.param_groups]
        self.scheduler = group_cosine_lr(optimizer, self.base_lrs, warmup_length, steps)

    def __call__(self, step):
        return self.scheduler(step)

    