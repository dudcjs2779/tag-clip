import re
from collections import defaultdict

image_block_pattern = "trunk\.blocks\.(\d+)"
text_block_pattern = "transformer\.resblocks\.(\d+)"
exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n

def clip_layer_wise_lr_decay(nameparams, pattern, initial_lr, decay):
    for p in nameparams:
        if not p['params'].requires_grad: continue
        block_match = re.search(pattern, p['name'])
        if block_match:
            block_num = int(block_match.group(1))
            p['lr'] = initial_lr * decay**block_num
        else:
            p['lr'] = initial_lr

    return nameparams

def exclude_weight_decay(nameparams, weight_decay):
    for p in nameparams:
        if exclude(p['name'], p['params']) and p['params'].requires_grad:
            p['weight_decay'] = 0
        else:
            p['weight_decay'] = weight_decay

    return nameparams

def bind_group(nameparams, base_lr, base_wd):
    grouped_params = defaultdict(list)
    
    for p in nameparams:
        p['lr'] = p.get('lr', base_lr)
        p['weight_decay'] = p.get('weight_decay', base_wd)
        
        key = (p['lr'], p['weight_decay'])
        
        grouped_params[key].append(p['params'])
        
    param_groups = [{'params': params, 'lr': lr, 'weight_decay': wd} 
                    for (lr, wd), params in grouped_params.items()]
    
    return param_groups

def make_lwld_clip_params_group(model, base_lr, decay, weight_decay, image_lr=None, text_lr=None, img_decay=None, text_decay=None):
    image_lr = image_lr or base_lr
    text_lr = text_lr or base_lr
    img_decay = img_decay or decay
    text_decay = text_decay or decay

    grad_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    rest_params = [{"name": name, "params": param, 'lr': base_lr} for name, param in grad_params if 'visual' not in name and 'text' not in name]
    visual_params = [{"name": name, "params": param} for name, param in grad_params if 'visual' in name]
    visual_params = clip_layer_wise_lr_decay(visual_params, image_block_pattern, image_lr, img_decay)
    text_params = [{"name": name, "params": param} for name, param in grad_params if 'text' in name]
    text_params = clip_layer_wise_lr_decay(text_params, text_block_pattern, text_lr, text_decay)
    all_params = rest_params + visual_params + text_params

    all_params = exclude_weight_decay(all_params, weight_decay)

    param_groups = bind_group(all_params, base_lr, weight_decay)

    return param_groups