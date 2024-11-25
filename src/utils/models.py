import os
import torch
import timm
import gc

from open_clip.pretrained import get_pretrained_cfg, download_pretrained
from open_clip.factory import create_model, get_model_config, load_checkpoint


def create_custom_model(model_name, precision, pretrained, device):
    model_kwargs = get_model_config(model_name)
    model_kwargs.pop('custom_text')

    model = create_model(
        model_name = model_name,
        precision = precision,
        output_dict=True,
        device = 'cpu',
        **model_kwargs
    )
    timm_kwargs = {"dynamic_img_size": True}
    trunk = timm.create_model(
        model_name=model_kwargs['vision_cfg']['timm_model_name'],
        num_classes=model.visual.trunk.num_classes,
        global_pool=model.visual.trunk.global_pool,
        pretrained=False,
        **timm_kwargs,
    )
    model.visual.trunk = trunk
    del trunk
    model.to(device=device)

    if pretrained:
        pretrained_cfg = get_pretrained_cfg(model_name, pretrained)
        if pretrained_cfg:
            checkpoint_path = download_pretrained(pretrained_cfg, cache_dir=None)
        elif os.path.exists(pretrained):
            checkpoint_path = pretrained

        assert checkpoint_path, f"Pretrained weights {pretrained} not found for model {model_name}"
        load_checkpoint(model, checkpoint_path) # load weight

    gc.collect()
    torch.cuda.empty_cache()
    return model