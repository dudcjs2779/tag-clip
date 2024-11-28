import os
import re
import random
import torch
import numpy as np
import pandas as pd
import gc

import math
import time
from tqdm import tqdm
from datetime import datetime

from open_clip_train.precision import get_autocast
from open_clip_train.train import AverageMeter, backward
from open_clip import get_input_dtype
from open_clip_train.file_utils import pt_load
from open_clip.loss import ClipLoss

try:
    import wandb
except ImportError:
    wandb = None

try:
    import faiss
except ImportError:
    faiss = None

import torch.utils.tensorboard as tensorboard
from torch.cuda.amp import GradScaler

from bitsandbytes.optim import AdamW8bit

from tag_clip.utils.params import setup_parser
from tag_clip.utils.basic_utils import read_config
from tag_clip.utils.optim_utils import make_lwld_clip_params_group
from tag_clip.utils.scheduler import GroupCosineLR
from tag_clip.utils.loss import MultiLabelClipLoss, TagClipLoss
from tag_clip.utils.models import create_custom_model
from tag_clip.utils.dataset import make_dataset
tqdm.pandas()

def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model
    
def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def load_random_state(checkpoint):
    torch.random.set_rng_state(checkpoint['torch_rng_state'])
    if torch.cuda.is_available() and checkpoint['cuda_rng_state'] is not None: 
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
    np.random.set_state(checkpoint['numpy_rng_state'])
    random.setstate(checkpoint['random_state'])

def extract_number(file_name):
    match = re.search(r'epoch_(\d+)', file_name)
    return int(match.group(1)) if match else float('inf')

def train_one_epoch(model, device, dataloader, loss, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features, accum_selected_gts, accum_real_gts = [], [], {}, [], []

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    pbar = tqdm(total=num_batches_per_epoch, desc=f"Training Epoch:{epoch+1}")
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        scheduler(step)
        images, texts, selected_gts, real_gts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                model_out = model(images, texts)
                logit_scale = model_out["logit_scale"]

                losses = loss(**model_out, output_dict=True) if isinstance(loss, ClipLoss) else loss(**model_out, sel_gts=selected_gts, gts=real_gts, output_dict=True)
                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)
                accum_selected_gts.append(selected_gts)
                accum_real_gts.append(real_gts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()

            selected_gts = [gt for gts in accum_selected_gts for gt in gts]
            real_gts = [gt for gts in accum_real_gts for gt in gts]
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True) if isinstance(loss, ClipLoss) else loss(**inputs, **inputs_no_accum, sel_gts=selected_gts, gts=real_gts, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features, accum_selected_gts, accum_real_gts = [], [], {}, [], []

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1

        batch_size = len(images)
        for key, val in losses.items():
            if key not in losses_m:
                losses_m[key] = AverageMeter()
            losses_m[key].update(val.item(), batch_size)

        logit_scale_scalar = logit_scale.item()
        log_data = {
            "data_time": data_time_m.val,
            "batch_time": batch_time_m.val,
            "scale": logit_scale_scalar,
            "image_lr": optimizer.param_groups[args.img_group_idx]["lr"],
            "text_lr": optimizer.param_groups[args.text_group_idx]["lr"],
            'contrastive_avg_loss': losses_m['loss'].avg,
            'total_avg_loss': losses_m['loss'].avg,
        }
        log_data.update({name:val.val for name,val in losses_m.items()})
        log_data = {"train/" + name: val for name, val in log_data.items()}

        # if args.debug:
        #     for i, group in enumerate(optimizer.param_groups):
        #         key = f"lr/optim_group{i}"
        #         log_data[key] = group['lr']

        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, step)

        if args.wandb:
            assert wandb is not None, 'Please install wandb.'
            log_data['step'] = step  # for backwards compatibility
            args.step = step
            wandb.log(log_data, step=step)

        
        pbar.postfix = f"Loss avg:{losses_m['loss'].avg:.5f}, Loss batch:{losses_m['loss'].val:.5f}" 
        # pbar.postfix = f"Loss avg:{losses_m['loss'].avg:.5f}, Loss batch:{losses_m['loss'].val:.5f}, gpu: {torch.cuda.memory_allocated()}, max_gpu: {torch.cuda.max_memory_allocated()}" 
        pbar.update(1)
    pbar.close()

def evaluate(model, device, valid_loader, loss, epoch, step, tb_writer):
    model.eval()
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    all_image_features = []
    all_text_features = []

    with torch.inference_mode():
        pbar = tqdm(total=len(valid_loader), desc=f"Validation Current Epoch:{epoch}")
        for i, batch in enumerate(valid_loader):
            images, texts, selected_gts, real_gts, _ = batch
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)

            with autocast():
                model_out = model(images, texts)
                image_features = model_out["image_features"]
                text_features = model_out["text_features"]
                logit_scale = model_out["logit_scale"]

                all_image_features.append(image_features.detach().cpu())
                all_text_features.append(text_features.detach().cpu())

            # pbar.postfix = f"gpu: {torch.cuda.memory_allocated()}, max_gpu: {torch.cuda.max_memory_allocated()}" 
            pbar.update(1)
        pbar.close()

        all_image_features = torch.cat(all_image_features, dim=0)
        all_text_features = torch.cat(all_text_features, dim=0)

        total_loss = loss(image_features, text_features, logit_scale) if isinstance(loss, ClipLoss) else loss(image_features, text_features, logit_scale, sel_gts=selected_gts, gts=real_gts)
        print(f"Validation Loss: {total_loss:.8f}")

        log_data = {
            "loss": total_loss,
        }

        if args.recall:
            R1, R5, R10 = compute_text_to_image_recall(all_image_features, all_text_features)
            print(f"Text to Image R@1: {R1}, R@5: {R5}, R@10: {R10}")
            log_data['R@1'] = R1
            log_data['R@5'] = R5
            log_data['R@10'] = R10
            
        log_data = {"valid/" + name: val for name, val in log_data.items()}

        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        if args.wandb:
            assert wandb is not None, 'Please install wandb.'
            log_data['epoch'] = epoch  # for backwards compatibility
            wandb.log(log_data, step=step)

def compute_text_to_image_recall(all_image_features, all_text_features):
    dim = all_text_features.shape[-1]
    image_index = faiss.IndexFlat(dim, faiss.METRIC_INNER_PRODUCT)

    embeddings = all_image_features.numpy().astype(np.float32)
    del all_image_features
    faiss.normalize_L2(embeddings)
    image_index.add(embeddings)

    query = all_text_features.numpy().astype(np.float32)
    del all_text_features
    faiss.normalize_L2(query)
    _, all_topk_idxs = image_index.search(query, 10)

    top1 = top5 = top10 = 0
    for i, idxs in enumerate(all_topk_idxs):
        if i in idxs[:1]:
            top1 += 1

        if i in idxs[:5]:
            top5 += 1

        if i in idxs[:10]:
            top10 += 1

    R1 = top1 / len(query)
    R5 = top5 / len(query)
    R10 = top10 / len(query)
    return R1, R5, R10


def main(args):
    if args.debug:
        args.workers = 0

    t_now = datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    log_path = "./logs"
    output_path = os.path.join(log_path, f"{t_now}_{args.model_name}")
    checkpoint_path = os.path.join(output_path, "checkpoint")
    os.makedirs(checkpoint_path, exist_ok=True)

    args.wandb = 'wandb' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to

    if args.tensorboard:
        tensorboard_dir = os.path.join(output_path, "tensorboard")
        if not os.path.exists(tensorboard_dir): os.makedirs(tensorboard_dir, exist_ok=True)

    random_seed(args.seed, 0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df = pd.read_parquet(args.train)

    print("\nModel create")
    model = create_custom_model(args.model_name, args.precision, args.pretrained, device)

    train_loader, valid_loader = make_dataset(df, args)
    del df

    param_groups = make_lwld_clip_params_group(
        model, args.image_encoder_lr, 1.0, args.wd, 
        image_lr=args.image_encoder_lr, text_lr=args.text_encoder_lr, 
        img_decay=args.img_lwld_decay, text_decay=args.text_lwld_decay
        )

    optimizer = AdamW8bit(
        param_groups,
        lr=args.image_encoder_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )
    args.img_group_idx = next(i for i, group in enumerate(optimizer.param_groups) if group['lr'] == args.image_encoder_lr)
    args.text_group_idx = next(i for i, group in enumerate(optimizer.param_groups) if group['lr'] == args.text_encoder_lr)

    scaler = GradScaler() if args.precision == "amp" else None

    total_steps = (train_loader.num_batches // args.accum_freq) * args.epochs
    scheduler = GroupCosineLR(optimizer, args.image_encoder_lr, args.warmup, total_steps)

    start_epoch = 0
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            load_random_state(checkpoint)
            model.load_state_dict(checkpoint["state_dict"])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)

    if args.loss_fn in "ML":
        loss = MultiLabelClipLoss()
    elif args.loss_fn in "KL":
        loss = TagClipLoss()
    else:
        loss = ClipLoss()

    print("\n## Config info:")
    for k, v in vars(args).items(): print(f"{k}: {v}")
    print()

    writer = None
    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        wandb.init(
            dir=log_path,
            project=args.wandb_project_name,
            name=None,
            id=args.run_id,
            notes=None,
            tags=[],
            resume='must' if args.resume or args.run_id else None,
            config=vars(args),
        )

    if args.tensorboard:
        writer = tensorboard.SummaryWriter(tensorboard_dir)

    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(model, device, train_loader, loss, epoch, optimizer, scaler, scheduler, args, tb_writer=writer)
        completed_epoch = epoch + 1

        if args.valid:
            evaluate(model, device, valid_loader, loss, completed_epoch, args.step, tb_writer=writer)

        checkpoint_dict = {
            "epoch": completed_epoch,
            "name": args.model_name,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            'torch_rng_state': torch.random.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'numpy_rng_state': np.random.get_state(),
            'random_state': random.getstate()
        }
        if scaler is not None:
            checkpoint_dict["scaler"] = scaler.state_dict()

        if completed_epoch == args.epochs or (
            args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
        ):
            torch.save(
                checkpoint_dict,
                os.path.join(checkpoint_path, f"epoch_{completed_epoch}.pt"),
            )

        checkpoint_files = os.listdir(checkpoint_path)
        if not args.max_save_count: args.max_save_count = args.epochs 
        if len(checkpoint_files) > args.max_save_count:
            checkpoint_files.sort(key=extract_number)
            os.remove(os.path.join(checkpoint_path, checkpoint_files[0]))

        del checkpoint_dict
        torch.cuda.empty_cache()
        gc.collect()
            
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args = read_config(args, parser)
    main(args)
