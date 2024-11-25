import os
import re
import cv2
import pandas as pd
import torch
import numpy as np

from PIL import Image
from itertools import groupby
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler

from open_clip import get_tokenizer

from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.geometric.functional import resize as A_resize
from albumentations.augmentations.crops.functional import get_center_crop_coords, crop as A_crop

from utils.data import make_tag_weights, load_cache_images, cache_process, make_frames, get_resize_and_frame_size

def make_dataset(df, args) -> tuple[DataLoader, DataLoader]:
    # mean = (0.48145466, 0.4578275, 0.40821073)
    # std = (0.26862954, 0.26130258, 0.27577711)
    mean = (0.68363303, 0.62883478, 0.63107163)
    std = (0.31656915, 0.31771758, 0.31206703)
    img_size=args.image_size[0]    #336, 448

    if args.dynamic_img_size:
        print(f"\n## Frame arguments: \nimage_size: {args.image_size} \npatch_size: {args.patch_size} \nstep_size: {args.step_size} \nmin_size: {args.min_size} \nmax_size: {args.max_size} \n")
        frames = make_frames(pixel_w=args.image_size[0], pixel_h=args.image_size[1], patch_size=args.patch_size, step_size=args.step_size, min_size=args.min_size, max_size=args.max_size)
        frame_aspect_ratios = [f[0] / f[1] for f in frames]

        frames = np.array(frames)
        frame_aspect_ratios = np.array(frame_aspect_ratios)

        df["resize_size"], df["frame"] = zip(*df["image_path"].progress_apply(
            lambda path: get_resize_and_frame_size(path, frames, frame_aspect_ratios))
            )
        
        print("\n## The number of images each frame contains.")
        print(df.groupby("frame").size(), "\n")
        
        transforms = [
            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2(),
        ]
    else:
        transforms = [
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255]), # padding
            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2(),
        ]

    transforms = A.Compose(transforms)
    
    if args.valid:
        if args.split_by_class:
            df['class'] = df['image_path'].apply(lambda path: os.path.basename(os.path.dirname(path)))
            train_df, valid_df = train_test_split(df, test_size=args.valid_size, stratify=df['class'], random_state=42)
        else:
            train_df, valid_df = train_test_split(df, test_size=args.valid_size, random_state=42)
    else:
        train_df = df
        valid_df = None

    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)
    tokenizer = get_tokenizer(args.model_name)
    
    valid_loader = None
    if args.dynamic_img_size:
        train_dataset = TrainDataset(train_df, transforms, tokenizer, args, with_gt=True)
        train_sampler = GroupSampler(train_dataset)
        train_batch_sampler = GroupBatchSampler(train_sampler, batch_size=args.batch_size, shuffle=True) 
        train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, collate_fn=collate_fn, pin_memory=True, num_workers=args.workers)

        if args.valid:
            valid_dataset = ValidDataset(valid_df, transforms, tokenizer, args, with_gt=True)
            valid_sampler = GroupSampler(valid_dataset)
            valid_batch_sampler = GroupBatchSampler(valid_sampler, batch_size=args.valid_batch_size)
            valid_loader = DataLoader(valid_dataset, batch_sampler=valid_batch_sampler, collate_fn=valid_collate_fn, pin_memory=True, num_workers=args.workers)
    else:
        train_dataset = TrainDataset(train_df, transforms, tokenizer, args, with_gt=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, pin_memory=True, shuffle=True, num_workers=args.workers)
        if args.valid:
            valid_dataset = ValidDataset(valid_df, transforms, tokenizer, args, with_gt=True)
            valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, collate_fn=valid_collate_fn, pin_memory=True, shuffle=False, num_workers=args.workers)
    
    train_loader.num_samples = len(train_dataset)
    train_loader.num_batches = len(train_loader)
    
    if args.valid:
        valid_loader.num_samples = len(valid_dataset)
        valid_loader.num_batches = len(valid_loader)

    return train_loader, valid_loader


class DupDataset(Dataset):
    def __init__(self, df, transforms, tokenizer=None):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = cv2.imread(self.df.loc[idx, 'image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        return image
    
    def get_pil_image(self, index):
        if isinstance(index, list):
            im_paths = self.df.loc[index, 'image_path']
            images = []
            for im_path in im_paths:
                image = Image.open(im_path)
                images.append(image)

            return images
        elif isinstance(index, int):
            im_path = self.df.loc[index, 'image_path']
            return Image.open(im_path)
        else:
            print("The index parameter's data type must be either list or int.")
    
    def get_batch(self, start, end):
        image_paths = self.df.loc[start:end-1, 'image_path'].tolist()
        image_list =[]
        for im_path in image_paths:
            image = cv2.imread(im_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transforms(image=image)['image']
            image_list.append(image.unsqueeze(0))
        return torch.concat(image_list, dim=0)
    
    def get_np_image(self, idx):
        image = cv2.imread(self.df.loc[idx, 'image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

class ImageInfo():
    def __init__(self, idx, image, tags, resize_size=None, frame=None):
        self.idx = idx
        self.image = image
        self.tags = tags
        self.resize_size = resize_size
        self.frame = frame
class TrainDataset(Dataset):
    def __init__(self, df, transforms, tokenizer, args, with_gt=False):
        self.df: pd.DataFrame = df
        self.dynamic_img_size = args.dynamic_img_size
        if self.dynamic_img_size:
            self.imageinfos: list[ImageInfo] = df.apply(lambda row: ImageInfo(row.name, row['image_path'], row['tags'], row['resize_size'], row['frame']), axis=1).to_list()
            self.frame_group_indices = df.groupby('frame').apply(lambda x: x.index.to_numpy()).values
        else:
            self.imageinfos: list[ImageInfo] = df.apply(lambda row: ImageInfo(row.name, row['image_path'], row['tags']), axis=1).to_list()

        self.transforms = transforms
        self.tokenizer = tokenizer
        self.weighted_shuffle = args.weighted_shuffle
        self.with_gt = with_gt
        self.cached = args.cache_image
        self.disk = args.cache_load_disk

        if not isinstance(args.gaussian_params, dict):
            args.gaussian_params = {key: float(value) for key, value in (arg.split("=", 1) for arg in args.gaussian_params)}

        if self.weighted_shuffle:
            self.tag_weights = make_tag_weights(df['tags'].values, args.gaussian_params) if args.weighted_shuffle else None

        if args.cache_image and args.cache_load:
            load_cache_images(self.imageinfos, self.transforms, args.save_cache_image, args.cache_dir, args.dynamic_img_size, args.cache_load_disk)
  
        if args.cache_image and not args.cache_load:
            cache_process(self.imageinfos, self.transforms, args.save_cache_image, args.cache_dir, args.dynamic_img_size, args.cache_load_disk)

        # del self.df

    def __len__(self):
        return len(self.imageinfos)

    def __getitem__(self, idx):
        info = self.imageinfos[idx]
        if self.cached:
            image = info.image
            if self.disk:
                image = torch.tensor(np.load(image)['image'], dtype=torch.float32)
        else:
            image = cv2.imread(info.image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.dynamic_img_size:
                image = A_resize(image, info.resize_size[::-1], cv2.INTER_AREA)
                crop_coords = get_center_crop_coords(image.shape, info.frame[::-1])
                image = A_crop(image, *crop_coords)
            image = self.transforms(image=image)['image']

        output = self.shuffle_tokenize(info.tags, self.tokenizer)

        return image, output['tag'], output.get("sel_gt", None), output.get("gt", None)

    def shuffle_tokenize(self, tags, tokenizer):
        output = {}
        result = torch.zeros(tokenizer.context_length, dtype=torch.long)
        
        if self.weighted_shuffle:
            tags = self.weighted_shuffle_tag(tags)
        else:
            np.random.shuffle(tags)

        tags = " , ".join(tags)     # To use "," as a separator.
        tags = re.sub('_', ' ', tags)

        tokens = tokenizer.encode(tags)
        gt = re.split("\s*,\s*", tokenizer.decode(tokens)) 
        gt[-1] = gt[-1].strip()

        max_idx = min(len(tokens)-1, tokenizer.context_length -2)

        if tokens[max_idx] == 267:
            sel_tokens =  tokens[0:max_idx]
        else:
            sel_tokens =  tokens[0:max_idx]
            truncate_idx = max_idx - sel_tokens[::-1].index(267) - 1
            sel_tokens = sel_tokens[:truncate_idx]

        sel_gt = re.split("\s*,\s*", tokenizer.decode(sel_tokens)) 
        sel_gt[-1] = sel_gt[-1].strip()

        if self.weighted_shuffle: np.random.shuffle(sel_gt)
        sel_tokens = tokenizer.encode(" , ".join(sel_gt))
        sel_tokens = [tokenizer.sot_token_id] + sel_tokens + [tokenizer.eot_token_id]
        
        result[:len(sel_tokens)] = torch.tensor(sel_tokens)
        output['tag'] = result
        output['sel_gt'] = set(sel_gt)
        output['gt'] = set(gt)

        return output
    
    def weighted_shuffle_tag(self, tags):
        weights = [self.tag_weights.get(tag, 1) for tag in tags]
        # print(sorted([(tag, weight) for tag, weight in zip(tags, weights)], key=lambda x: x[1], reverse=True))
        # print(sorted([(tag, weight) for tag, weight in zip(tags, np.array(weights) / sum(weights))], key=lambda x: x[1], reverse=True))
        return list(np.random.choice(tags, size=len(tags), replace=False, p=np.array(weights) / sum(weights)))
    
class ValidDataset(TrainDataset):
    def __init__(self, df, transforms, tokenizer, args, with_gt=False):
        super().__init__(df, transforms, tokenizer, args, with_gt=with_gt)
        self.weighted_shuffle = False
        self.tag_outputs = [self.shuffle_tokenize(info.tags, self.tokenizer) for info in self.imageinfos]

    def __getitem__(self, idx):
        info = self.imageinfos[idx]
        if self.cached:
            image = info.image
            if self.disk:
                image = torch.tensor(np.load(image)['image'], dtype=torch.float32)
        else:
            image = cv2.imread(info.image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.dynamic_img_size:
                image = A_resize(image, info.resize_size[::-1], cv2.INTER_AREA)
                crop_coords = get_center_crop_coords(image.shape, info.frame[::-1])
                image = A_crop(image, *crop_coords)
            image = self.transforms(image=image)['image']

        output = self.tag_outputs[idx]

        return image, output['tag'], output.get("sel_gt", None), output.get("gt", None), idx

class GroupSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        # return iter([i for i in range(len(self.data_source)) if i % 2 == 0])
        return iter([i for i in range(len(self.dataset.imageinfos))])

    def __len__(self):
        return len(self.dataset)
    
class GroupBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, shuffle=False):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")
        
        self.sampler = sampler
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        self.batches = self.make_batches(self.sampler.dataset.frame_group_indices, self.batch_size, self.shuffle)
        for batch in self.batches:
            yield batch
            
                
    def __len__(self) -> int:
        total_len = 0
        for indices in self.sampler.dataset.frame_group_indices:
            if len(indices) % self.batch_size == 0:
                total_len += len(indices) // self.batch_size
            else:
                total_len += (len(indices) // self.batch_size) + 1
 
        return total_len
    
    def make_batches(self, group_indices, batch_size, shuffle=False):
        if shuffle:
            for indices in group_indices:
                np.random.shuffle(indices)

        batches = np.array([
            indices[i:i + batch_size]
            for indices in group_indices
            for i in range(0, len(indices), batch_size)
        ], dtype=object)

        if shuffle:
            np.random.shuffle(batches)

        return batches
    
def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    tags = torch.stack([item[1] for item in batch])
    sel_gts = [item[2] for item in batch]
    gts = [item[3] for item in batch]

    return images, tags, sel_gts, gts


def valid_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    tags = torch.stack([item[1] for item in batch])
    sel_gts = [item[2] for item in batch]
    gts = [item[3] for item in batch]
    indices = [item[4] for item in batch]

    return images, tags, sel_gts, gts, indices

class ImageProcessor():
    def __init__(self, image_size:tuple=(448, 448), patch_size=16, step_size=64, min_size=210, max_size=840):
        self.mean = (0.68363303, 0.62883478, 0.63107163)
        self.std = (0.31656915, 0.31771758, 0.31206703)

        self.frames = make_frames(
            pixel_w=image_size[0], 
            pixel_h=image_size[1], 
            patch_size=patch_size, 
            step_size=step_size, 
            min_size=min_size, 
            max_size=max_size
            )
        self.frame_aspect_ratios = [f[0] / f[1] for f in self.frames]

        self.frames = np.array(self.frames)
        self.frame_aspect_ratios = np.array(self.frame_aspect_ratios)

        transforms = [
            A.Normalize(mean=self.mean, std=self.std, max_pixel_value=255),
            ToTensorV2(),
        ]
        self.transforms = A.Compose(transforms)

    def __call__(self, img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        resize, frame = get_resize_and_frame_size(img_path, self.frames, self.frame_aspect_ratios)
        image = A_resize(image, resize[::-1], cv2.INTER_AREA)
        crop_coords = get_center_crop_coords(image.shape, frame[::-1])
        image = A_crop(image, *crop_coords)

        return self.transforms(image=image)['image']
    
    def get_pil_image(self, img_path):
        breakpoint()
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        resize, frame = get_resize_and_frame_size(img_path, self.frames, self.frame_aspect_ratios)
        image = A_resize(image, resize[::-1], cv2.INTER_AREA)
        crop_coords = get_center_crop_coords(image.shape, frame[::-1])
        image = A_crop(image, *crop_coords)

        image = Image.fromarray(image)

        return image

