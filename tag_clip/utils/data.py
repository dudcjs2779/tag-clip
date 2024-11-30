import os
import torch
import cv2
import numpy as np
import math
from PIL import Image
from tqdm import tqdm
from collections import Counter
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.geometric.functional import resize as A_resize
from albumentations.augmentations.crops.functional import get_center_crop_coords, crop as A_crop

ext_type = (".jpg", ".png", ".jpeg")

def get_files(directory, pair=True):
    images = []
    txts = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(ext_type):
                image_path = os.path.abspath(os.path.join(root, file))
                txt_path = os.path.splitext(os.path.join(root, file))[0] + ".txt"
                txt_path = os.path.abspath(txt_path)

                if pair:
                    if os.path.isfile(txt_path):
                        images.append(image_path)
                        txts.append(txt_path)
                    else:
                        print(image_path)
                        print(txt_path)
                        print(f"Not pair data, ignored: {file}")
                else:
                    images.append(image_path)
                    txts.append(txt_path)
    
    print(f"{len(images)} images, {len(txts)} txts")
    return images, txts

def remove_zone_files(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith("zone.identifier"):
                os.remove(os.path.join(root, file))
                count += 1
    
    print(f"{count} zone.identifier files are removed")

def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    tensor = tensor * std + mean
    return tensor


def tensor_to_pil_image(tensor, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]):
    if isinstance(tensor, np.ndarray):
        tensor = torch.tensor(tensor)

    # 텐서를 numpy 배열로 변환 (배치와 채널 확인)
    if len(tensor.shape) == 4:  # 배치가 포함된 텐서라면 배치를 제거
        tensor = tensor.squeeze(0)

    tensor = denormalize(tensor, mean=mean, std=std,)
    
    # 텐서가 [C, H, W] 형태일 때 [H, W, C]로 변환
    tensor = tensor.permute(1, 2, 0)

    # 텐서 값을 0-1 범위에서 0-255 범위로 스케일링
    tensor = tensor.detach().cpu().numpy()
    tensor = (tensor * 255).astype(np.uint8)
    
    # numpy 배열을 PIL 이미지로 변환
    pil_image = Image.fromarray(tensor)

    return pil_image

def numpy_to_pil_image(arr, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]):
    if len(arr.shape) == 4:  # 배치가 포함된 텐서라면 배치를 제거
        arr = arr.squeeze(0)

    arr = denormalize(arr, mean=mean, std=std,)
    
    # 텐서가 [C, H, W] 형태일 때 [H, W, C]로 변환
    arr = arr.transpose(1, 2, 0)

    # 텐서 값을 0-1 범위에서 0-255 범위로 스케일링
    arr = (arr * 255).astype(np.uint8)

    # numpy 배열을 PIL 이미지로 변환
    pil_image = Image.fromarray(arr)

    return pil_image

def load_cache_images(infos, transforms, save, cache_dir, dynamic_img_size, disk):
    assert cache_dir, "The cache_dir argument is None. If you want to load cache images, set the cache_dir argument. Otherwise, set the cache_load argument to false."
    assert os.path.exists(cache_dir), f"{cache_dir} cache_dir path is not exists."

    count = 0
    for i, info in tqdm(enumerate(infos), desc="Loading cache images", total=len(infos)):
        npz_name = os.path.splitext(os.path.basename(info.image))[0] + ".npz"
        npz_path = os.path.join(cache_dir, npz_name)
        if os.path.exists(npz_path):
            info.image = npz_path if disk else torch.tensor(np.load(npz_path)['image'], dtype=torch.float32)
            count += 1
        else:
            info.image = cache_image(info, transforms, save, cache_dir, dynamic_img_size, disk)

    print(f"\nTotal images: {len(infos)}, Cache images loaded: {count}, New Cache images: {len(infos) - count}")

def cache_process(infos, transforms, save, cache_dir, dynamic_img_size, disk):
    if save:
        if not cache_dir: cache_dir = "./my_dataset/img_caches"
        if not os.path.exists(cache_dir): os.makedirs(cache_dir, exist_ok=True)

    for i, info in tqdm(enumerate(infos), desc="Caching images", total=len(infos)):
        infos[i].image = cache_image(info, transforms, save, cache_dir, dynamic_img_size, disk)

def cache_image(info, transforms, save=False, cache_dir=None, dynamic_img_size=False, disk=False):
    img = cv2.imread(info.image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if dynamic_img_size:
        img = A_resize(img, info.resize_size[::-1], cv2.INTER_AREA)
        crop_coords = get_center_crop_coords(img.shape, info.frame[::-1])
        img = A_crop(img, *crop_coords)
    img = transforms(image=img)['image']

    if save: 
        cache_name = os.path.splitext(os.path.basename(info.image))[0] + ".npz"
        cache_path = os.path.join(cache_dir, cache_name)
        np.savez_compressed(cache_path, image=img.numpy())

    if disk: img = cache_path

    return img

def make_frames(pixel_w, pixel_h, patch_size, step_size, min_size, max_size) -> list:
    max_pixel = pixel_w * pixel_h

    frames = set()
    min_size = round(min_size / patch_size) * patch_size
    max_size = round(max_size / patch_size) * patch_size
    width = min_size
    frames.add((width, width))

    width = round(min_size / patch_size) * patch_size
    while width < max_size:
        height = min(max_size, math.floor((max_pixel / width) / patch_size) * patch_size)
        if height >= min_size:
            frames.add((width, height))
            frames.add((height, width))

        width += step_size

    return sorted(list(frames))

def get_resize_and_frame_size(img, frames, frame_aspect_ratios):
    if not isinstance(frames, np.ndarray): frames = np.array(frames)
    if not isinstance(frame_aspect_ratios, np.ndarray): frame_aspect_ratios = np.array(frame_aspect_ratios)

    width, height = Image.open(img).size
    aspect_ratio = width / height

    frame_id = np.abs(frame_aspect_ratios - aspect_ratio).argmin()
    frame_w, frame_h = frames[frame_id]

    if aspect_ratio < frame_w / frame_h:    # 원본이 세로가 긺
        resize_w = frame_w
        resize_h = round(resize_w / aspect_ratio)
    else:                                   # 원본이 가로가 긺
        resize_h = frame_h
        resize_w = round(resize_h * aspect_ratio)

    return (resize_w, resize_h), (frame_w, frame_h)

def gaussian_weight(freq, max_freq, mu_p=0.7, sigma_p=0.2, scale=3, min_offset=2.0):
    x = max_freq / freq
    x = np.log(x)
    max_weight = np.log(max_freq)

    mu = max_weight * mu_p
    sigma = max_weight * sigma_p
    scale = max_weight * scale
    # print(mu, sigma, scale)

    weight = (1 / (math.sqrt(2 * math.pi * sigma**2))) * math.exp(-((x - mu)**2) / (2 * sigma**2))
    weight = weight * scale +  min_offset
    return weight

def make_tag_weights(tags_list, params):
    tag_counter = Counter()

    for tags in tags_list:
        tag_counter.update(tags)

    max_freq = tag_counter.most_common(1)[0][1]
    tag_weights = {tag: gaussian_weight(freq,  max_freq, **params) for tag, freq in tag_counter.most_common()}

    return tag_weights

class ImageProceeser():
    def __init__(self, dynamic_img_size=False, mean=None, std=None, pixel_w=448, pixel_h=448, patch_size=14, step_size=42, min_size=210, max_size=840):
        self.dynamic_img_size = dynamic_img_size
        self.mean = mean if mean else (0.68363303, 0.62883478, 0.63107163)
        self.std = std if std else (0.31656915, 0.31771758, 0.31206703)
        self.transforms = self.get_transforms(pixel_w)

        if self.dynamic_img_size:
            self.frames = make_frames(pixel_w, pixel_h, patch_size, step_size, min_size, max_size)
            self.frame_aspect_ratios = frame_aspect_ratios = [f[0] / f[1] for f in self.frames]

    def image_processer(self, image):
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.dynamic_img_size:
            resize_size,  frame_size = get_resize_and_frame_size(image, self.frames, self.frame_aspect_ratios)
            img = A_resize(img, resize_size[::-1], cv2.INTER_AREA)
            crop_coords = get_center_crop_coords(img.shape, frame_size[::-1])
            img = A_crop(img, *crop_coords)
        img = self.transforms(image=img)['image']
        return img

    def get_transforms(self, img_size):
        if self.dynamic_img_size:
            transforms = [
                A.Normalize(mean=self.mean, std=self.std, max_pixel_value=255),
                ToTensorV2(),
            ]
        else:
            transforms = [
                A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
                A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255]), # padding
                A.Normalize(mean=self.mean, std=self.std, max_pixel_value=255),
                ToTensorV2(),
            ]
            
        return A.Compose(transforms)




