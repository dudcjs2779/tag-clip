import os
import math
from PIL import Image
import shutil
from tqdm import tqdm
from tag_clip.utils.data import get_files

def resize_images(image_dir, max_res=4096, max_res_offset=512):
    image_paths, _ = get_files(image_dir)

    count = 0
    for im_path in tqdm(image_paths):
        image = Image.open(im_path)
        width, height = image.size
        
        if width * height > (max_res + max_res_offset)**2:
            aspect_ratio = width / height
            resize_h  = math.ceil(math.sqrt(max_res**2 / aspect_ratio))
            resize_w = math.ceil(resize_h * aspect_ratio)

            image = Image.open(im_path)
            resized_image = image.resize((resize_w, resize_h), Image.Resampling.BICUBIC)
            resized_image.save(im_path)
            count += 1

            print(im_path)
            print(f"{image.size} => {resized_image.size}")
            print()

    print(f"total images: {len(image_paths)}, resized images: {count}")


def move_long_images(image_dir, dst_dir=None):
    image_paths, txt_paths = get_files(image_dir)
    total_len = len(image_paths)

    if not dst_dir:
        parent_dir = os.path.dirname(image_dir)
        dst_dir = os.path.join(parent_dir, "long_image")

    if not os.path.exists(dst_dir): os.makedirs(dst_dir, exist_ok=True)

    count = 0
    for i in tqdm(range(len(image_paths))):
        im_path = image_paths[i]
        txt_path = txt_paths[i]
        image = Image.open(im_path)
        width, height = image.size
        
        long_side = max(width, height)
        short_side = min(width, height)

        aspect_ratio = long_side / short_side

        if aspect_ratio > 3.5:
            image_name = os.path.basename(im_path)
            txt_name = os.path.basename(txt_path)

            image_dst = os.path.join(dst_dir, image_name)
            txt_dst = os.path.join(dst_dir, txt_name)

            shutil.move(im_path, image_dst)
            shutil.move(txt_path, txt_dst)
            count += 1

    image_paths, txt_paths = get_files(image_dir)

    print(f"total images: {total_len}, detected long images: {count}")
    print(f"After move image: {len(image_paths)}, txt: {len(txt_paths)}")

