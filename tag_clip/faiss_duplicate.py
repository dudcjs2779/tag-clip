import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from albumentations.pytorch import ToTensorV2
import open_clip
import pandas as pd
import albumentations as A
import cv2
import faiss
import shutil
from datetime import datetime
from tag_clip.utils.data import get_files
from tag_clip.utils.dataset import DupDataset
from tag_clip.utils.basic_utils import read_config


def setup_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Directory path of images want to check duplicates. Subfolders are also searchable.",
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default=None,
        help=(
            "Faiss index path."
        )
    )
    parser.add_argument(
        "--dup_dir",
        type=str,
        default='./dup_dir',
        help=(
            "Directory path of duplicate images to move. When remove argument is true."
        )
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default='./index',
        help=(
            "Save path for updated index and dataframe after processing duplicate images."
        )
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help=(
            "Batch size for compute similarity."
        )
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.995,
        help=(
            "Similarity threshold to be considered a duplicate image."
        )
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help=(
            "Faiss search method topk. Particular image has many duplicates, set larger this argument. "
            "if set to 20, it can filter up to 20 duplicates for each image. "
        )
    )
    parser.add_argument(
        "--do_test",
        type=bool,
        default=False,
        help=(
            "Verify that a query brings itself to the top1 after update index."
        )
    )
    parser.add_argument(
        "--remove",
        type=str,
        default=False,
        help=(
            "Remove duplicates instead of move to {dup-dir}."
        )
    )
    parser.add_argument(
        "--pair",
        type=str,
        default=False,
        help=(
            "If caption data exists, it is processed together."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-32",
        help=(
            "CLIP model. same with open_clip."
        )
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="laion2b_s34b_b79k",
        help=(
            "Pretrained for data. same with open_clip."
        )
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help=(
            "Load arguments from config file."
        )
    )

    return parser


def make_duplicate_group(index, df, batch_size, threshold=0.99, topk=20) -> dict:
    ntotal = index.ntotal

    dup_dict = {}
    for i in tqdm(range(0, ntotal, batch_size)):
        end = min(i + batch_size, ntotal)
        query = index.reconstruct_n(i, end - i)
        dis, indices = index.search(query, topk)

        for j, (dis_j, idx_j) in enumerate(zip(dis, indices)):
            dups = idx_j[dis_j > threshold]
            highest_res = df.loc[dups, 'width'].idxmax()
            dup_dict[highest_res] = list(dups[dups != highest_res])

    return dup_dict

def move_duplicate(dup_dict, index, df, dst_dir, remove=False, pair=False):
    if not os.path.exists: os.makedirs(dst_dir, exist_ok=True)
    deleted = set()
    keep = set()
    count = 0
    for keep_idx, dup_idxs in tqdm(dup_dict.items()):
        keep.add(keep_idx)
        for dup_idx in dup_idxs:
            if dup_idx not in keep and dup_idx not in deleted:
                if remove:
                    del_image = df.loc[dup_idx, 'image_path']
                    del_txt = os.path.splitext(del_image)[0] + '.txt'

                    if pair:
                        os.remove(del_image)
                        os.remove(del_txt)
                    else:
                        os.remove(del_image)

                    deleted.add(dup_idx)
                    count += 1
                else:
                    del_image = df.loc[dup_idx, 'image_path']
                    del_txt = os.path.splitext(del_image)[0] + '.txt'

                    image_name = os.path.basename(del_image)
                    txt_name = os.path.basename(del_txt)

                    dir_name = os.path.basename(os.path.dirname(del_image))

                    im_dst = os.path.join(dst_dir, dir_name, image_name)
                    txt_dst = os.path.join(dst_dir, dir_name, txt_name)
                    os.makedirs(os.path.dirname(im_dst), exist_ok=True)

                    if pair:
                        shutil.move(del_image, im_dst)
                        shutil.move(del_txt, txt_dst)
                    else:
                        shutil.move(del_image, im_dst)

                    deleted.add(dup_idx)
                    count += 1


    deleted = np.array(list(deleted), dtype=np.int64)
    df = df.drop(index=deleted)
    df = df.reset_index(drop=True)
    index = faiss.index_gpu_to_cpu(index)
    index.remove_ids(deleted)

    print(f"{count} duplicate images are moved to {dst_dir}")
    print(f"total index embeddings are {index.ntotal}, dataframe total rows are {len(df)}\n")
    return index, df



def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ## Prepare Datas
    print('Load Data')
    image_paths, _ = get_files(args.image_dir, pair=args.pair)

    df = pd.DataFrame(image_paths, columns=["image_path"])
    df['width'], df['height'] = zip(*df['image_path'].apply(lambda x: Image.open(x).size))

    index_flat = faiss.read_index(args.index_path)

    ## Filtering duplicate images
    print('Filtering')
    dup_dict = make_duplicate_group(index_flat, df, batch_size=args.batch_size, threshold=args.threshold, topk=args.topk)
    index_flat, df = move_duplicate(dup_dict, index_flat, df, args.dup_dir, remove=args.remove, pair=args.pair)
    print()

    image_paths, _ = get_files(args.image_dir, pair=args.pair)   # for check

    ## Test
    if args.do_test:
        print()
        print('Testing')
        img_size = 224
        transform = [
            # A.Resize(width=img_size, height=img_size, interpolation=cv2.INTER_AREA),
            # A.CenterCrop(width=img_size, height=img_size),
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255]), # 패딩
            A.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711], max_pixel_value=255),
            ToTensorV2(),
        ]
        transform = A.Compose(transform)

        my_dataset = DupDataset(df, transform)
        model = open_clip.create_model('ViT-B-32', pretrained='laion2b_s34b_b79k')
        model.eval()  
        model.to(device)
        # tokenizer = open_clip.get_tokenizer('ViT-B-32')

        sample_idx = 5
        query = my_dataset[sample_idx].unsqueeze(0).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(query)

        embeddings = image_features.detach().cpu().numpy().astype(np.float32)
        faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
        dis, indices = index_flat.search(embeddings, 20)

        print(f'Input image ID: {sample_idx}')
        print(f'Top1 retriaval data: ID:{indices[0][0]}, Score:{dis[0][0]}\n')

    ## Save Dataframe and Index
    print('Save Result')
    t_now = datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    dir_path = os.path.join(args.save_dir, t_now)
    if not os.path.exists(dir_path): os.makedirs(dir_path, exist_ok=True)

    df.to_parquet(os.path.join(dir_path, f'images.parquet'))
    faiss.write_index(index_flat, os.path.join(dir_path, f'images.index'))
    print(f'Saved at {dir_path}')

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args = read_config(args, parser)
    main(args)
    
