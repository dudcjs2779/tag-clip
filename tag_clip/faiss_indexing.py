import os
import argparse
from datetime import datetime
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

from torch.utils.data import DataLoader
from utils.data import get_files
from utils.dataset import DupDataset
from utils.basic_utils import read_config

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Directory path of images.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default='./index',
        help=(
            "Save path for index"
        )
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help=(
            "Batch size for clip."
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

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    image_paths, _ = get_files(args.image_dir, pair=args.pair)
    df = pd.DataFrame(image_paths, columns=["image_path"])

    print("Load Model")
    model = open_clip.create_model('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    model.to(device)
    # tokenizer = open_clip.get_tokenizer('ViT-B-32')

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
    dataloader = DataLoader(
        my_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
    )

    dim = 512
    index_flat = faiss.IndexFlat(dim, faiss.METRIC_INNER_PRODUCT)

    print("Start Embedding")
    count = 0
    for batch in tqdm(dataloader):
        image = batch
        image = image.to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(image)

        embeddings = image_features.detach().cpu().numpy().astype(np.float32)
        faiss.normalize_L2(embeddings)
        index_flat.add(embeddings)

        
    print("Save index")
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir, exist_ok=True)
    t_now = datetime.now().strftime('%y-%m-%d_%H-%M-%S')

    faiss.write_index(index_flat, os.path.join(args.save_dir, f"{t_now}.index"))

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args = read_config(args, parser)
    main(args)
