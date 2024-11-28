import os
import re
import argparse
import shutil
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from tag_clip.utils.data import get_files
import html
from open_clip.factory import get_tokenizer
from tag_clip.utils.basic_utils import read_config
tqdm.pandas()

def setup_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Directory path of images and tags. Image and tag(txt file) must be in same directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default='./my_dataset/table_data',
        help=(
            "Save path for dataframe and When remove arg is flase, long tag datas are moved at this directroy."
        )
    )
    parser.add_argument(
        "--tag_csv",
        type=str,
        default='my_dataset/table_data/tags.csv',
        help=(
            "tag.csv path. info of all tags."
        )
    )
    parser.add_argument(
        "--remove",
        type=bool,
        default=False,
        help=(
            "Remove long tag datas instead of move to {save_dir}."
        )
    )
    parser.add_argument(
        "--max_token_len",
        type=int,
        default=500,
        help=(
            "if tags token length grater than this value, that tags treated as long data."
        )
    )
    parser.add_argument(
        "--general_limit",
        type=int,
        default=200,
        help=(
            "General tags with a frequency of less than this value will be excluded."
        )
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="EVA02-L-14-336",
        help=(
            "tokenizer name, same with open_clip."
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

def make_tag_df(image_paths, txt_paths):
    datas = []
    for i in range(len(image_paths)):
        im_path = image_paths[i]
        txt_path = txt_paths[i]

        with open(txt_path, 'r') as f:
            text = f.read()
        tags = re.split(',\s*', text)

        data = {'image_path': im_path, 'txt_path': txt_path, 'tags': tags}

        datas.append(data)

    return pd.DataFrame(datas)

def preprocess_gel_tag(df):
    df = df.dropna(subset='name')
    df = df[df['count'] > 0]
    df = df[df['id'] != 685386]
    df['name'] = df['name'].apply(lambda x: html.unescape(x))
    df = df.drop_duplicates(subset="id")
    df = df.drop_duplicates(subset="name")
    df = df.reset_index(drop=True)

    return df

def process_tag(df_gel:pd.DataFrame, txt_paths:list, general_limit=200) -> list:
    all_set = set(df_gel['name'].unique())
    type_0 = set(df_gel[(df_gel['type'] == 0) & (df_gel['count'] < general_limit)]['name'].unique())    # general tag
    type_1 = set(df_gel[df_gel['type'] == 1]['name'].unique())  # artist tag
    type_3 = set(df_gel[df_gel['type'] == 3]['name'].unique())  # copyright tag
    type_5 = set(df_gel[df_gel['type'] == 5]['name'].unique())  # metadata tag(ex: highres, absurdres)
    type_6 = set(df_gel[df_gel['type'] == 6]['name'].unique())  # deprecated tag
    remove_tag_set = [type_0, type_1, type_3, type_5, type_6]

    new_tags_list = []
    before = set()
    after = set()
    for i in tqdm(range(len(txt_paths)), desc="processing_tag"):
        txt_path = txt_paths[i]

        with open(txt_path, 'r') as f:
            text = f.read()
        tags = re.split(',\s*', text)
        before.update(tags)

        for tag_set in remove_tag_set:
            tags = [tag for tag in tags if tag not in tag_set and tag in all_set]
        after.update(tags)
            
        new_tags_list.append(tags)

    print(f'The number of unique tags {len(before)} to {len(after)}')
    return new_tags_list

def get_token_len(tags, tokenizer):
    tags_txt = ", ".join(tags)
    tags_txt = re.sub("_", " ", tags_txt)
    return len(tokenizer.encode(tags_txt))

def move_long_token_data(df:pd.DataFrame, max_token_len, save_dir, remove=False) -> pd.DataFrame:
    temp = df[df['tags_len'] >= max_token_len]

    count = 0
    for im_path, txt_path in tqdm(temp[['image_path', 'txt_path']].values, desc="Processing long tag"):
        if remove:
            if os.path.isfile(im_path): os.remove(im_path)
            if os.path.isfile(txt_path): os.remove(txt_path)
            count+=1
        else:
            image_name = os.path.basename(im_path)
            txt_name = os.path.basename(txt_path)
            dir_name = os.path.basename(os.path.dirname(im_path))

            dst_dir = os.path.join(os.path.dirname(save_dir), "long_tag", dir_name)
            os.makedirs(dst_dir, exist_ok=True)
            
            dst_image = os.path.join(dst_dir, image_name)
            dst_txt = os.path.join(dst_dir, txt_name)

            shutil.move(im_path, dst_image)
            shutil.move(txt_path, dst_txt)
            count+=1

    df = df.drop(index=temp.index)
    df = df.reset_index(drop=True)

    print(f'{len(df)} processed total datas, {len(temp)} long tag datas are found, {count} tags are processed')
    return df

def apply_tag(df):
    for txt_path, tags in tqdm(df[['txt_path', 'tags']].values, desc="Applying to txt files"):
        tags = ", ".join(tags)

        with open(txt_path, 'w') as f:
            f.write(tags)

def main(args):
    t_now = datetime.now().strftime('%y-%m-%d_%H-%M-%S')

    # Load Data
    print("Load Data")
    image_paths, txt_paths = get_files(args.image_dir, pair=True)
    df_gel = pd.read_csv('my_dataset/table_data/tags.csv')

    # Backup
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir, exist_ok=True)
    df = make_tag_df(image_paths, txt_paths)
    print(f"\nBackup datas before processing at {os.path.join(args.save_dir, f'{t_now}_tag_backup.parquet')}.")
    df.to_parquet(os.path.join(args.save_dir, f'{t_now}_tag_backup.parquet'))

    df_gel = preprocess_gel_tag(df_gel)
    tokenizer = get_tokenizer(args.tokenizer_name)

    print("\nFiltering tags")
    df['tags'] = process_tag(df_gel, txt_paths, args.general_limit)

    print("\nDrop long tag datas")
    df['tags_len'] = df['tags'].progress_apply(lambda tags: get_token_len(tags, tokenizer))
    df = move_long_token_data(df, args.max_token_len, args.save_dir, remove=args.remove)

    print("\nApply & Save result")
    df.to_parquet(os.path.join(args.save_dir, f"{t_now}_processed.parquet"), index=False)
    apply_tag(df)

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args = read_config(args, parser)
    main(args)

