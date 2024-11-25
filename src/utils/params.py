import argparse

def setup_parser():
    parser = argparse.ArgumentParser()

    ## model
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--precision",
        choices= ["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "pure_bf16", "pure_fp16", "fp32"],
        default='amp',
        help="Floating point precision."
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="path to latest checkpoint that want to resume train."
    )
    parser.add_argument(
        "--save_frequency",
        type= int,
        default=1,
        help="How often to save checkpoints."
    )
    parser.add_argument(
        "--max_save_count",
        type= int,
        default=None,
        help="How many checkpoints to save. Checkpoint will be deleted starting from the oldest one."
    )

    ## tokenizer
    parser.add_argument(
        "--weighted_shuffle",
        type=bool,
        default=False,
        help="Select tags based on their frequency when training."
    )
    parser.add_argument(
        "--gaussian_params",
        type=list,
        default=[],
        help="Sets the parameters of a normal distribution that determines how tags are weighted based on their frequency. Using with weighted_shuffle."
    )

    ## train
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs to train for."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size per GPU."
    )
    parser.add_argument(
        "--accum_freq",
        type=int,
        default=1,
        help="Update the model every --acum-freq steps."
    )
    parser.add_argument(
        "--loss_fn",
        choices= ["CE", "ML", "KL"],
        default='KL',
        help=(
            "Loss funtion. "
            "CE: Cross Entropy, the default loss function of clip."
            "ML: Multilabel Soft Margin Loss. this loss fucntion might useful when training with short tag or class data. It can reflects multi label."
            "KL: KL-Divergence, this loss fucntion is useful when training with tag data. It reflects how many ground true tags are included in negative text, not just actual ground true."
        )
    )

    ## validation
    parser.add_argument(
        "--valid_batch_size",
        type=int,
        default=16,
        help="Validate batch size."
    )
    parser.add_argument(
        "--valid",
        type=bool,
        default=True,
        help="Validate every epoch."
    )
    parser.add_argument(
        "--valid_size",
        type=int,
        default=2000,
        help="Validation set size."
    )
    parser.add_argument(
        "--recall",
        type=bool,
        default=False,
        help="Compute text-to-image R@1, R@5, R@10 in validation."
    )
    parser.add_argument(
        "--split_by_class",
        type=bool,
        default=False,
        help="Split train/valid dataset by class."
    )

    ## logger
    parser.add_argument(
        "--report_to",
        type=str,
        default='tensorboard',
        help="Options are ['wandb', 'tensorboard']."
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default='my-clip',
        help="wandb project name."
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="wandb run id. Must set this argument when use resume argument."
    )

    # optimizer
    parser.add_argument(
        "--image_encoder_lr",
        type=float,
        default=None,
        help="Learning rate for image encoder."
    )
    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=None,
        help="Learning rate for text encoder."
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1000,
        help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.05,
        help="Weight decay."
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.09,
        help="Adam beta 1."
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.98,
        help="Adam beta 2."
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="Adam epsilon."
    )
    parser.add_argument(
        "--img_lwld_decay",
        type=float,
        default=1.0,
        help="Layer-wise learning rate decay of image_encoder."
    )
    parser.add_argument(
        "--text_lwld_decay",
        type=float,
        default=1.0,
        help="Layer-wise learning rate decay of text_encoder."
    )

    ## data
    parser.add_argument(
        "--train",
        type=str,
        default=None,
        help=(
            "Training data that the type of dataframe with a .parquet extension. "
            "it is necessary image path column(str) and tag column(numpy.ndarray)."
        )
    )
    parser.add_argument(
        "--image_key",
        type=str,
        default="image_path",
        help="For pd.dataframe, the name of column contains image path.(str)"
    )
    parser.add_argument(
        "--tag_key",
        type=str,
        default="tags",
        help="For pd.dataframe, the name of column contains tags.(numpy.ndarray)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Dataloader num_wokers."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache images directory path."
    )
    parser.add_argument(
        "--cache_image",
        type=bool,
        default=False,
        help="Cache images to save time load images."
    )
    parser.add_argument(
        "--save_cache_image",
        type=bool,
        default=False,
        help="Save cache images."
    )
    parser.add_argument(
        "--cache_load",
        type=bool,
        default=False,
        help="Load & Use saved cache images instead of original images."
    )
    parser.add_argument(
        "--cache_load_disk",
        type=bool,
        default=False,
        help="Load cache image from disk. This argument is useful when RAM is not enough."
    )

    ## dataset
    parser.add_argument(
        "--dynamic_img_size",
        type=bool,
        default=True,
        help="Keep aspect ratio of images as much as possible. Images of various sizes are used for training."
    )
    parser.add_argument(
        "--image_size",
        type=list,
        default=[448, 448],
        help="Image size for train. Images will be resized automatically."
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=16,
        help="Model patch size for make frame."
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=64,
        help="Pixel difference between frames to use when cropping the image."
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=256,
        help="Minimum size of one side of the frame."
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=1024,
        help="Maximum size of one side of the frame."
    )

    ## debug
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="Debug option."
    )

    ## Config
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Load arguments from config file."
    )

    return parser