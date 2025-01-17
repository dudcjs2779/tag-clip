# tag-clip
This project is about a CLIP model trained with images and tags.
And I referenced a lot [open_clip](https://github.com/mlfoundations/open_clip/).

## CLIP-Approach
![image](sample/CLIP.png)


## Model
I used eva02_base_patch16_clip_224.merged2b_s8b_b131k model and pretrain weight. [EVA-02](https://arxiv.org/abs/2303.11331) [EVA-CLIP](https://arxiv.org/abs/2303.15389).  
<img src="sample/EVA-02.png" width="600">

## Usage
I recommend create conda env with python 3.10.
```
conda create -n tag_clip python=3.10

# activate env
conda activate tag_clip
or
source activate tag_clip
```

Install the required packages with the following commands.
```
pip install -e .
```

### Model inference
```
python tag_clip/inference_example.py
```

### Model train 
Run `src/eva_clip.py` for training. You can easily set arguments with a config file `config/eva02_base_patch16_clip_default.toml`.  
In the config file, change the `train`, `image_key`, `tag_key` arguments to match your dataset.  
- train: parquet file path.
- image_key: the name of column contains image path.(str)
- tag_key: the name of column contains tags.(numpy.ndarray)
  
For a detailed description of the arguments, check out `src/utils/params.py.`  

Optionally, install [faiss](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) to calculate the recall of the validation. This corresponds to the `recall` argument of config file.
```
python tag_clip/eva_clip.py --config_path="config/eva02_base_patch16_clip_default.toml"
```

## Evaluate
train: 29187, valid:3000   
<img src="sample/valid_recall.png" width="800">  
Epoch15
- R@1: 0.87733
- R@5: 0.96767
- R@10: 0.98233


