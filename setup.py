from setuptools import setup, find_packages

setup(
    name="tag_clip",
    version="0.1.0",
    description="Code to train the clip with tag",
    author="dudcjs2779",
    author_email="dudcjs2779@gmail.com",
    url="https://github.com/dudcjs2779/tag-clip.git",
    packages=find_packages(),
    install_requires=[
        "torch==2.1.2",
        "open_clip_torch==2.29.0",
        "opencv-python",
        "albumentations",
        "scikit-learn",
        "toml",
        "bitsandbytes",
        "numpy==1.26.4",
        "pandas",
        "tensorboard",
        "wandb",
        "pyarrow"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
