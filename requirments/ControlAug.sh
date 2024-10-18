#!/bin/bash

# Install pip packages
pip install ftfy==6.1.1 \
            regex \
            tqdm==4.65.0 \
            pycocotools==2.0.6 \
            scipy==1.10.1 \
            opencv-python \
            numpy==1.23.1 \
            git+https://github.com/FANGAreNotGnu/CLIP.git \
            einops==0.3.0 \
            gradio==3.16.2 \
            albumentations==1.3.0 \
            opencv-contrib-python==4.3.0.36 \
            imageio==2.9.0 \
            imageio-ffmpeg==0.4.2 \
            pytorch-lightning==1.5.0 \
            omegaconf==2.1.1 \
            test-tube>=0.7.5 \
            streamlit==1.12.1 \
            webdataset==0.2.5 \
            kornia==0.6 \
            open_clip_torch==2.0.2 \
            invisible-watermark>=0.1.5 \
            streamlit-drawable-canvas==0.8.0 \
            torchmetrics==0.6.0 \
            timm==0.6.12 \
            addict==2.4.0 \
            yapf==0.32.0 \
            prettytable==3.6.0 \
            safetensors==0.2.7 \
            basicsr==1.4.2 \
            openmim==0.3.9 \
            fasttext \
            scikit-learn \
            lvis \
            fvcore \
            easydict \
            nltk \
            git+https://github.com/openai/CLIP.git \
            git+https://github.com/lvis-dataset/lvis-api.git \
            instaboostfast \
            lmdb \
            ipython \
            eniops

echo "All packages installed successfully."
