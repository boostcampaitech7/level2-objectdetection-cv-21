#!/bin/bash

# Install pip packages
pip install gradio==3.16.2 \
            albumentations==1.3.0 \
            opencv-contrib-python==4.10.0.84 \
            imageio==2.9.0 \
            imageio-ffmpeg==0.4.2 \
            pytorch-lightning==1.5.0 \
            omegaconf==2.1.1 \
            test-tube>=0.7.5 \
            streamlit==1.12.1 \
            einops==0.3.0 \
            transformers==4.19.2 \
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
            pycocotools==2.0.7

echo "Conda environment 'dad_control' has been created and packages have been installed."
