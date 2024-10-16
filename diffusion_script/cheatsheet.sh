python3 generate_with_imprior.py \
        -a "/data/ephemeral/home/dataset/train.json" \
        -i /data/ephemeral/home/dataset/ \
        -e "/data/ephemeral/home/media/data/ControlAug/cnet/experiments/coco10s1_512p" \
        -l 4883 \
        -s 2 \
        -p 512 \
        -m 0 \
        --vpg_mode HED \
        --ckpt_path /data/ephemeral/home/media/wacv/ControlNet/models/control_sd15_hed.pth \
        --seed 1 \
        --prompt_mode blip_large \
        --model_config_path '/data/ephemeral/home/github/ControlNet/models/cldm_v15.yaml'
