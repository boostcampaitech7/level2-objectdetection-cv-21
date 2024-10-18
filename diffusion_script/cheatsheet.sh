python3 generate_with_imprior.py
        -a "/data/ephemeral/home/dataset/train_split.json"
        -i /data/ephemeral/home/dataset/
        -e "/data/ephemeral/home/media/data/ControlAug/cnet/experiments/cat1172_random_1024"
        -l 1172
        -s 1
        -p 1024
        -m 0
        --vpg_mode HED
        --ckpt_path /data/ephemeral/home/media/wacv/ControlNet/models/control_sd15_hed.pth
        --seed 1
        --prompt_mode cat
        --model_config_path '/data/ephemeral/home/github/ControlNet/models/cldm_v15.yaml'
        # 한 줄로 붙여넣기를 사용하세요


python3 mix_annotations.py \
        -a "/data/ephemeral/home/dataset/train.json" \
        --gt_image_folder "/data/ephemeral/home/dataset/" \
        -s "/data/ephemeral/home/media/data/ControlAug/cnet/experiments/cat300_inorder_1024/syn_n300_HED_p1024_prcat_dfsNone_seed1_imprior" \
        -t "/data/ephemeral/home/media/data/ControlAug/cnet/experiments/cat300_inorder_1024/mix_n300_HED_p1024_prcat_dfsNone_seed1_imprior" \
        -f nofilter \
        -n 1172 # 총 생성된 이미지 수