python3 generate_with_imprior.py
        -a "/data/ephemeral/home/dataset/train.json"
        -i /data/ephemeral/home/dataset/
        -e "/data/ephemeral/home/media/data/ControlAug/cnet/experiments/cat20_test_1024"
        -l 10
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
        -s "/data/ephemeral/home/media/data/ControlAug/cnet/experiments/cat20_test_realistic/syn_n20_HED_p512_prcat_dfsNone_seed1_imprior" \
        -t "/data/ephemeral/home/media/data/ControlAug/cnet/experiments/cat20_test_realistic/mix_n20_HED_p512_prcat_dfsNone_seed1_imprior" \
        -f nofilter \
        -n 20 # 총 생성된 이미지 수