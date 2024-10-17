python3 generate_with_imprior.py \
        -a "/data/ephemeral/home/dataset/train.json" \
        -i /data/ephemeral/home/dataset/ \
        -e "/data/ephemeral/home/media/data/ControlAug/cnet/experiments/cat20_test_realistic" \
        -l 20 \ # 총 원본 이미지 수
        -s 1 \ # 생성할 이미지의 배수
        -p 512 \ # 이미지 크기
        -m 0 \
        --vpg_mode HED \
        --ckpt_path /data/ephemeral/home/media/wacv/ControlNet/models/control_sd15_hed.pth \
        --seed 1 \
        --prompt_mode cat \
        --model_config_path '/data/ephemeral/home/github/ControlNet/models/cldm_v15.yaml'


python3 mix_annotations.py \
        -a "/media/data/coco_fsod/seed${SEED}/${SHOT}shot_novel.json" \
        -s syn_file="/data/ephemeral/home/media/data/ControlAug/cnet/experiments/cat20_test_realistic/syn_n20_HED_p512_prcat_dfsNone_seed1_imprior" \
        -t mix_file="/data/ephemeral/home/media/data/ControlAug/cnet/experiments/cat20_test_realistic/mix_n20_HED_p512_prcat_dfsNone_seed1_imprior" \
        -f nofilter \
        -n 20 # 총 원본 이미지 수