import os
import argparse
import pdb  # 디버거 모듈 추가

# 기본 설정 (원하는 모드로 수정 가능)
DEFAULT_PORT = "8097"
MODE = "train"  # "train" 또는 "predict"로 설정 가능

pdb.set_trace()  # 여기서 실행이 중단됨 (중단점)


#  n (next): 다음 줄로 이동
#  s (step): 함수 내부로 진입
#  c (continue): 다음 중단점까지 계속 실행
#  q (quit): 디버깅 종료


if MODE == "train":
    os.system(f"python C:/pyproject/EnlightenGAN/train.py \
        --dataroot ../final_dataset \
        --no_dropout \
        --name enlightening \
        --model single \
        --dataset_mode unaligned \
        --which_model_netG sid_unet_resize \
        --which_model_netD no_norm_4 \
        --patchD \
        --patch_vgg \
        --patchD_3 5 \
        --n_layers_D 5 \
        --n_layers_patchD 4 \
        --fineSize 320 \
        --patchSize 32 \
        --skip 1 \
        --batchSize 32 \
        --self_attention \
        --use_norm 1 \
        --use_wgan 0 \
        --use_ragan \
        --hybrid_loss \
        --times_residual \
        --instance_norm 0 \
        --vgg 1 \
        --vgg_choose relu5_1 \
        --gpu_ids 0,1,2 \
        --display_port={DEFAULT_PORT}"
    )

elif MODE == "predict":
    for i in range(1):
        os.system(f"python predict.py \
            --dataroot ../test_dataset \
            --name enlightening \
            --model single \
            --which_direction AtoB \
            --no_dropout \
            --dataset_mode unaligned \
            --which_model_netG sid_unet_resize \
            --skip 1 \
            --use_norm 1 \
            --use_wgan 0 \
            --self_attention \
            --times_residual \
            --instance_norm 0 \
            --resize_or_crop='no' \
            --which_epoch {200 - i * 5}"
        )
