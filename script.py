import os
import argparse

DEFAULT_PORT = "8097"
MODE = "train"

if MODE == "train":
    os.system(f"python train.py \
        --dataroot ../final_dataset \
        --no_dropout \
        --name enlightening \
        --model single \
        --dataset_mode unaligned \
        --which_model_netG sid_unet_resize_student \
        --which_model_netD no_norm_4 \
        --patchD \
        --patch_vgg \
        --patchD_3 5 \
        --n_layers_D 5 \
        --n_layers_patchD 4 \
        --fineSize 320 \
        --patchSize 32 \
        --skip 1 \
        --batchSize 8 \
        --self_attention \
        --use_norm 1 \
        --use_wgan 0 \
        --use_ragan \
        --hybrid_loss \
        --times_residual \
        --instance_norm 0 \
        --vgg 1 \
        --vgg_choose relu5_1 \
        --gpu_ids 0 \
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

