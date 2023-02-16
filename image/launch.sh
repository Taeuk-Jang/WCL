#!/bin/bash

# CUDA_VISIBLE_DEVICES=0,1, python main_dcl_image.py --dataset_name cub --batch_size 256 --tau_plus 0.3 --epochs 400
# CUDA_VISIBLE_DEVICES=0,1, python main_dcl_image.py --dataset_name celeba --batch_size 256 --tau_plus 0.3 --epochs 20

python linear_binary.py --dataset_name celeba --model_path ../results/celeba/hcl/celeba_hard_model_256_0.3_1.0_20.pth --epochs 20 --lr 1e-5




# CUDA_VISIBLE_DEVICES=0,1,2,3 python main_image.py --dataset_name celeba --batch_size 256 --estimator hard --tau_plus 0.3 --beta 1.0 --epochs 20