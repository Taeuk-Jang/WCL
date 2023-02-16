# python main-wctr.py --dataset_name cifar100 --tau_plus 0.05 --epochs 600
# python linear.py --dataset_name cifar100 --model_path ../results/cifar100/wcl/biased_tri_large_lr/cifar100_hard_model_d_256_0.05_0.001_600.pth


# CUDA_VISIBLE_DEVICES=1,2,3 python main-wctr.py --dataset_name celeba --tau_plus 0.3 --epochs 20 --batch_size 256 --feature_dim 256 --lr_b 1e-3
# CUDA_VISIBLE_DEVICES=1,2,3 python main-wctr.py --dataset_name celeba --tau_plus 0.1 --epochs 20 --batch_size 256 --feature_dim 256
# CUDA_VISIBLE_DEVICES=1,2,3 python main-wctr.py --dataset_name celeba --tau_plus 0.05 --epochs 20 --batch_size 256 --feature_dim 256

python linear_binary.py --dataset_name celeba --model_path ../results/celeba/wcl/no_orient_new/celeba_model_d_256_0.3_0.5_0.001_20.pth --epochs 20 --lr 1e-5
python linear_binary.py --dataset_name celeba --model_path ../results/celeba/wcl/no_orient_new/celeba_model_b_256_0.3_0.5_0.001_20.pth --epochs 20 --lr 1e-5

# python linear_binary.py --dataset_name celeba --model_path ../results/celeba/wcl/no_orient_new/celeba_model_d_256_0.1_0.001_20.pth --epochs 20 --lr 1e-5
# python linear_binary.py --dataset_name celeba --model_path ../results/celeba/wcl/no_orient_new/celeba_model_b_256_0.1_0.01_20.pth --epochs 20 --lr 1e-5

# python linear_binary.py --dataset_name celeba --model_path ../results/celeba/wcl/no_orient_new/celeba_model_d_256_0.05_0.001_20.pth --epochs 20 --lr 1e-5
# python linear_binary.py --dataset_name celeba --model_path ../results/celeba/wcl/no_orient_new/celeba_model_b_256_0.05_0.01_20.pth --epochs 20 --lr 1e-5

# python main-wctr.py --dataset_name celeba --tau_plus 0.20 --epochs 600

