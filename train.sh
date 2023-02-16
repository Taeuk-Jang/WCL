# python main-wctr.py --dataset_name cifar100 --tau_plus 0.05 --epochs 600
# python linear.py --dataset_name cifar100 --model_path ../results/cifar100/wcl/biased_tri_large_lr/cifar100_hard_model_d_256_0.05_0.001_600.pth


# python main-wctr.py --dataset_name cub --tau_plus 0.1 --epochs 400 --batch_size 256 --feature_dim 256  --lr_b 1e-3
# python main-wctr.py --dataset_name cub --tau_plus 0.1 --epochs 400 --batch_size 256 --feature_dim 256 --temperature 1 --lr_b 1e-3
# python main-wctr.py --dataset_name cub --tau_plus 0.1 --epochs 400 --batch_size 256 --feature_dim 256 --temperature 0.3 --lr_b 1e-3


# python linear_binary.py --dataset_name cub --model_path ../results/cub/wcl/no_orient_new/cub_model_d_256_0.1_0.3_0.001_100.pth --epochs 150
# # python linear_binary.py --dataset_name cub --model_path ../results/cub/wcl/no_orient_new/cub_model_b_256_0.1_1_0.001_400.pth --epochs 150

# python linear_binary.py --dataset_name cub --model_path ../results/cub/wcl/no_orient_new/cub_model_d_256_0.1_0.5_0.001_400.pth --epochs 150
# # python linear_binary.py --dataset_name cub --model_path ../results/cub/wcl/no_orient_new/cub_model_b_256_0.1_0.3_0.001_400.pth --epochs 150

# python linear_binary.py --dataset_name cub --model_path ../results/cub/wcl/no_orient_new/cub_model_d_256_0.1_1.0_0.001_400.pth --epochs 150
# # python linear_binary.py --dataset_name cub --model_path ../results/cub/wcl/no_orient_new/cub_model_b_256_0.1_0.5_0.001_400.pth --epochs 150

 

# python main-wctr.py --dataset_name cub --tau_plus 0.20 --epochs 600



# ### STL

# python main-wctr.py --dataset_name imagenet --tau_plus 0.01 --epochs 500 --batch_size 256 --feature_dim 256  --lr_d 3e-2 --lr_b 1e-2 --temperature 0.07
# python main-wctr.py --dataset_name imagenet --tau_plus 0.01 --epochs 500 --batch_size 256 --feature_dim 256  --lr_d 3e-2 --lr_b 5e-2 --temperature 0.07
# python main-wctr.py --dataset_name imagenet --tau_plus 0.01 --epochs 500 --batch_size 256 --feature_dim 256  --lr_d 3e-2 --lr_b 5e-3 --temperature 0.07


# python linear_binary.py --dataset_name cub --model_path ../results/cub/wcl/no_orient_new/cub_model_d_256_0.1_0.3_0.001_100.pth --epochs 150
CUDA_VISIBLE_DEVICES=2,3 python linear_binary.py --dataset_name cub --model_path ../results/cub/wcl/no_orient_new/cub_model_b_256_0.1_1.0_0.001_400.pth --epochs 150

# python linear_binary.py --dataset_name cub --model_path ../results/cub/wcl/no_orient_new/cub_model_d_256_0.1_0.5_0.001_400.pth --epochs 150
CUDA_VISIBLE_DEVICES=2,3 python linear_binary.py --dataset_name cub --model_path ../results/cub/wcl/no_orient_new/cub_model_b_256_0.1_0.5_0.001_400.pth --epochs 150
