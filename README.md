## This is repository for NeurIPS 2022 Submission of Difficulty based contrastive learning.

The code is based on SIMCLR (fundamental contrastive learning paper).
https://github.com/sthalles/SimCLR.git
Please follow the instruction in the repository to download the datasets.

Most of the implemented code is on loss functions, data loader, and in jupyter notebook to visualize the results in tsne visualization.

main-wctr.py : implementation of the proposed weighted contrastive learning network.
    functions
    - triplet : triplet loss to train biased encoder.
                Here, I tried different versions of triplet losses.
                eg.,
                    1) ||x-x^+||_2 - \sum^N_i ||x-x^-(i)||_2
                    2) \sum^N_i ( ||x-x^+||_2 - ||x-x^-(i)||_2 )
                    
                    1-debiased) ||x-x^+||_2 - 1/\tau^- ( \sum^N_i  ||x-x^-(i)||_2 - \tau^+ * ||x-x^+||_2))
                    2-debiased) \sum^N_i ( ||x-x^+||_2 - 1/\tau^- (||x-x^-(i)||_2 - \tau^+ * ||x-x^+||_2))
                    
                    
    - W        : measures relative difficulty of sample based on the embedding from biased/debiased encoders.
    - criterion: compute weighted contrastive learning loss.
    
    
linear.py     : This module is used for finetuning with features from penultimate layer.

test_triplet  : This module is used for sanity check of the biased encoder.

visualization.ipynb : This notebook is used to illustrate t-SNE visualization of the methods as in the term paper.

utils.py      : This includes dataloaders for all dataset that is used for experiments.
                This includes: CIFAR-10, CIFAR-100, STL10, CUB, CelebA, Adult, COMPAS.



## Citation
```
@inproceedings{jang2023difficulty,
  title={Difficulty-Based Sampling for Debiased Contrastive Representation Learning},
  author={Jang, Taeuk and Wang, Xiaoqian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={24039--24048},
  year={2023}
}
```
