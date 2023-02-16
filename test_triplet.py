## The code is based on SIMCLR (fundamental contrastive learning paper).
# https://github.com/sthalles/SimCLR.git
# Most of the implemented code is on my loss functions, data loader, and in jupyter notebook to visualize the results in tsne visualization.

import argparse
import os
import pandas

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from model import Model

# torch.set_printoptions(profile="full")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Home device: {}'.format(device))

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask

# def triplet(out_1,out_2,batch_size):
#     #biased representation learning
#     out = torch.cat([out_1, out_2], dim=0) # 2 * bs x fs
#     s = (torch.pow(out.unsqueeze(0) - out.unsqueeze(1), 2) / temperature).sum(-1)
    
#     mask = get_negative_mask(batch_size).to(device)
#     s = s.masked_select(mask).view(2 * batch_size, -1)  # (2 * bs, bs - 2) : subtract self and its augment

#     pos = (torch.pow(out_1 - out_2, 2)  / temperature).sum(-1)
#     neg = s.sum(-1)

# #     return torch.clamp(pos - neg + 5, min = 0)
#     return pos.mean() - neg.mean()

# def triplet(out_1,out_2,tau_plus,estimator,batch_size,temperature):
#     #debiased easy amplifying representation learning
#     N = batch_size * 2 - 2
    
#     out = torch.cat([out_1, out_2], dim=0) # 2 * bs x fs
#     s = torch.pow(out.unsqueeze(0) - out.unsqueeze(1), 2).sum(-1) # 2 * bs x 2 * bs
    
#     mask = get_negative_mask(batch_size).to(device)
#     s = s.masked_select(mask).view(2 * batch_size, -1)  # (2 * bs, 2 * bs - 2) : subtract self and its augment

#     pos = torch.pow(out_1 - out_2, 2) # 2 * bs x fs
#     pos = torch.cat([pos, pos], dim=0).sum(-1) # 2 * bs x 1
    
#     if estimator == 'biased':
#         neg = s.sum(-1) # 2 * bs x 1
#     else:
#         neg = (-tau_plus * pos + s.mean(-1)) / (1 - tau_plus) # 2 * bs x 1

# #     return torch.clamp(pos - neg + 5, min = 0)
#     return (pos - neg).mean()
def triplet(out_1,out_2,tau_plus,estimator,batch_size,temperature):
    #debiased easy amplifying representation learning
    N = batch_size * 2 - 2
    
    out = torch.cat([out_1, out_2], dim=0) # 2 * bs x fs
    s = torch.pow(out.unsqueeze(0) - out.unsqueeze(1), 2).sum(-1) # 2 * bs x 2 * bs
    
    mask = get_negative_mask(batch_size).to(device)
    s = s.masked_select(mask).view(2 * batch_size, -1)  # (2 * bs, 2 * bs - 2) : subtract self and its augment
    #randomly select one negative per anchor
    s = s.gather(1, torch.randint(0, N, size = (2* batch_size,)).view(-1,1).to(device))

    pos = torch.pow(out_1 - out_2, 2).sum(-1) # 2 * bs x fs
    pos = torch.cat([pos, pos], dim=0) # 2 * bs x 1
    
    if estimator == 'biased':
        neg = s # 2 * bs x 1
    else:
        neg = (-tau_plus * N * pos + s) / (1 - tau_plus) # 2 * bs x 1

#     return torch.clamp(pos - neg + 5, min = 0)
    return (pos - neg).mean()
    
def train(net_b, data_loader, optimizer_b, temperature, estimator, tau_plus, beta):
    net_b.train()
    
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    total_loss_tri = 0
    for pos_1, pos_2, target in train_bar:
        pos_1, pos_2 = pos_1.to(device,non_blocking=True), pos_2.to(device,non_blocking=True)
        
        feature_1, out_1_b = net_b(pos_1)
        feature_2, out_2_b = net_b(pos_2)

        loss_tri = triplet(out_1_b, out_2_b, tau_plus, estimator,batch_size, temperature)

        loss = loss_tri 

        optimizer_b.zero_grad()
        loss.backward()
        optimizer_b.step()

        total_num += batch_size
        total_loss_tri += loss_tri.item() * batch_size
#         total_loss_ori += loss_ori.item() * batch_size

#         train_bar.set_description('Train Epoch: [{}/{}] loss_tri : {:.4f}, loss_ori : {:.4f}, loss_crt: {:.4f}'\
#                   .format(epoch, epochs, total_loss_tri / total_num, total_loss_ori / total_num, total_loss_crt / total_num))
        train_bar.set_description('Train Epoch: [{}/{}] loss_tri : {:.4f}, '\
                  .format(epoch, epochs, total_loss_tri / total_num))


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader, temperature):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.to(device, non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        if 'cifar' in dataset_name:
            feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device) 
        elif 'stl' in dataset_name:
            feature_labels = torch.tensor(memory_data_loader.dataset.labels, device=feature_bank.device) 

        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1).long(), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:,:1] == target.long().unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:,:5] == target.long().unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('KNN Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--root', type=str, default='../data', help='Path to data directory')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--tau_plus', default=0.1, type=float, help='Positive class priorx')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=400, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--estimator', default='biased', type=str, help='Choose loss function')
    parser.add_argument('--dataset_name', default='stl10', type=str, help='Choose loss function')
    parser.add_argument('--beta', default=1.0, type=float, help='Choose loss function')
    parser.add_argument('--anneal', default=None, type=str, help='Beta annealing')

    # args parse
    args = parser.parse_args()
    feature_dim, temperature, tau_plus, k = args.feature_dim, args.temperature, args.tau_plus, args.k
    batch_size, epochs, estimator = args.batch_size, args.epochs,  args.estimator
    dataset_name = args.dataset_name
    beta = args.beta
    anneal = args.anneal

    #configuring an adaptive beta if using annealing method
    if anneal=='down':
        do_beta_anneal=True
        n_steps=9
        betas=iter(np.linspace(beta,0,n_steps))
    else:
        do_beta_anneal=False
    
    # data prepare
    train_data, memory_data, test_data = utils.get_dataset(dataset_name, root=args.root)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True, drop_last=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)

    # model setup and optimizer config
    
    model_b = Model(feature_dim).cuda()
    model_b = nn.DataParallel(model_b)

    lr_d = 1e-3
    
    optimizer_b = optim.Adam(model_b.parameters(), lr=1e-3, weight_decay=1e-6)
    
    c = len(memory_data.classes)
    print('# Classes: {}'.format(c))

    # training loop
    save_dir = '../results/{}/triplet/{}'.format(dataset_name, estimator)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(1, epochs + 1):
        train_loss = train(model_b, train_loader, optimizer_b, temperature, estimator, tau_plus, beta)
        
        if do_beta_anneal is True:
            if epoch % (int(epochs/n_steps)) == 0:
                beta=next(betas)

        if epoch % 25 == 0:
            test_acc_1, test_acc_5 = test(model_b, memory_loader, test_loader, temperature)
            
            torch.save(model_b.state_dict(), os.path.join(save_dir,'{}_{}_model_b_{}_{}_{}_{}.pth'.format(dataset_name,estimator,batch_size,tau_plus,lr_d,epoch)))
