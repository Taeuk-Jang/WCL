''' 
We don't need orientation.
If we have orientation, each anchor should be posistioning same place in the representation space.
We only need the relative distance between biased/debiased encoder.
'''


## The code is based on SIMCLR (fundamental contrastive learning paper).
# https://github.com/sthalles/SimCLR.git
# Most of the implemented code is on my loss functions, data loader, and in jupyter notebook to visualize the results in tsne visualization.

import argparse
import os
import pandas
import shutil
import sys

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from model import Model, Image_Model
import logging
from logger import *


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

def triplet(out_1,out_2,tau_plus,batch_size,temperature, debias = True):
    N = batch_size * 2 - 2
    out = torch.cat([out_1, out_2], dim=0) # 2 * bs x fs
    s = torch.pow(out.unsqueeze(0) - out.unsqueeze(1), 2).sum(-1) # 2 * bs x 2 * bs
    mask = get_negative_mask(batch_size).to(device)
    
    if debias:
        s = s.masked_select(mask).view(2 * batch_size, -1)  # (2 * bs, 2 * bs - 2) : subtract self and its augment

        pos = (torch.pow(out_1 - out_2, 2))
        pos = torch.cat([pos, pos], dim=0).sum(-1)

        neg = (-tau_plus * N * pos + s.sum(-1)) / (1 - tau_plus)

    else:
        neg = s.masked_select(mask).view(2 * batch_size, -1)  # (2 * bs, 2 * bs - 2) : subtract self and its augment
        
        pos = (torch.pow(out_1 - out_2, 2))
        pos = torch.cat([pos, pos], dim=0).sum(-1)

    return (pos - neg).mean()
#     return torch.maximum(pos - neg + 256, torch.zeros_like(neg)).mean()

def W(out_d, out_b, batch_size):
    mask = get_negative_mask(batch_size).to(device)
    
    # difficulty by distance
#     s_d =  torch.pow(out_d.unsqueeze(0) - out_d.unsqueeze(1), 2).sum(-1)
#     s_d = s_d.masked_select(mask).view(2 * batch_size, -1) / temperature
#     s_b =  torch.pow(out_b.unsqueeze(0) - out_b.unsqueeze(1), 2).sum(-1)
#     s_b = s_b.masked_select(mask).view(2 * batch_size, -1) / temperature
    
#     difficulty by cosine similarity
    s_d = torch.exp(torch.mm(out_d, out_d.t().contiguous()) / temperature)
    s_d = s_d.masked_select(mask).view(2 * batch_size, -1) # (2 * bs, 2 * bs - 2) : subtract self and its augment
    s_d = F.normalize(s_d, dim = -1)
    
    s_b = torch.exp(torch.mm(out_b, out_b.t().contiguous()) / temperature)
    s_b = s_b.masked_select(mask).view(2 * batch_size, -1) # (2 * bs, 2 * bs - 2) : subtract self and its augment
    s_b = F.normalize(s_b, dim = -1)
    
    weight = 1 + s_d / (s_b + s_d + 1e-6)
#     print(weight)
    if np.isnan(weight.sum().item()):
        print('weight NaN')
        
    return weight.detach()
    
def orientation(out_1_d,out_2_d,out_1_b,out_2_b,batch_size):
    #space sharing
    out_d = torch.cat([out_1_d,out_2_d], dim=0)
    out_b = torch.cat([out_1_b,out_2_b], dim=0)
#     print(out_d)
#     print(out_b)
#     print(nn.MSELoss(reduction = 'sum')(out_d, out_b)/batch_size)
#     return nn.MSELoss(reduction = 'sum')(out_d, out_b)/batch_size
#    return (torch.pow(out_d - out_b, 2) / temperature).mean()
    return -torch.log(torch.exp((out_d * out_b).sum(-1)/temperature)).mean() 
    
def criterion(out_1_d, out_2_d, out_1_b, out_2_b, tau_plus, batch_size, beta, temperature):
    # neg score
    out = torch.cat([out_1_d, out_2_d], dim=0)
    out_b = torch.cat([out_1_b, out_2_b], dim=0)
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = get_negative_mask(batch_size).to(device)
    neg = neg.masked_select(mask).view(2 * batch_size, -1) # (2 * bs, bs - 2) : subtract self and its augment

    # pos score
    pos = torch.exp(torch.sum(out_1_d * out_2_d, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)
    
    weight = W(out, out_b, batch_size) # (2 * bs, bs - 2)

    # negative samples similarity scoring
    N = batch_size * 2 - 2
#         imp = (beta* neg.log()).exp()
#         reweight_neg = (imp*neg).sum(dim = -1) / imp.mean(dim = -1)
    reweight_neg = weight * neg

    Ng = (-tau_plus * N * pos + reweight_neg.sum(dim = -1)) / (1 - tau_plus)
    # constrain (optional)
    Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))

    # contrastive loss
    loss = (-torch.log(pos / (pos + Ng) )).mean()
    
    if np.isnan(loss.mean().item()):
#         print("pos : ", pos)
#         print("Ng : ", Ng)
        print("neg : ", neg)
    
        np.savetxt('pos.txt', pos.detach().cpu().numpy(), delimiter=',')
        np.savetxt('Ng.txt', Ng.detach().cpu().numpy(), delimiter=',')
        np.savetxt('neg.txt', neg.detach().cpu().numpy(), delimiter=',')
        np.savetxt('weight.txt', weight.detach().cpu().numpy(), delimiter=',')

    return loss

def train(net_d, net_b, data_loader, optimizer_d, optimizer_b, temperature, tau_plus, beta, logger_train):
    net_d.train()
    net_b.train()
    
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    total_loss_tri = 0
    total_loss_ori = 0
    total_loss_crt = 0
    
    if dataset_name in ['celeba', 'cub']:
        for pos_1, pos_2, _, target in train_bar:
            pos_1, pos_2 = pos_1.to(device,non_blocking=True), pos_2.to(device,non_blocking=True)

            feature_1, out_1_d = net_d(pos_1)
            feature_2, out_2_d = net_d(pos_2)

            feature_1, out_1_b = net_b(pos_1)
            feature_2, out_2_b = net_b(pos_2)

            loss_tri = triplet(out_1_b, out_2_b, tau_plus, batch_size, temperature, True)
            loss_crt = criterion(out_1_d, out_2_d, out_1_b, out_2_b, tau_plus, batch_size, beta, temperature)

            if np.isnan(loss_crt.item()):
                print('NaN:loss_crt')
                torch.cuda.empty_cache()
                continue

            loss = loss_tri + loss_crt

            optimizer_b.zero_grad()
            optimizer_d.zero_grad()
            loss.backward()
            optimizer_b.step()
            optimizer_d.step()

            total_num += batch_size
            total_loss_tri += loss_tri.item() * batch_size
            total_loss_crt += loss_crt.item() * batch_size

            desc = 'Train Epoch: [{}/{}] loss_tri : {:.4f}, loss_crt: {:.4f}'\
                      .format(epoch, epochs, total_loss_tri / total_num, total_loss_crt / total_num)

            train_bar.set_description(desc)
    else:
        for pos_1, pos_2, target in train_bar:
            pos_1, pos_2 = pos_1.to(device,non_blocking=True), pos_2.to(device,non_blocking=True)

            feature_1, out_1_d = net_d(pos_1)
            feature_2, out_2_d = net_d(pos_2)

            feature_1, out_1_b = net_b(pos_1)
            feature_2, out_2_b = net_b(pos_2)

            loss_tri = triplet(out_1_b, out_2_b, tau_plus, batch_size, temperature, True)
            loss_crt = criterion(out_1_d, out_2_d, out_1_b, out_2_b, tau_plus, batch_size, beta, temperature)

            if np.isnan(loss_crt.item()):
                print('NaN:loss_crt')
                torch.cuda.empty_cache()
                continue

    #         loss = loss_tri + 0.1 * loss_ori + loss_crt
            loss = loss_tri + loss_crt

            optimizer_b.zero_grad()
            optimizer_d.zero_grad()
            loss.backward()
            optimizer_b.step()
            optimizer_d.step()

            total_num += batch_size
            total_loss_tri += loss_tri.item() * batch_size
    #         total_loss_ori += loss_ori.item() * batch_size
            total_loss_crt += loss_crt.item() * batch_size

    #         desc = 'Train Epoch: [{}/{}] loss_tri : {:.4f}, loss_ori : {:.4f}, loss_crt: {:.4f}'\
    #                   .format(epoch, epochs, total_loss_tri / total_num, total_loss_ori / total_num, total_loss_crt / total_num)
            desc = 'Train Epoch: [{}/{}] loss_tri : {:.4f}, loss_crt: {:.4f}'\
                      .format(epoch, epochs, total_loss_tri / total_num, total_loss_crt / total_num)

            train_bar.set_description(desc)
    logger_train.info(desc)
        
#         train_bar.set_description('Train Epoch: [{}/{}] loss_tri : {:.4f}, loss_crt: {:.4f}'\
#                   .format(epoch, epochs, total_loss_tri / total_num, total_loss_crt / total_num))

    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, net_b, memory_data_loader, test_data_loader, temperature, logger_train):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    
    
    with torch.no_grad():
        # generate feature bank
        feature_labels = []
        if dataset_name in ['celeba', 'cub']:
            
            for data, _, sens, target in tqdm(memory_data_loader, desc='Feature extracting'):
                feature, out = net(data.to(device, non_blocking=True))
                feature_bank.append(feature)
                feature_labels.extend(target * 2 + sens)
                
                total_correct = np.array([0] * c)
                total_num_class = np.array([0] * c)
            
        else:
            for data, _,  target in tqdm(memory_data_loader, desc='Feature extracting'):
                feature, out = net(data.to(device, non_blocking=True))
                feature_bank.append(feature)
                feature_labels.extend(target)
        
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        if 'cifar' in dataset_name:
            feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device) 
        elif 'stl' in dataset_name:
            feature_labels = torch.tensor(memory_data_loader.dataset.labels, device=feature_bank.device) 
        elif dataset_name in ['celeba', 'cub']:
            feature_labels = torch.tensor(feature_labels, device=feature_bank.device) 

        feature_labels = torch.tensor(feature_labels).cuda()
        
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        
        if dataset_name in ['celeba', 'cub']:
            for data, _, sens, target in test_bar:
                data, sens, target = data.to(device, non_blocking=True), sens.to(device, non_blocking=True), target.to(device, non_blocking=True)
                
                group = abs(target) * 2 + abs(sens)
                
                feature, out = net(data)

                total_num += data.size(0)

                for i in range(c):
                    total_num_class[i] += torch.sum((group == i).float()).item()

                # compute cos similarity between each feature vector and feature bank ---> [B, N]
                sim_matrix = torch.mm(feature, feature_bank)
                # [B, K]
                sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
                # [B, K]
                sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
                sim_weight = (sim_weight / temperature).exp()

                # [B*K, C]
                one_hot_label = F.one_hot(sim_labels.view(-1, 1).long(), c)
                # weighted score ---> [B, C]
                pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

                pred_labels = pred_scores.argsort(dim=-1, descending=True)
                total_top1 += torch.sum((pred_labels[:,:1] == target.long().unsqueeze(dim=-1)).any(dim=-1).float()).item()

                for i in range(c):
                    total_correct[i] += torch.sum((pred_labels[group == i,:1] == target[group == i].long().unsqueeze(dim=-1)).any(dim=-1).float()).item()

                desc = 'KNN Test Epoch: [{}/{}] Acc :{:.2f}%, '\
                            .format(epoch, epochs, total_top1 / total_num * 100)

                for i in range(c):
                    desc += 'group {} acc : {:.2f}%, '.format(i, (total_correct[i] / total_num_class[i]) * 100)

                test_bar.set_description(desc)
                
                logger_train.info(desc)

            return total_top1 / total_num * 100, [(total_correct[i] / total_num_class[i]) * 100 for i in range(c)]
        
                
        else:
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


                desc = 'KNN Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'\
                            .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100)

                test_bar.set_description(desc)
        
            logger_train.info(desc)

            return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--root', type=str, default='../data', help='Path to data directory')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--tau_plus', default=0.1, type=float, help='Positive class priorx')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--dataset_name', default='cifar10', type=str, help='Choose loss function')
    parser.add_argument('--beta', default=1.0, type=float, help='Choose loss function')
    parser.add_argument('--anneal', default=None, type=str, help='Beta annealing')
    parser.add_argument('--lr_b', default=1e-2, type=float, help='Beta annealing')
    parser.add_argument('--lr_d', default=1e-3, type=float, help='Beta annealing')

    # args parse
    args = parser.parse_args()
    feature_dim, temperature, tau_plus, k = args.feature_dim, args.temperature, args.tau_plus, args.k
    batch_size, epochs = args.batch_size, args.epochs
    dataset_name = args.dataset_name
    beta = args.beta

    #configuring an adaptive beta if using annealing method
    
    # data prepare
    train_data, memory_data, test_data = utils.get_dataset(dataset_name, root=args.root)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=24, pin_memory=True, drop_last=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=24, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=24, pin_memory=True)

    
    if dataset_name in ['celeba', 'cub', 'imagenet']:
        # model setup and optimizer config
        model_d = Image_Model().cuda()
        model_d = nn.DataParallel(model_d)

    #     model_path = '../results/cifar10/wcl/no_orient/cifar10_model_d_256_0.1_0.0001_475.pth'
    #     state_dict = torch.load(model_path)
    #     model_d.load_state_dict(state_dict)

        model_b = Image_Model().cuda()
        model_b = nn.DataParallel(model_b)
    
    else:
        # model setup and optimizer config
        model_d = Model(feature_dim).cuda()
        model_d = nn.DataParallel(model_d)

    #     model_path = '../results/cifar10/wcl/no_orient/cifar10_model_d_256_0.1_0.0001_475.pth'
    #     state_dict = torch.load(model_path)
    #     model_d.load_state_dict(state_dict)

        model_b = Model(feature_dim).cuda()
        model_b = nn.DataParallel(model_b)

#     model_path = '../results/cifar10/wcl/no_orient/cifar10_model_b_256_0.1_0.0001_475.pth'
#     state_dict = torch.load(model_path)
#     model_b.load_state_dict(state_dict)

    '''
    gamma = 0.5 after 350 epochs
    initially
    1r_d = 1e-3
    lr_b = 1e-2
    at 400 epochs
    1r_d = 1e-4
    lr_b = 1e-3
    '''
    
    if dataset_name == 'celeba':
        test_epoch = 20
    else:
        test_epoch = 100
    
    if dataset_name in ['celeba', 'cub', 'imagenet']:
        optimizer_d = optim.SGD(model_d.parameters(), lr=args.lr_d, weight_decay=1e-4, momentum = 0.9)
        optimizer_b = optim.SGD(model_b.parameters(), lr=args.lr_b, weight_decay=1e-4, momentum = 0.9)
    else:
        optimizer_d = optim.Adam(model_d.parameters(), lr=args.lr_d, weight_decay=1e-6)
        optimizer_b = optim.Adam(model_b.parameters(), lr=args.lr_b, weight_decay=1e-6)
    
    scheduler_b = optim.lr_scheduler.MultiStepLR(optimizer_b, milestones=[350], gamma=0.1)
    scheduler_d = optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=[350], gamma=0.1)
    
    if dataset_name in ['celeba', 'cub']:
        c = len(memory_data.sens_groups) * len(memory_data.classes)
    else:
        c = len(memory_data.classes)
        
    print('# Classes: {}'.format(c))

    # training loop
    save_dir = '../results/{}/wcl/CVPR23'.format(dataset_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    #     train_logger = Logger(train_log_dir)
    setup_logger('log_train', os.path.join(save_dir, 'log_{}_{}_{}.txt'.format(batch_size,tau_plus,args.lr_d)))
    logger_train = logging.getLogger('log_train')
    
    logger_train.info(sys.argv)
    shutil.copyfile('main-wctr.py', os.path.join(save_dir, 'main-wctr.py'))

    for epoch in range(0, epochs + 1):
        train_loss = train(model_d, model_b, train_loader, optimizer_d, optimizer_b, temperature, tau_plus, beta, logger_train)

        if epoch % test_epoch == 0:
            test_acc_1, test_acc_5 = test(model_d, model_b, memory_loader, test_loader, temperature, logger_train)
            
            torch.save(model_d.state_dict(), os.path.join(save_dir,'{}_model_d_{}_{}_{}_{}_{}_{}.pth'.format(dataset_name,batch_size,tau_plus,temperature,args.lr_d,args.lr_b,epoch)))
            torch.save(model_b.state_dict(), os.path.join(save_dir,'{}_model_b_{}_{}_{}_{}_{}_{}.pth'.format(dataset_name,batch_size,tau_plus,temperature,args.lr_d,args.lr_b,epoch)))
            
            
        scheduler_b.step()
        scheduler_d.step()
        
    torch.save(model_d.state_dict(), os.path.join(save_dir,'{}_model_d_{}_{}_{}_{}_{}_{}.pth'.format(dataset_name,batch_size,tau_plus,temperature,args.lr_d,args.lr_b,epoch)))
    torch.save(model_b.state_dict(), os.path.join(save_dir,'{}_model_b_{}_{}_{}_{}_{}_{}.pth'.format(dataset_name,batch_size,tau_plus,temperature,args.lr_d,args.lr_b,epoch)))
