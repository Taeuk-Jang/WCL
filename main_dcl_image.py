import argparse
import os

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import utils
from model import Model, Image_Model
import logging
from logger import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Home device: {}'.format(device))

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


def train(net, data_loader, train_optimizer, temperature, debiased, tau_plus):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    
    
    if dataset_name in ['celeba', 'cub']:
        for pos_1, pos_2, _, target in train_bar:
            pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
            feature_1, out_1 = net(pos_1)
            feature_2, out_2 = net(pos_2)

            # neg score
            out = torch.cat([out_1, out_2], dim=0)
            neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
            mask = get_negative_mask(batch_size).cuda()
            neg = neg.masked_select(mask).view(2 * batch_size, -1)

            # pos score
            pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
            pos = torch.cat([pos, pos], dim=0)

            # estimator g()
            if debiased:
                N = batch_size * 2 - 2
                Ng = (-tau_plus * N * pos + neg.sum(dim = -1)) / (1 - tau_plus)
                # constrain (optional)
                Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
            else:
                Ng = neg.sum(dim=-1)

            # contrastive loss
            loss = (- torch.log(pos / (pos + Ng) )).mean()

            train_optimizer.zero_grad()
            loss.backward()
            train_optimizer.step()

            total_num += batch_size
            total_loss += loss.item() * batch_size

            train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
    else:
        for pos_1, pos_2, target in train_bar:
            pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
            feature_1, out_1 = net(pos_1)
            feature_2, out_2 = net(pos_2)

            # neg score
            out = torch.cat([out_1, out_2], dim=0)
            neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
            mask = get_negative_mask(batch_size).cuda()
            neg = neg.masked_select(mask).view(2 * batch_size, -1)

            # pos score
            pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
            pos = torch.cat([pos, pos], dim=0)

            # estimator g()
            if debiased:
                N = batch_size * 2 - 2
                Ng = (-tau_plus * N * pos + neg.sum(dim = -1)) / (1 - tau_plus)
                # constrain (optional)
                Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
            else:
                Ng = neg.sum(dim=-1)

            # contrastive loss
            loss = (- torch.log(pos / (pos + Ng) )).mean()

            train_optimizer.zero_grad()
            loss.backward()
            train_optimizer.step()

            total_num += batch_size
            total_loss += loss.item() * batch_size

            train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        
        if dataset_name in ['celeba', 'cub']:
            feature_labels = []
            for data, _, sens, target in tqdm(memory_data_loader, desc='Feature extracting'):
                feature, out = net(data.to(device, non_blocking=True))
                feature_bank.append(feature)
                feature_labels.extend(target * 2 + sens)
                
                total_correct = np.array([0] * c)
                total_num_class = np.array([0] * c)
        else:
            for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
                feature, out = net(data.cuda(non_blocking=True))
                feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        if 'cifar' in dataset_name:
            feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device) 
        elif 'stl' in dataset_name:
            feature_labels = torch.tensor(memory_data_loader.dataset.labels, device=feature_bank.device) 
        elif dataset_name in ['celeba', 'cub']:
            feature_labels = torch.tensor(feature_labels, device=feature_bank.device) 

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
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
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
                total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
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
    parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--debiased', default=True, type=bool, help='Debiased contrastive loss or standard loss')
    parser.add_argument('--dataset_name', default='stl10', type=str, help='Choose loss function')

    # args parse
    args = parser.parse_args()
    feature_dim, temperature, tau_plus, k = args.feature_dim, args.temperature, args.tau_plus, args.k
    batch_size, epochs, debiased = args.batch_size, args.epochs,  args.debiased
    dataset_name = args.dataset_name

    # data prepare
    train_data, memory_data, test_data = utils.get_dataset(dataset_name, root=args.root)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # model setup and optimizer config
    model = Image_Model().cuda()
    model = nn.DataParallel(model)

    optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-6)
    if dataset_name in ['celeba', 'cub']:
        c = len(memory_data.sens_groups) * len(memory_data.classes)
    else:
        c = len(memory_data.classes)
        
    print('# Classes: {}'.format(c))

    if dataset_name == 'cub':
        test_epoch = 100
    elif dataset_name == 'celeba':
        test_epoch = 10
        
    # training loop
    save_dir = '../results/{}/dcl'.format(dataset_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    setup_logger('log_train', os.path.join(save_dir, 'log_{}_{}_{}.txt'.format(batch_size,tau_plus, 1e-3)))
    logger_train = logging.getLogger('log_train')
    
    logger_train.info(sys.argv)
    
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, temperature, debiased, tau_plus)
        if epoch % test_epoch == 0:
            test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)
            torch.save(model.state_dict(), '../results/{}/dcl/model_{}.pth'.format(dataset_name, epoch))
