import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

import os
import shutil
import pickle
import logging
from logger import *

import utils
from model import Model, Image_Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Home device: {}'.format(device))

class Net(nn.Module):
    def __init__(self, num_class, pretrained_path):
        super(Net, self).__init__()

        # encoder
        model = Image_Model().to(device)
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(pretrained_path, map_location='cuda:0'))
       # self.f=model.f
        self.f = model.module.model
        # classifier
        self.fc = nn.Linear(512, num_class, bias=True)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out


# train or test for one epoch
def train_val_binary(net, data_loader, train_optimizer):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    num_class = len(data_loader.dataset.classes)
    num_sens = len(data_loader.dataset.sens_groups)
    num_group = num_sens * num_class
    
    total_loss, total_correct, total_num, data_bar = 0.0, 0.0, 0, tqdm(data_loader)
    total_correct_min = np.array([0] * num_group)
    total_num_class = np.array([0] * num_group)
    
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, sens, target in data_bar:
            data, sens, target = data.to(device,non_blocking=True), sens.to(device,non_blocking=True), target.to(device,non_blocking=True)
            out = net(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            group = abs(target) * 2 + abs(sens)
            
            for i in range(num_group):
                idx = group == i
                total_correct_min[i] += torch.sum((prediction[idx, 0:1] == target[idx].unsqueeze(dim=-1)).any(dim=-1).float()).item()
                total_num_class[i] += sum(idx).item()

            desc = '{} Epoch: [{}/{}] Loss: {:.4f} ACC: {:.2f}% ||'\
                         .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,\
                         total_correct / total_num * 100)

            for i in range(num_group):
                desc += 'group {} acc : {:.2f}%, '.format(i, (total_correct_min[i] / total_num_class[i]) * 100)
            
            data_bar.set_description(desc)

        logger_train.info(desc)
    return total_loss / total_num, total_correct / total_num * 100, min(total_correct_min / total_num_class) * 100, np.argmin(total_correct_min / total_num_class), \
                    [(total_correct_min[i] / total_num_class[i]) * 100 for i in range(num_group)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Evaluation')
    parser.add_argument('--root', type=str, default='../data', help='Path to data directory')
    parser.add_argument('--model_path', type=str, default='../results/cifar10/wcl/no_orient/cifar10_model_b_256_0.1_0.001_250.pth',
                        help='The pretrained model path')
    parser.add_argument('--batch_size', type=int, default=256, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')
    parser.add_argument('--dataset_name', default='cifar10', type=str, help='Choose loss function')
    parser.add_argument('--model_name', default='wcl', type=str, help='Choose model name')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')

    args = parser.parse_args()
    model_path, batch_size, epochs = args.model_path, args.batch_size, args.epochs
    dataset_name = args.dataset_name
    lr = args.lr
    save_dir = os.path.split(model_path)[0]
    model_config = os.path.split(model_path)[1]
#     save_dir = '../results/{}'.format(args.dataset_name)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
        #     train_logger = Logger(train_log_dir)
    setup_logger('log_train', os.path.join(save_dir, 'log_cls_b_{}.txt'.format(model_config)))
    logger_train = logging.getLogger('log_train')
    shutil.copyfile('linear_binary.py', os.path.join(save_dir, 'linear_binary.py'))

    logger_train.info(sys.argv)
    save_pkl = os.path.join(save_dir, 'linear_b_summary_{}.pkl'.format(args.model_name))
    
    train_data, _, test_data = utils.get_dataset(dataset_name, root=args.root, pair=False)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = Net(num_class=len(train_data.classes), pretrained_path=model_path).to(device)
    for param in model.f.parameters():
        param.requires_grad = False
    model = nn.DataParallel(model)

    optimizer = optim.SGD(model.module.fc.parameters(), lr=lr, weight_decay=1e-4, momentum = 0.9)
    
    loss_criterion = nn.CrossEntropyLoss()
    results = {'train_loss': [], 'train_acc': [], 'train_acc_group': [], 'train_acc(min)' : [], 'worst_group':[],
               'test_loss': [], 'test_acc': [], 'test_acc_group': [], 'test_acc(min)' : [], 'worst_group(test)':[]}

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_acc_min, worst_group, group_acc = train_val_binary(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['train_acc_group'].append(group_acc)
        results['train_acc(min)'].append(train_acc_min)
        results['worst_group'].append(worst_group)
        if epoch % 5 == 0:
            test_loss, test_acc, test_acc_min, worst_group, group_acc = train_val_binary(model, test_loader, None)

#             os.makedirs('../results/')
#             try:
#                 results=pickle.load( open(save_dir, "rb" ))
#             except:
#                 results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [], 'train_acc(min)' : [],
#                'test_loss': [], 'test_acc@1': [], 'test_acc@5': [], 'test_acc(min)' : []}
#             results[model_path]=test_acc_1
            results['test_loss'].append(test_loss)
            results['test_acc'].append(test_acc)
            results['test_acc_group'].append(group_acc)
            results['test_acc(min)'].append(test_acc_min)
            results['worst_group(test)'].append(worst_group)
            
            pickle.dump(results, open( save_pkl, "wb" ) )

    torch.save(model.state_dict(), os.path.join(save_dir, 'summary_{}.pth'.format(model_config)))
