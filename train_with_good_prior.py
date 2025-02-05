import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from utils import get_dataset, get_model, test_DNN, test_BNN, train_BNN
from distill import get_conv_layers
from torch.distributions import Normal
import argparse
from termcolor import colored

def main():
    
    class opt:
        
        type = 'dnn'
        model = 'resnet20'
        moped = False
        multi_moped = False
        multi_gpu = False
        data = 'cifar'
        bs = 512
        
    args_dnn = opt()
    args_bnn = opt()
    args_bnn.type = 'uni'
    args_bnn.train_sampler = False
    args_bnn.t = 1.0
    args_bnn.bs = 512
            
    dnn = get_model(args_dnn)
    ckpt = torch.load(args.weight)
    dnn.load_state_dict(ckpt)
    train_loader, test_loader = get_dataset(args_dnn)

    bnn = get_model(args_bnn)

    acc, loss = test_DNN(dnn, test_loader)
    print(colored(f"Acc: {acc:.2f}%, Loss: {loss:.4f}", 'green'))

    # Calculate Sparsity
    total = 0
    zero = 0
    for name, param in dnn.named_parameters():
        if 'weight' in name:
            total += param.numel()
            zero += torch.sum(param == 0).item()
    sparsity = zero/total
    print(f"Sparsity: {sparsity*100:.2f}%")

    dnn_conv_layers = get_conv_layers(dnn)
    bnn_conv_layers = get_conv_layers(bnn)

    for dnn_layer, bnn_layer in zip(dnn_conv_layers, bnn_conv_layers):
        
        mu = dnn_layer.weight.detach().cpu().clone()#.flatten().detach().cpu().clone()
        std = torch.where(mu == 0 , torch.ones_like(mu), torch.ones_like(mu) * 1e-3)
        bnn_layer.prior_weight_mu = mu
        bnn_layer.prior_weight_sigma = std
        
        # raise ValueError('Stop Here')

    import datetime
    from torch.utils.tensorboard import SummaryWriter
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # log_path를 위한 파라미터들을 dict로 구성
    log_params = {
        'data': 'cifar',
        'model': 'resnet20',
        'date': date.split('-')[0],
        'type': 'uni',
        'bs': 512,
        'lr': 1e-3,
        'mc_runs': 10,
        'temp': 1.0,
        'epochs': 1000,
        'kd': True,
        'martern': True,
        'alpha': 0,
        'moped': False,
        'multi_moped': False,
        'timestamp': date,
        'sparsity': sparsity
        }

        # log_params의 항목들을 key=value 형식으로 자동으로 조합하여 log_path 구성
    params_str = "_".join([f"{key}_{value}" for key, value in log_params.items() if key not in ['data', 'model', 'date', 'type']])
    
    log_path = f"runs/{log_params['data']}/{log_params['model']}/{log_params['date']}/{log_params['type']}/{log_params['sparsity']}/{params_str}"
        
    writer = SummaryWriter(log_path)

    train_BNN(
        epoch = 1000,
        model = bnn.cuda(),
        train_loader = train_loader,
        test_loader = test_loader,
        optimizer = optim.Adam(bnn.parameters(), lr=1e-3),
        writer = writer,
        mc_runs = 1,
        bs = 512,
        device = 'cuda',
        args = args_bnn
        
    )

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train a Bayesian Neural Network')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--mc_runs', type=int, default=1, help='Number of Monte Carlo runs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--bs', type=int, default=512, help='Batch size')
    parser.add_argument('--model', type=str, default='simple', help='Model to train [simple, lenet, vgg7, resnet20]')
    parser.add_argument('--type', type=str, default='dnn', help='Type of model [dnn, uni, multi]')
    parser.add_argument('--multi-gpu', action='store_true', help='Use multi-GPU')
    parser.add_argument('--t', type=float, default=1.0, help='Cold Posterior temperature')
    parser.add_argument('--data', type=str, default='mnist', help='Dataset to use [mnist, cifar]')
    parser.add_argument('--train_sampler', type=bool, default=False, help='Do not use this argument')
    parser.add_argument('--distill', action='store_true', help='Use distillation')
    parser.add_argument('--weight', type=str, help='DNN weight path for distillation')
    parser.add_argument('--moped', action='store_true', help='Use MOPED')
    parser.add_argument('--alpha', type=float, default= 0.0, help = 'Distill Coefficient')
    parser.add_argument('--martern', action='store_true', help='Use Martern Prior')
    parser.add_argument('--multi_moped', action='store_true', help='Use Multi-MOPED')
    parser.add_argument('--prune', action='store_true', help='Use pruning')
    args = parser.parse_args()
    
    main()