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
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn

def main():
    
    class opt:  
        type = 'dnn'
        model = 'lenet'
        moped = False
        multi_moped = False
        multi_gpu = False
        data = 'cifar'
        bs = 512
        train_sampler = False
        
    args_dnn = opt()
    args_bnn = opt()
    args_bnn.type = 'uni'
    args_bnn.train_sampler = False
    
    dnn = get_model(args_dnn)
    ckpt = torch.load(args.weight)
    dnn.load_state_dict(ckpt)
    train_loader, test_loader = get_dataset(args_dnn)

    const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",  # Flipout or Reparameterization
        "moped_enable": True,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.5,
    }
        
    dnn_to_bnn(dnn, const_bnn_prior_parameters)

    import datetime
    from torch.utils.tensorboard import SummaryWriter
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # log_path를 위한 파라미터들을 dict로 구성
    log_params = {
        'data': 'cifar',
        'model': 'lenet',
        'date': date.split('-')[0],
        'type': 'uni',
        'bs': 512,
        'lr': 1e-3,
        'mc_runs': 1,
        'temp': 1.0,
        'epochs': 1000,
        'kd': True,
        'martern': True,
        'alpha': 0,
        'moped': False,
        'multi_moped': False,
        'timestamp': date,
        'sparsity': 'MODEP'
        }

        # log_params의 항목들을 key=value 형식으로 자동으로 조합하여 log_path 구성
    params_str = "_".join([f"{key}_{value}" for key, value in log_params.items() if key not in ['data', 'model', 'date', 'type']])
    
    log_path = f"runs/{log_params['data']}/{log_params['model']}/{log_params['date']}/{log_params['type']}/{log_params['sparsity']}/{params_str}_pruned_more_small_other_more_bigger"
        
    writer = SummaryWriter(log_path)

    args_bnn.moped = True
    args_bnn.t = 1.0
    args_bnn.bs = 512
    
    train_BNN(
        epoch = 1000,
        model = dnn.cuda(),
        train_loader = train_loader,
        test_loader = test_loader,
        optimizer = optim.Adam(dnn.parameters(), lr=1e-3),
        writer = writer,
        mc_runs = 1,
        bs = 512,
        device = 'cuda',
        args = args_bnn   
    )

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train a Bayesian Neural Network')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train')
    parser.add_argument('--mc_runs', type=int, default=1, help='Number of Monte Carlo runs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--bs', type=int, default=512, help='Batch size')
    parser.add_argument('--model', type=str, default='lenet', help='Model to train [simple, lenet, vgg7, resnet20]')
    parser.add_argument('--type', type=str, default='uni', help='Type of model [dnn, uni, multi]')
    parser.add_argument('--multi-gpu', action='store_true', help='Use multi-GPU')
    parser.add_argument('--t', type=float, default=1.0, help='Cold Posterior temperature')
    parser.add_argument('--data', type=str, default='cifar', help='Dataset to use [mnist, cifar]')
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