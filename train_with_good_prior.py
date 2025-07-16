import torch
import torch.optim as optim
from utils.utils import get_dataset, get_model, test_DNN, test_BNN, train_BNN
from distill import get_conv_layers, get_linear_layers
import argparse
from termcolor import colored
import copy
import logging
import datetime
from torch.utils.tensorboard import SummaryWriter

def main(args):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    args_dnn = copy.deepcopy(args)
    args_dnn.type = 'dnn'
    
    dnn = get_model(args = args_dnn, logger = logger)
    ckpt = torch.load(args.weight)
    dnn.load_state_dict(ckpt)
    train_loader, test_loader = get_dataset(args = args_dnn, logger = logger)
    
    # Calculate Sparsity
    total = 0
    zero = 0
    for name, param in dnn.named_parameters():
        if 'weight' in name:
            total += param.numel()
            zero += torch.sum(param == 0).item()
    sparsity = zero/total
    args.sparsity = sparsity  
    
    bnn = get_model(args = args, logger = logger)

    # Optimizer
    if args.optimizer == 'sgd':
        
        optim = torch.optim.SGD(bnn.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov = args.nesterov)
    
    elif args.optimizer == 'adam':
    
        optim = torch.optim.Adam(bnn.parameters(), lr=args.lr)
        args.momentum = None
        args.nesterov = None
        args.weight_decay = None
    
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")        
        
    logging.info(colored(f"Optimizer: {args.optimizer}, Learning rate: {args.lr}, Weight decay: {args.weight_decay}, Momentum: {args.momentum}", 'green'))
    
    # 1. 파라미터를 두 그룹으로 나눕니다.
    # 'log_a_q' 또는 'log_b_q'를 이름에 포함하는 파라미터 (사전 분포 파라미터)
    # prior_params = [param for name, param in bnn.named_parameters() if 'log_a_q' in name or 'log_b_q' in name]

    # # 그 외 모든 파라미터 (기존 가중치 파라미터)
    # base_params = [param for name, param in bnn.named_parameters() if not ('log_a_q' in name or 'log_b_q' in name)]

    # 2. 각 그룹에 다른 learning rate를 적용할 파라미터 리스트를 생성합니다.
    # 계층적 모델이 아닐 경우 prior_params 리스트는 비어있게 됩니다.
    # param_groups = [
    #     {'params': base_params},
    #     {'params': prior_params, 'lr': args.lr_prior} # 사전 분포 파라미터에만 다른 lr 적용
    # ]
    
    # if args.data in ['cifar10', 'cifar100', 'tinyimagenet', 'imagenet'] or 'vit' in args.model:
    #     if not args.prune:
    #         args.lr = 1e-1 # 기본 LR

    #     args.epochs = 90
    #     if args.optimizer == 'adam':
    #         # Adam 옵티마이저에 파라미터 그룹 전달
    #         optim = torch.optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    #         args.momentum = None
    #         args.nesterov = None
    #     else:
    #         # SGD 옵티마이저에 파라미터 그룹 전달
    #         optim = torch.optim.SGD(param_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    #     args.scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[30, 60], gamma=0.1)
    
    # else:
    #     if args.optimizer == 'sgd':
    #         optim = torch.optim.SGD(bnn.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov = args.nesterov)
    
    #     elif args.optimizer == 'adam':
        
    #         optim = torch.optim.Adam(bnn.parameters(), lr=args.lr)
    #         args.momentum = None
    #         args.nesterov = None
    #         args.weight_decay = None
            
    #     else:
    #         raise ValueError(f"Unsupported optimizer: {args.optimizer}")        
        
    #     logging.info(colored(f"Optimizer: {args.optimizer}, Learning rate: {args.lr}, Weight decay: {args.weight_decay}, Momentum: {args.momentum}", 'green'))
    args.scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[100000], gamma=0.1) # We don't want to change the learning rate schedule for now.
        
    log_params = {
        'data': args.data,
        'model': args.model,
        'date': date.split('-')[0],
        'type': args.type,
        'bs': args.bs,
        'opt': args.optimizer,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'nesterov': args.nesterov,
        'lr': args.lr,
        'mc_runs': args.mc_runs,
        'epochs': args.epochs,
        'moped': args.MOPED,
        'timestamp': date,
        'sparsity': sparsity,
        'std': args.std,
        'scale': args.scale,
        'ig_a': args.ig_a,
        'ig_b': args.ig_b,
    }

    params_str = "_".join([f"{key}_{value}" for key, value in log_params.items() if key not in ['data', 'model', 'date', 'type', 'scale']])
    
    log_path = f"runs/{log_params['data']}/{log_params['model']}/{log_params['date']}/{log_params['type']}/{log_params['std']}/{log_params['sparsity']}/{params_str}"
        
    writer = SummaryWriter(log_path)

    file_handler = logging.FileHandler(log_path + '/log.txt')
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # bnn = get_model(args = args, logger = logger)

    acc, loss = test_DNN(dnn, test_loader, device, args)
    logger.info(colored("Testing DNN", 'green'))
    logger.info(colored(f"Sparsity: {sparsity*100:.2f}%, Acc: {acc:.2f}%, Loss: {loss:.4f}", 'green'))
    
    # Set the prior for the convolutional layers
    dnn_conv_layers = get_conv_layers(dnn)
    bnn_conv_layers = get_conv_layers(bnn)

    for dnn_layer, bnn_layer in zip(dnn_conv_layers, bnn_conv_layers):
        
        mu = dnn_layer.weight.detach().cpu().clone()
        
        if args.MOPED: 
            std = torch.where(mu == 0 , torch.ones_like(mu), torch.ones_like(mu))
            prior_variance_hypo_a = torch.ones_like(mu)
            prior_variance_hypo_b = torch.ones_like(mu)
            
        else:
            std = torch.where(mu == 0 , torch.ones_like(mu), torch.ones_like(mu) * args.std)
            
            prior_variance_hypo_a = torch.where(mu == 0, torch.ones_like(mu), torch.ones_like(mu) * args.ig_a)
            prior_variance_hypo_b = torch.where(mu == 0, torch.ones_like(mu), torch.ones_like(mu) * args.ig_b)
            
        bnn_layer.prior_weight_mu = mu
        # Hierarchial models do not use prior_weight_sigma
        bnn_layer.prior_weight_sigma = std
        bnn_layer.prior_variance_hypo_a = prior_variance_hypo_a
        bnn_layer.prior_variance_hypo_b = prior_variance_hypo_b
        # logging.info(colored(f"Setting a Layer: {dnn_layer}", 'yellow'))
        
    # Set the prior for the linear layers
    dnn_linear_layer = get_linear_layers(dnn)
    bnn_linear_layer = get_linear_layers(bnn)
    # print(colored(f"Number of Linear Layers: {len(dnn_linear_layer)}", 'red'))
    # print(colored(f"Number of Linear Layers: {len(bnn_linear_layer)}", 'red'))
    for dnn_layer, bnn_layer in zip(dnn_linear_layer, bnn_linear_layer):
        
        mu = dnn_layer.weight.detach().cpu().clone()
        
        if args.MOPED: 
            std = torch.where(mu == 0 , torch.ones_like(mu), torch.ones_like(mu))
            prior_variance_hypo_a = torch.ones_like(mu)
            prior_variance_hypo_b = torch.ones_like(mu)
            
        else:
            std = torch.where(mu == 0 , torch.ones_like(mu), torch.ones_like(mu) * args.std)
            prior_variance_hypo_a = torch.where(mu == 0, torch.ones_like(mu), torch.ones_like(mu) * args.ig_a)
            prior_variance_hypo_b = torch.where(mu == 0, torch.ones_like(mu), torch.ones_like(mu) * args.ig_b)
            
        bnn_layer.prior_weight_mu = mu
        # Hierarchial models do not use prior_weight_sigma
        bnn_layer.prior_weight_sigma = std
        bnn_layer.prior_hypo_a_weight = prior_variance_hypo_a
        bnn_layer.prior_hypo_b_weight = prior_variance_hypo_b
        
        # logger.info(colored(f"Setting a Layer: {dnn_layer}", 'yellow'))

    # Save the arguments
    with open(f"{log_path}/config.txt", "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
       
    train_BNN(
        epoch = args.epochs,
        model = bnn.cuda(),
        train_loader = train_loader,
        test_loader = test_loader,
        optimizer = optim,
        writer = writer,
        mc_runs = args.mc_runs,
        bs = args.bs,
        device = device,
        args = args,
        logger = logger
    )

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train a Bayesian Neural Network')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train')
    parser.add_argument('--mc_runs', type=int, default=30, help='Number of Monte Carlo runs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lr_prior', type=float, default=1e-3, help='Learning rate for prior parameters')
    parser.add_argument('--ig_a', type=float, default=1.0, help='Inverse Gamma a parameter')
    parser.add_argument('--ig_b', type=float, default=1.0, help='Inverse Gamma b parameter')
    parser.add_argument('--bs', type=int, default=128, help='Batch size')
    parser.add_argument('--model', type=str, help='Model to train [resnet18, resnet20, densenet30, densenet121, mobilenetv2]')
    parser.add_argument('--type', type=str, default='dnn', help='Type of model [dnn, uni, multi]')
    parser.add_argument('--multi-gpu', action='store_true', help='Use multi-GPU')
    parser.add_argument('--t', type=float, default=1.0, help='Cold Posterior temperature')
    parser.add_argument('--data', type=str, help='Dataset to use [cifar10, cifar100, svhn, tinyimagenet]')
    parser.add_argument('--train_sampler', type=bool, default=False, help='Do not use this argument')
    parser.add_argument('--weight', type=str, help='DNN weight path for ')
    parser.add_argument('--moped', action='store_true', help='DO NOT USE')
    parser.add_argument('--MOPED', action='store_true', help='USE MOPED -> N(w_MLE, 1)')
    parser.add_argument('--alpha', type=float, default= 0.0, help = 'Distill Coefficient')
    parser.add_argument('--martern', action='store_true', help='Use Martern Prior')
    parser.add_argument('--multi_moped', action='store_true', help='Use Multi-MOPED')
    parser.add_argument('--prune', action='store_true', help='Use pruning')
    parser.add_argument('--optimizer', type=str, help='Optimizer to use [sgd]')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--nesterov', action='store_true', help='Use Nesterov')
    parser.add_argument('--std', type = float, default = 1e-3, help='Set a std for a good prior')
    parser.add_argument('--scale', type=str, default='BS', help='KLD scale')
    parser.add_argument('--prior_type', type=str, help='Prior type [normal, laplace]')

    args = parser.parse_args()
    
    print(colored(f"Arguments: {args}", 'yellow'))
    main(args)
