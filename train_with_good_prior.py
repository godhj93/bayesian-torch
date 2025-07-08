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
    
    if args.data == 'cifar100' or args.data == 'tinyimagenet':
        
        args.epochs = 300
        args.lr = 1e-1
        optim = torch.optim.SGD(bnn.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov = args.nesterov)
        args.scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[100, 200], gamma=0.1)
        print(colored(f"Scheduler: MultiStepLR, Milestones: [100, 200], Gamma: 0.1", 'red'))
        print(colored(f"Epochs: {args.epochs}", 'red'))
        print(colored(f"Optim: SGD, Learning rate: {args.lr}, Weight decay: {args.weight_decay}, Momentum: {args.momentum}", 'red'))
        
    elif args.data == 'imagenet':
        # Multi Step Learning rate Schedule
        args.lr = 1e-1
        args.epochs = 90
        if args.optimizer == 'adam':
            optim = torch.optim.Adam(bnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            args.momentum = None
            args.nesterov = None
        else:
            optim = torch.optim.SGD(bnn.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov = args.nesterov)
        args.scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[30, 60], gamma=0.1)
        
    else:
        args.scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[1000000000], gamma=1.0) # No scheduler
        print(colored(f"No LR Scheduler: MultiStepLR, Milestones: [1000000000], Gamma: 1.0", 'red'))
        
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
        'name': args.name
    }

    params_str = "_".join([f"{key}_{value}" for key, value in log_params.items() if key not in ['data', 'model', 'date', 'type', 'name']])
    
    log_path = f"runs/{log_params['data']}/{log_params['model']}/{log_params['date']}/{log_params['type']}/{log_params['sparsity']}/{params_str}"
        
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
        else:
            std = torch.where(mu == 0 , torch.ones_like(mu), torch.ones_like(mu) * args.std)
            
        bnn_layer.prior_weight_mu = mu
        bnn_layer.prior_weight_sigma = std

        logging.info(colored(f"Setting a Layer: {dnn_layer}", 'yellow'))
        
    # Set the prior for the linear layers
    dnn_linear_layer = get_linear_layers(dnn)
    bnn_linear_layer = get_linear_layers(bnn)
    print(colored(f"Number of Linear Layers: {len(dnn_linear_layer)}", 'red'))
    print(colored(f"Number of Linear Layers: {len(bnn_linear_layer)}", 'red'))
    for dnn_layer, bnn_layer in zip(dnn_linear_layer, bnn_linear_layer):
        
        mu = dnn_layer.weight.detach().cpu().clone()
        
        if args.MOPED: 
            std = torch.where(mu == 0 , torch.ones_like(mu), torch.ones_like(mu))
        else:
            std = torch.where(mu == 0 , torch.ones_like(mu), torch.ones_like(mu) * args.std)
            
        bnn_layer.prior_weight_mu = mu
        bnn_layer.prior_weight_sigma = std
        
        logger.info(colored(f"Setting a Layer: {dnn_layer}", 'yellow'))
        
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
    parser.add_argument('--name', type=str, default='', help='Name of the run for logging')
    args = parser.parse_args()
    
    main(args)
