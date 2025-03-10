import torch
import torch.optim as optim
from utils import get_dataset, get_model, test_DNN, train_BNN
from distill import get_conv_layers
import argparse
from termcolor import colored

def main(args):
    
    class opt:
        
        type = 'dnn'
        model = 'resnet20'
        moped = False
        multi_moped = False
        multi_gpu = False
        data = 'cifar'
        bs = 128
        augmentaiton = False
        
    args_dnn = opt()            
    dnn = get_model(args_dnn)
    ckpt = torch.load(args.weight)
    dnn.load_state_dict(ckpt)
    args_dnn.data = args.data
    args_dnn.augmentaiton = args.augmentaiton
    train_loader, test_loader = get_dataset(args_dnn)

    args_bnn = opt()
    args_bnn.type = args.type
    args_bnn.t = args.t
    args_bnn.bs = args.bs
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
        
        mu = dnn_layer.weight.detach().cpu().clone()
        std = torch.where(mu == 0 , torch.ones_like(mu), torch.ones_like(mu))
        bnn_layer.prior_weight_mu = mu
        bnn_layer.prior_weight_sigma = std

    import datetime
    from torch.utils.tensorboard import SummaryWriter
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    log_params = {
        'data': args.data,
        'model': args.model,
        'date': date.split('-')[0],
        'type': args.type,
        'bs': args.bs,
        'lr': args.lr,
        'mc_runs_train': args.mc_runs_train,
        'mc_runs_test': args.mc_runs_test,
        'temp': args.t,
        'epochs': args.epochs,
        'kd': True,
        'martern': True,
        'alpha': 0,
        'moped': False,
        'multi_moped': False,
        'timestamp': date,
        'sparsity': sparsity
        }
    
    params_str = "_".join([f"{key}_{value}" for key, value in log_params.items() if key not in ['data', 'model', 'date', 'type']])
    
    log_path = f"runs/{log_params['data']}/{log_params['model']}/{log_params['date']}/{log_params['type']}/{log_params['sparsity']}/{params_str}_pruned_more_small_other_more_bigger"
        
    writer = SummaryWriter(log_path)

    train_BNN(
        epoch = args.epochs,
        model = bnn.cuda(),
        train_loader = train_loader,
        test_loader = test_loader,
        optimizer = optim.Adam(bnn.parameters(), lr=1e-3),
        writer = writer,
        mc_runs_train = args.mc_runs_train,
        mc_runs_test = args.mc_runs_test,
        bs = args.bs,
        device = 'cuda',
        args = args_bnn   
    )

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train a Bayesian Neural Network')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train')
    parser.add_argument('--mc_runs_train', type=int, default=1, help='Number of Monte Carlo runs for trian')
    parser.add_argument('--mc_runs_test', type=int, default=50, help='Number of Monte Carlo runs for test')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--bs', type=int, default=512, help='Batch size')
    parser.add_argument('--model', type=str, default='resnet20', help='Model to train [simple, lenet, vgg7, resnet20, resnet56, resnet110]')
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
    parser.add_argument('--augmentaiton', type = bool, default = False, help = 'Augmentaiton')
    args = parser.parse_args()
    
    main(args)