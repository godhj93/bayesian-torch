from models import *
from utils.utils import test_BNN, test_DNN, get_model, get_dataset
import argparse
from termcolor import colored
import torch
from torchvision import datasets, transforms

def main(args):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = get_model(args)
    model.load_state_dict(torch.load(args.weight))
    print(colored(f"Pretrained weight is loaded from {args.weight}", 'green'))
    
    _, test_loader = get_dataset(args)
    
    if args.type == 'dnn':
        
        acc, nll = test_DNN(model, test_loader)
        
        print(colored(f"Acc: {acc:.4f}, NLL: {nll:.4f}", 'blue'))
    
    elif args.type == 'uni':
        
        acc, nll, kld = test_BNN(model, test_loader, bs = 128, device = device, mc_runs = args.mc_runs)
        
        print(colored(f"Acc: {acc:.4f}, NLL: {nll:.4f}, KLD: {kld:.4f}", 'blue'))
        
    else:
        
        raise NotImplementedError("Not implemented yet")
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Test a Pretrained Model')
    parser.add_argument('--type', type=str, help='[dnn, uni, multi]')
    parser.add_argument('--model', type=str, help='Model to train [resnet30, densenet30, vgg7]')
    parser.add_argument('--bs', type=int, default=128, help='Batch size')
    parser.add_argument('--data', type=str, default='cifar', help='Dataset to use [mnist, cifar]')
    parser.add_argument('--mc_runs', type=int, default=30, help='Monte Carlo runs')
    parser.add_argument('--weight', type=str, help='Path to load weights')
    parser.add_argument('--moped', action='store_true', help='Use mode posterior')
    parser.add_argument('--multi_moped', action='store_true', help='Use mode posterior')
    parser.add_argument('--multi_gpu', action='store_true', help='Use mode posterior')
    
    args = parser.parse_args()
    
    print(colored(args, 'green'))
    main(args)