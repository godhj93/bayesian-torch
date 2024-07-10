from models import *
from utils import test_BNN, test_DNN
import argparse
from termcolor import colored
import torch
from torchvision import datasets, transforms

def main(args):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define the models
    if args.model == 'dnn':
        model = LeNet()
    elif args.model == 'uni':
        model = LeNet_BNN_uni()
    elif args.model == 'multi':
        model = LeNet_BNN()
    else:
        raise ValueError('Model not found')
        
    model.to(device)
    model.load_state_dict(torch.load(args.weight))
    print(colored(f"Model loaded from {args.weight}", 'blue'))
    test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False)
    
    
    if args.model == 'dnn':
        acc = test_DNN(model = model,
                 test_loader= test_loader)
        nnl, kl = None, None
    else:
        acc, nnl, kl = test_BNN(model = model,
                 test_loader = test_loader,
                 mc_runs = args.mc_runs,
                 bs = args.bs)
    
    print(colored(f"Test accuracy: {acc:.3f}, NNL: {nnl:.3f}, KL: {kl:.3f}", 'green'))
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model', type=str, default='multi', help='[dnn, uni, multi]')
    parser.add_argument('--bs', type=int, default=10000, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--mc_runs', type=int, default=10, help='Monte Carlo runs')
    parser.add_argument('--weight', type=str, help='Path to load weights')
    args = parser.parse_args()
    
    print(colored(args, 'green'))
    main(args)