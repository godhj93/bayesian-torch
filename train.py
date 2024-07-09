import torch
from models import LeNet, LeNet_BNN, LeNet_BNN_uni
from utils import train_DNN, train_BNN
import argparse
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import datetime
from termcolor import colored

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
        
    # Optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # MNIST dataset
    train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False)
    
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = 'runs/{}_bs{}_lr{}_mc{}_ep{}_{}'.format(args.model, args.bs, args.lr, args.mc_runs, args.epochs, date)
    writer = SummaryWriter(log_path)
    
    if args.model == 'dnn':
        train_DNN(epoch = args.epochs, 
                  model = model, 
                  train_loader = train_loader, 
                  test_loader = test_loader, 
                  optimizer = optim, 
                  writer = writer)
    else:
        train_BNN(epoch = args.epochs, 
                  model = model, 
                  train_loader = train_loader, 
                  test_loader = test_loader, 
                  optimizer = optim, 
                  mc_runs = args.mc_runs, 
                  bs = args.bs, 
                  writer = writer)





if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train a Bayesian Neural Network')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--mc_runs', type=int, default=10, help='Number of Monte Carlo runs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--bs', type=int, default=1024, help='Batch size')
    parser.add_argument('--model', type=str, default='multi', help='Model to train [dnn, uni, multi]')
    
    args = parser.parse_args()
    print(colored(args, 'blue'))
    main(args)