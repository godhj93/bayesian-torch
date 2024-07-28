import torch
from utils import train_DNN, train_BNN, get_model, get_dataset
import argparse
from torch.utils.tensorboard import SummaryWriter
import datetime
from termcolor import colored


def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = get_model(args)
    
    train_loader, test_loader = get_dataset(args)

    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = 'runs/{}/{}/{}_bs{}_lr{}_mc{}_temp_{}_ep{}_{}'.format(args.data, args.model, args.type, args.bs, args.lr, args.mc_runs, args.t, args.epochs, date)
    writer = SummaryWriter(log_path)
    
    # Optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    if args.model == 'dnn':
        train_DNN(epoch = args.epochs, 
                  model = model, 
                  train_loader = train_loader, 
                  test_loader = test_loader, 
                  optimizer = optim, 
                  writer = writer,
                  device = device)
    else:
        train_BNN(epoch = args.epochs, 
                  model = model, 
                  train_loader = train_loader, 
                  test_loader = test_loader, 
                  optimizer = optim, 
                  mc_runs = args.mc_runs, 
                  bs = args.bs, 
                  writer = writer,
                  device = device,
                  args = args)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train a Bayesian Neural Network')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--mc_runs', type=int, default=30, help='Number of Monte Carlo runs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--bs', type=int, default=1024, help='Batch size')
    parser.add_argument('--model', type=str, default='simple', help='Model to train [simple, lenet, vgg7, resnet20]')
    parser.add_argument('--type', type=str, default='dnn', help='Type of model [dnn, uni, multi]')
    parser.add_argument('--multi-gpu', action='store_true', help='Use multi-GPU')
    parser.add_argument('--t', type=float, default=1.0, help='Cold Posterior temperature')
    parser.add_argument('--data', type=str, default='mnist', help='Dataset to use [mnist, cifar]')
    parser.add_argument('--train_sampler', type=bool, default=False, help='Do not use this argument')
    args = parser.parse_args()
    
    print(colored(args, 'blue'))
    main(args)