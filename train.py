import torch
from utils import train_DNN, train_BNN, get_model, get_dataset, test_DNN
from distill import distill, set_martern_prior
import argparse
from torch.utils.tensorboard import SummaryWriter
import datetime
from termcolor import colored


def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = get_model(args)
    
    train_loader, test_loader = get_dataset(args)

    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = 'runs/{}/{}/{}/bs{}_lr{}_mc{}_temp_{}_ep{}_kd_{}_martern_{}_alpha_{}_moped_{}_{}'.format(args.data, args.model, args.type, args.bs, args.lr, args.mc_runs, args.t, args.epochs, args.distill, args.martern, args.alpha, args.moped, date)
    writer = SummaryWriter(log_path)
    
    if args.distill:
        dnn_model = get_model(args, distill=True)
        dnn_model.load_state_dict(torch.load(args.weight))
        print(colored(f"Distilling from {args.weight}", 'green'))
        print(colored(f"Test accuracy of DNN: {test_DNN(dnn_model, test_loader)}", 'green'))
        model = distill(dnn_model, model, steps = 10000, alpha= args.alpha, device = device, writer = writer)
        
    if args.martern:
        dnn_model = get_model(args, distill=True)
        dnn_model.load_state_dict(torch.load(args.weight))
        print(colored(f"Weight is loaded from {args.weight}", 'green'))
        print(colored(f"Test accuracy of DNN: {test_DNN(dnn_model, test_loader)}", 'green'))
        model = set_martern_prior(dnn_model, model, device = device)
        
    # Optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    args.scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[50, 75], gamma=0.1)
    
    if args.type == 'dnn':
        train_DNN(epoch = args.epochs, 
                  model = model, 
                  train_loader = train_loader, 
                  test_loader = test_loader, 
                  optimizer = optim, 
                  writer = writer,
                  device = device,
                  args = args)
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
    parser.add_argument('--mc_runs', type=int, default=100, help='Number of Monte Carlo runs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--bs', type=int, default=1024, help='Batch size')
    parser.add_argument('--model', type=str, default='simple', help='Model to train [simple, lenet, vgg7, resnet20]')
    parser.add_argument('--type', type=str, default='dnn', help='Type of model [dnn, uni, multi]')
    parser.add_argument('--multi-gpu', action='store_true', help='Use multi-GPU')
    parser.add_argument('--t', type=float, default=1.0, help='Cold Posterior temperature')
    parser.add_argument('--data', type=str, default='mnist', help='Dataset to use [mnist, cifar]')
    parser.add_argument('--train_sampler', type=bool, default=False, help='Do not use this argument')
    parser.add_argument('--distill', action='store_true', help='Use distillation')
    parser.add_argument('--weight', type=str, help = 'DNN weight path for distillation')
    parser.add_argument('--moped', action='store_true', help='Use MOPED')
    parser.add_argument('--alpha', type=float, default= 0.0, help = 'Distill Coefficient')
    parser.add_argument('--martern', action='store_true', help='Use Martern Prior')
    args = parser.parse_args()
    
    print(colored(args, 'blue'))
    main(args)