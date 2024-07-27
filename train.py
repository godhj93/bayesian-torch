import torch
from models import LeNet, LeNet_BNN, LeNet_BNN_uni, resnet20_multi
from bayesian_torch.models.bayesian.resnet_variational import resnet20 as resnet20_uni
from bayesian_torch.models.deterministic.resnet import resnet20 as resnet20_deterministic
from utils import train_DNN, train_BNN
import argparse
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import datetime
from termcolor import colored
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os 
    
def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.data == 'mnist':
        # Define the models
        if args.model == 'dnn':
            model = LeNet()
        elif args.model == 'uni':
            model = LeNet_BNN_uni()
        elif args.model == 'multi':
            model = LeNet_BNN()
        else:
            raise ValueError('Model not found')
    
        # MNIST dataset
        train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())
        
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
    
    elif args.data == 'cifar':
        # Define the models
        if args.model == 'dnn':
            model = resnet20_deterministic()
        elif args.model == 'uni':
            model = resnet20_uni()
        elif args.model == 'multi':
            model = resnet20_multi()
        else:
            raise ValueError('Model not found')
        
        # Simple data augmentation
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transoform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        # CIFAR dataset
        train_dataset = datasets.CIFAR10(root='./data/', train=True, transform=transform_train, download=True)
        test_dataset = datasets.CIFAR10(root='./data/', train=False, transform=transoform_test)
        
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
        
        
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = 'runs/{}/{}_bs{}_lr{}_mc{}_temp_{}_ep{}_{}'.format(args.data, args.model, args.bs, args.lr, args.mc_runs, args.t, args.epochs, date)
    writer = SummaryWriter(log_path)
    
    # Multi-GPU
    if torch.cuda.device_count() > 1 and args.multi_gpu:
        import torch.distributed as dist
        from torch.utils.data import DistributedSampler
        
        # DDP 초기화
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl', init_method='env://')
        
        # Set device
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        
        # DistributedSampler 설정
        args.train_sampler = DistributedSampler(train_dataset)
        args.test_sampler = DistributedSampler(test_dataset, shuffle=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, sampler=args.train_sampler, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, sampler=args.test_sampler, num_workers=4, pin_memory=True)
        
        # 모델을 DDP로 래핑
        model = DDP(model.to(device), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        
    else:
        model.to(device)
    
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
    parser.add_argument('--mc_runs', type=int, default=100, help='Number of Monte Carlo runs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--bs', type=int, default=1024, help='Batch size')
    parser.add_argument('--model', type=str, default='multi', help='Model to train [dnn, uni, multi]')
    parser.add_argument('--multi-gpu', action='store_true', help='Use multi-GPU')
    parser.add_argument('--t', type=float, default=1.0, help='Cold Posterior temperature')
    parser.add_argument('--data', type=str, default='mnist', help='Dataset to use [mnist, cifar]')
    parser.add_argument('--train_sampler', type=bool, default=False, help='Do not use this argument')
    args = parser.parse_args()
    
    print(colored(args, 'blue'))
    main(args)