import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import os 
from termcolor import colored
from bayesian_torch.models.dnn_to_bnn import get_kl_loss

# Models
from models import SimpleCNN, SimpleCNN_uni, SimpleCNN_multi, LeNet5, LeNet5_uni, LeNet5_multi, VGG7, VGG7_uni, VGG7_multi, resnet20_multi
from bayesian_torch.models.bayesian.resnet_variational import resnet20 as resnet20_uni
from bayesian_torch.models.deterministic.resnet import resnet20 as resnet20_deterministic
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn
# Dataset
from torchvision import datasets, transforms

# Distirbuted Data Parallel
from torch.nn.parallel import DistributedDataParallel as DDP
import os 
import torch.distributed as dist
from torch.utils.data import DistributedSampler
    
def train_BNN(epoch, model, train_loader, test_loader, optimizer, writer, args, mc_runs, bs, device):

    model.to(device)
    best_loss = torch.inf
    best_nll = torch.inf
    best_acc = 0
    
    for e in range(epoch):
        if args.train_sampler:
            args.train_sampler.set_epoch(e)            
            
        model.train()
        nll_total = []
        kl_total = []
        correct = 0
        total = 0
        
        pbar = tqdm(enumerate(train_loader))
        for batch_idx, (data, target) in pbar:

            data, target = data.to(device), target.to(device)
            outputs =[]
            kls = []
            
            for _ in range(mc_runs):
                if not args.moped:
                    output, kl = model(data)
                    outputs.append(output)
                    kls.append(kl)
                else:
                    output = model(data)
                    kl = get_kl_loss(model)
                    outputs.append(output)
                    kls.append(kl)
                
            output = torch.mean(torch.stack(outputs), dim=0)
            kl_loss = torch.mean(torch.stack(kls), dim=0).mean()
            
            _, predicted = torch.max(output.data, 1)
            
            nll = F.cross_entropy(output, target)
            
            loss = nll * (1/args.t) + kl_loss / bs # args.t: Cold posterior temperature
            # loss = nll
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            nll_total.append(nll.detach().cpu())
            kl_total.append(kl_loss.detach().cpu() / bs)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            acc = correct / total            
            
            pbar.set_description(colored(f"[Train] Epoch: {e+1}/{epoch}, Acc: {acc:.5f}, NLL: {np.mean(nll_total):.5f} KL: {np.mean(kl_total):,}", 'blue'))
            
        acc, nll, kl = test_BNN(model, test_loader, mc_runs, bs, device, args.moped)
        print(colored(f"[Test] Acc: {acc:.5f}, NLL: {nll:.5f}, KL: {kl:,}", 'yellow'))
        
        # Tensorboard
        writer.add_scalar('Train/accuracy', acc, e)
        writer.add_scalar('Train/loss/NLL', np.mean(nll_total), e)
        writer.add_scalar('Train/loss/KL', np.mean(kl_total), e)
        writer.add_scalar('Train/loss/total', np.mean(nll_total) + np.mean(kl_total), e)

        writer.add_scalar('Test/accuracy', acc, e)
        writer.add_scalar('Test/loss/NLL', nll, e)
        writer.add_scalar('Test/loss/KL', kl, e)
        writer.add_scalar('Test/loss/total', nll + kl, e)
        
        # Evaluate the best model by the total loss (test)
        if best_loss > nll + kl:
            best_loss = nll + kl
            
            # Remove Multi-GPU
            # if torch.cuda.device_count() > 1:
            #     torch.save(model.module.state_dict(), os.path.join(writer.log_dir, 'best_model.pth'))
            # else:
            torch.save(model.state_dict(), os.path.join(writer.log_dir, 'best_model.pth'))    
            
            print(colored(f"Best model saved at epoch {e}", 'green'))
            
        if best_nll > nll:
            best_nll = nll
            torch.save(model.state_dict(), os.path.join(writer.log_dir, 'best_nll_model.pth'))
            print(colored(f"Best NLL model saved at epoch {e}", 'green'))
            
        if best_acc < acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(writer.log_dir, 'best_acc_model.pth'))
            print(colored(f"Best ACC model saved at epoch {e}", 'green'))

def test_BNN(model, test_loader, mc_runs, bs, device, moped=False):
    model.eval()
    correct = 0
    total = 0
    nll_total = []
    kl_total = []
    with torch.no_grad():
        
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            outputs = []
            kls = []
            for _ in range(mc_runs):
                if not moped:
                    output, kl = model(data)
                    outputs.append(output)
                    kls.append(kl)
                else:
                    output = model(data)
                    kl = get_kl_loss(model)
                    outputs.append(output)
                    kls.append(kl)
                    
            output = torch.mean(torch.stack(outputs), dim=0).to(device)
            kl = torch.mean(torch.stack(kls), dim=0).mean().to(device)

            _, predicted = torch.max(output.data, 1)
            
            nll = F.cross_entropy(output, target) 
            
            nll_total.append(nll.item())
            kl_total.append(kl.item() / bs)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
    return correct / total, np.mean(nll_total), np.mean(kl_total)

def train_DNN(epoch, model, train_loader, test_loader, optimizer, device, writer, args):
    
    
    model.to(device)    
    model.train()
    nlls = []
    correct = 0
    total = 0
    best_loss = torch.inf
    for e in range(epoch):
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=0)
        for batch_idx, (data, target) in pbar:
            
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            nlls.append(loss.item())
            correct += (predicted == target).sum().item()
            total += target.size(0)
            acc_train = correct / total
            pbar.set_description(colored(f"[Train] Epoch: {e+1}/{epoch}, Acc: {acc_train:.3f}, NLL: {np.mean(nlls):.3f}, LR: {optimizer.param_groups[0]['lr']:.5f}", 'blue'))
        
        acc_test, nll_test = test_DNN(model, test_loader)
        
        print(colored(f"[Test] Acc: {acc_test:.3f}, NLL: {nll_test:.3f}", 'yellow'))
        
        # args.scheduler.step()
        
        writer.add_scalar('Train/accuracy', acc_train, e)
        writer.add_scalar('Train/loss/NLL', np.mean(nlls), e)
        writer.add_scalar('Test/accuracy', acc_test, e)
        writer.add_scalar('Test/loss/NLL', np.mean(nlls), e)
        
        if best_loss > nll_test:
            best_loss = nll_test
            torch.save(model.state_dict(), os.path.join(writer.log_dir, 'best_model.pth'))
            print(colored(f"Best model saved at epoch {e+1}", 'green'))
        
def test_DNN(model, test_loader):

    model.cuda()
    model.eval()
    correct = 0
    total = 0
    nlls = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            loss = F.cross_entropy(output, target)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            nlls.append(loss.item())
    return correct / total, np.mean(nlls)


def get_model(args, distill=False):
    
    if distill:
        args.type = 'dnn'
        print(colored(f"Distillation mode is on", 'red'))
        
    if args.type == 'dnn':
        
        if args.model == 'simple':
            model = SimpleCNN()
            
        elif args.model == 'lenet':
            model = LeNet5()
            
        elif args.model == 'vgg7':
            model = VGG7()
            
        elif args.model == 'resnet20':
            model = resnet20_deterministic()
            
        else:
            raise ValueError('Model not found')
        
    elif args.type == 'uni':
        
        if args.model == 'simple':
            model = SimpleCNN_uni()
            
        elif args.model == 'lenet':
            model = LeNet5_uni()
            
        elif args.model == 'vgg7':
            model = VGG7_uni()
            
        elif args.model == 'resnet20':
            model = resnet20_uni()
            
        else:
            raise ValueError('Model not found')
    
    elif args.type == 'multi':
        
        if args.model == 'simple':
            model = SimpleCNN_multi()
            
        elif args.model == 'lenet':
            model = LeNet5_multi()
            
        elif args.model == 'vgg7':
            model = VGG7_multi()
            
        elif args.model == 'resnet20':
            model = resnet20_multi()
            
        else:
            raise ValueError('Model not found')
    
    if args.moped:
        const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",  # Flipout or Reparameterization
        "moped_enable": True,  # initialize mu/sigma from the dnn weights
        "moped_delta": 0.2,
        }
        
        model.load_state_dict(torch.load(args.weight))
        dnn_to_bnn(model, const_bnn_prior_parameters)
        
        args.type = 'uni'
        print(colored(f"MOPED is on", 'red'))
        
    # Check the number of parameters
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    if torch.cuda.device_count() > 1 and args.multi_gpu:      
        device = 'cuda'  
        # DDP 초기화
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl', init_method='env://')
        
        # Set device
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        
        # 모델을 DDP로 래핑
        model = DDP(model.to(device), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        print(colored(f"Model is wrapped by DDP", 'red'))
    
    if args.distill:
        args.type = 'multi'
        
        
    
    return model

def get_dataset(args):
    
    if args.data == 'mnist':
       
        # Simple data augmentation 
        trasform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])
        transform_test = transforms.Compose([
            # transforms.Resize((32, 32)),
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(root='./data/', train=True, transform=trasform_train, download=True)
        test_dataset = datasets.MNIST(root='./data/', train=False, transform=transform_test)
        
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
    
    elif args.data == 'cifar':
        
        # Simple data augmentation
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
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
    
    if torch.cuda.device_count() > 1 and args.multi_gpu:
        
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
        print(colored(f"Data is wrapped by DistributedSampler", 'red'))
        
    return train_loader, test_loader
    
    