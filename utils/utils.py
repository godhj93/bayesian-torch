import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import os 
from termcolor import colored
from bayesian_torch.models.dnn_to_bnn import get_kl_loss
import copy
# MODEL
from utils.models.resnet_multi import resnet20_multi
from utils.models.densenet_dnn import densenet_bc_30
from utils.models.densenet_uni import densenet_bc_30_uni
from utils.models.mobilenetv2_dnn import MobileNetV2_dnn
from utils.models.mobilenetv2_uni import MobileNetV2_uni
from utils.models.vgg_dnn import VGG7
from utils.models.vgg_uni import VGG7_uni
from utils.models.lenet_dnn import LeNet5_dnn
from utils.models.lenet_uni import LeNet5_uni
# from utils.models.resnet18_dnn import ResNet18_dnn
from bayesian_torch.models.deterministic.resnet_large import resnet18 as ResNet18_dnn
from bayesian_torch.models.bayesian.resnet_variational_large import resnet18 as ResNet18_uni
from utils.models.vit_tiny_dnn import ViT_Tiny_dnn
from utils.models.vit_tiny_uni import ViT_Tiny_uni
from utils.models.mlp_dnn import MLP_dnn
from utils.models.mlp_uni import MLP_uni
from utils.models.wideresnet_dnn import *
from utils.models.basic_rnn import RNN_dnn
from utils.models.basic_rnn_uni import RNN_uni
from bayesian_torch.models.bayesian.resnet_variational import resnet20 as resnet20_uni
from bayesian_torch.models.deterministic.resnet import resnet20 as resnet20_deterministic
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn
from bayesian_torch.layers.variational_layers.conv_variational import Conv2dReparameterization, Conv2dReparameterization_Multivariate
from bayesian_torch.layers.variational_layers.linear_variational import LinearReparameterization
from bayesian_torch.models.bayesian.resnet_hvariational import resnet20 as resnet20_hvariational
from bayesian_torch.layers.variational_layers.hiearchial_variational_layers import Conv2dReparameterizationHierarchical, LinearReparameterizationHierarchical
# Dataset
from torchvision import datasets, transforms

# Distirbuted Data Parallel
from torch.nn.parallel import DistributedDataParallel as DDP
import os 
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from torchvision.datasets import ImageFolder

import torch.nn.utils.prune as prune

def train_BNN(epoch, model, train_loader, test_loader, optimizer, writer, args, mc_runs, bs, device, logger):

    model.to(device)
    best_loss = torch.inf
    best_nll = torch.inf
    best_acc = 0
    
    early_stopping = EarlyStopping(patience=100, min_delta=0.0)
    
    for e in range(epoch):
        if args.train_sampler:
            args.train_sampler.set_epoch(e)            
            
        model.train()
        nll_total = []
        kl_total = []
        correct = 0
        total = 0
        
        pbar = tqdm(enumerate(train_loader))
        N = len(train_loader.dataset)
        
        if args.scale == 'N':
            scaling = N
        else:
            scaling = bs
            
        for batch_idx, (data, target) in pbar:
    
            data, target = data.to(device), target.to(device)
            outputs =[]
            kls = []
            
            for _ in range(1): # For training, mc_runs is set to 1
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
            
            loss = nll * (1/args.t) + kl_loss / scaling #N # args.t: Cold posterior temperature
            # loss = nll
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            nll_total.append(nll.detach().cpu())
            kl_total.append(kl_loss.detach().cpu() / scaling)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            acc = correct / total            
            
            pbar.set_description(colored(f"[Train] Epoch: {e+1}/{epoch}, Acc: {acc:.5f}, NLL: {np.mean(nll_total):.5f} KL: {np.mean(kl_total):,}, KL scaling: {scaling}", 'blue'))
            
        args.scheduler.step()
        
        acc_test, nll, kl = test_BNN(model = model, test_loader = test_loader, bs = bs, mc_runs = mc_runs, device = device, args = args)
        logger.info(f"[Test] Acc: {acc_test:.5f}, NLL: {nll:.5f}, KL: {kl:,}, KL scaling: {scaling}")
        
        # args.scheduler.step()
        # print(colored(f"Learning rate: {optimizer.param_groups[0]['lr']}", 'red'))
        # Tensorboard
        writer.add_scalar('Train/accuracy', acc, e)
        writer.add_scalar('Train/loss/NLL', np.mean(nll_total), e)
        writer.add_scalar('Train/loss/KL', np.mean(kl_total), e)
        writer.add_scalar('Train/loss/total', np.mean(nll_total) + np.mean(kl_total), e)
        writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], e)
        writer.add_scalar('Test/accuracy', acc_test, e)
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
            
            logger.info(f"Best model saved at epoch {e}")
            
        if best_nll > nll:
            best_nll = nll
            torch.save(model.state_dict(), os.path.join(writer.log_dir, 'best_nll_model.pth'))
            logger.info(f"Best NLL model saved at epoch {e}")
            
        if best_acc < acc_test:
            best_acc = acc_test
            torch.save(model.state_dict(), os.path.join(writer.log_dir, 'best_acc_model.pth'))
            logger.info(f"Best ACC model saved at epoch {e}")
            
        early_stopping(val_loss=nll, model=model)
        
        if early_stopping.early_stop:
            logger.info(f"Early stopping at epoch {e+1}")
            # best_model_weight = early_stopping.best_model_state
            # save_path = os.path.join(writer.log_dir, f'best_model.pth')
            # torch.save(best_model_weight, save_path)
            
            # Logging the best model
            # model.load_state_dict(best_model_weight).to(device).eval()
            # best_acc, best_loss, best_kld = test_BNN(model = model, test_loader = test_loader, bs = bs, mc_runs = mc_runs, device = device, args = args)
            # logger.info(f"Best NLL model loaded: {best_acc:.5f}, {best_loss:.5f}, {best_kld:.5f}")
            
            return False
        
    torch.save(model.state_dict(), os.path.join(writer.log_dir, 'last_model.pth'))
    logger.info(f"Last model saved")
    
    # Logging the best model
    # model.load_state_dict(best_model_weight).to(device).eval()
    # best_acc, best_loss, best_kld = test_BNN(model = model, test_loader = test_loader, bs = bs, mc_runs = mc_runs, device = device, args = args)
    # logger.info(f"Best NLL model loaded: {best_acc:.5f}, {best_loss:.5f}, {best_kld:.5f}")

def test_BNN(model, test_loader, bs, device, args, moped=False, mc_runs = 30):
    
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    nll_total = []
    kl_total = []
    mc_runs = 30
    
    N = len(test_loader.dataset)
    if args.scale == 'N':
        scaling = N
    else:
        scaling = bs
        
    with torch.no_grad():
        
        for data, target in tqdm(test_loader, desc=f'Testing [MC_runs={mc_runs}]'):
            data, target = data.to(device), target.to(device)
            
            outputs = []
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
                    
            output = torch.mean(torch.stack(outputs), dim=0).to(device)
            kl = torch.mean(torch.stack(kls), dim=0).mean().to(device)

            _, predicted = torch.max(output.data, 1)
            
            nll = F.cross_entropy(output, target) 
            
            nll_total.append(nll.item())
            kl_total.append(kl.item() / scaling)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
    return correct / total, np.mean(nll_total), np.mean(kl_total)

def train_DNN(epoch, model, train_loader, test_loader, optimizer, device, writer, args, logger):
    
    model.to(device)    
    model.train()
    nlls = []
    correct = 0
    total = 0
    
    best_loss = torch.inf
    best_acc = 0
    best_model_found = False
    
    early_stopping = EarlyStopping(patience=100, min_delta=0.0)
    
    for e in range(epoch):
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=0)
        model.train()
        # for batch_idx, (data, target) in pbar:
        for batch_idx, batch_data in pbar:
            
            if args.model == 'basic_rnn':
                data, target = batch_data["input_ids"].to(device), batch_data["label"].to(device)
            else:
                data, target = batch_data[0], batch_data[1]
                data, target = data.to(device).squeeze(1), target.to(device)
                
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

            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], batch_idx + e * len(train_loader))
            
        args.scheduler.step()
        
        acc_test, nll_test = test_DNN(model, test_loader, device, args)
        logger.info(f"[Test] Acc: {acc_test:.3f}, NLL: {nll_test:.3f}")
        
        if args.prune:
            writer.add_scalar('Train/accuracy', acc_train, e + 1 + args.total_epoch)
            writer.add_scalar('Train/loss/NLL', np.mean(nlls), e + 1 + args.total_epoch)
            writer.add_scalar('Test/accuracy', acc_test, e + 1 + args.total_epoch)
            writer.add_scalar('Test/loss/NLL', np.mean(nll_test), e + 1 + args.total_epoch)

        else:
            writer.add_scalar('Train/accuracy', acc_train, e)
            writer.add_scalar('Train/loss/NLL', np.mean(nlls), e)
            writer.add_scalar('Test/accuracy', acc_test, e)
            writer.add_scalar('Test/loss/NLL', np.mean(nll_test), e)
        
        
        
        if best_loss > nll_test:
            best_loss = nll_test
            torch.save(model.state_dict(), os.path.join(writer.log_dir, 'best_model.pth'))
            logger.info(f"Best model saved at epoch {e+1}")
        
        if args.prune:

            logger.info(f"Original best NLL: {args.best_nll:.4f}, Current NLL: {nll_test:.4f}")
            logger.info(f"Original best ACC: {args.best_acc:.4f}, Current ACC: {acc_test:.4f}")
            
            if best_acc <= acc_test:
                best_acc = acc_test
            
            if nll_test <= best_loss:
                best_loss = nll_test
            
            # if best_acc >= args.best_acc and acc_test >= best_acc: 
            if best_loss <= args.best_nll and nll_test <= best_loss:
                
                logger.info(f"Early stopping at epoch {e+1}")
                best_model_weight = model.state_dict()
                save_path = os.path.join(writer.log_dir, f'pruned_model_iter_{args.prune_iter}.pth')
                save_pruned_model(model, save_path)
                
                best_model_found = True
                return False
            elif e == epoch - 1 and not best_model_found:
                logger.info(f"Stop to fine-tune at {e+1} epoch since the NLL does not recovered")
                return True
            
        early_stopping(val_loss=nll_test, model=model)
            
        if early_stopping.early_stop:
            logger.info(f"Early stopping at epoch {e+1}")
            best_model_weight = early_stopping.best_model_state
            save_path = os.path.join(writer.log_dir, f'best_model.pth')
            torch.save(best_model_weight, save_path)
            return False

    torch.save(model.state_dict(), os.path.join(writer.log_dir, 'last_model.pth'))
    logger.info(f"Last model saved")
    
    # model.load_state_dict(best_model_weight, strict=False)
    # print(colored(f"Best model returned", 'green'))
    
    
def test_DNN(model, test_loader, device, args):

    model.cuda()
    model.eval()
    correct = 0
    total = 0
    nlls = []
    with torch.no_grad():
        
        # for data, target in test_loader:
        for batch_data in test_loader:
            
            if args.model == 'basic_rnn':
                data, target = batch_data["input_ids"].to(device), batch_data["label"].to(device)
                
            else:
                data, target = batch_data[0], batch_data[1]
                data, target = data.to(device).squeeze(1), target.to(device)
                
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            loss = F.cross_entropy(output, target)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            nlls.append(loss.item())
            
    return correct / total, np.mean(nlls)


def save_pruned_model(model, save_path):
    """
    Save the pruned model with masks removed.
    Args:
        model (torch.nn.Module): The model to save.
        save_path (str): Path to save the model.
    """
    # Remove pruning masks before saving
    for name, module in model.named_modules():
        if hasattr(module, "weight") and hasattr(module, "weight_orig"):
            prune.remove(module, "weight")
    
    # Save the model's state_dict
    torch.save(model.state_dict(), save_path)
    # print(f"Pruned model saved (masks removed) at: {save_path}")

def get_model(args, logger, distill=False):
    
    if distill:
        args.type = 'dnn'
        print(colored(f"Getting DNN model", 'red'))
        
    if args.type == 'dnn':
            
        if args.model == 'resnet20' or args.model == 'resnet20_h':
            model = resnet20_deterministic()

        elif args.model == 'resnet18':
            if args.data == 'imagenet':
                model = ResNet18_dnn(num_classes=1000, pretrained=True)
            # else:
                # model = ResNet18_dnn(num_classes=100, pretrained=False)
            
        elif args.model == 'wrn10-1':
            model = wrn10_1()
            
        elif args.model == 'wrn10-2':
            model = wrn10_2()
            
        elif args.model == 'wrn16-4':
            model = wrn16_4()
            
        elif args.model == 'wrn16-2':
            model = wrn16_2()
        
        elif args.model == 'wrn22-4':
            model = wrn22_4()
            
        elif args.model == 'wrn22-2':
            model = wrn22_2()
            
        elif args.model == 'densenet30':
            model = densenet_bc_30()
            
        elif args.model == 'mobilenetv2':
            model = MobileNetV2_dnn(num_classes=10, width_mult=1.0)
        
        elif args.model == 'vit-tiny-layernorm':
            model = ViT_Tiny_dnn(num_classes=10)

        elif args.model == 'vit-tiny-dyt':
            model = ViT_Tiny_dnn(num_classes=10, norm='dyt')
        
        elif args.model == 'vit-tiny-rms':
            model = ViT_Tiny_dnn(num_classes=10, norm='rms')

        elif args.model == 'vit-tiny-layernorm-relu':
            model = ViT_Tiny_dnn(num_classes=10, norm='layernorm', act='relu')
            print(model)
        elif args.model == 'mlp':
            model = MLP_dnn(input_size=28*28, hidden_size=100, output_size=10)
            
        elif args.model == 'basic_rnn':
            model = RNN_dnn(vocab_size=30522) # vocab_size from ag_news dataset
            
        else:
            raise ValueError('Model not found')
        
    elif args.type == 'uni':
            
        if args.model == 'resnet20':
            model = resnet20_uni(args.prior_type)
            
        elif args.model == 'resnet20_h':
            model = resnet20_hvariational()
            
        elif args.model == 'densenet30':
            model = densenet_bc_30_uni(prior_type=args.prior_type)

        elif args.model == 'resnet18':
            model = ResNet18_uni(pretrained=False)
            
        elif args.model == 'mobilenetv2':
            model = MobileNetV2_uni()
            
        elif args.model == 'vit-tiny-layernorm': 
            model = ViT_Tiny_uni(num_classes=10)

        elif args.model == 'vit-tiny-dyt':
            model = ViT_Tiny_uni(num_classes=10, norm='dyt')

        elif args.model == 'vit-tiny-rms':
            model = ViT_Tiny_uni(num_classes=10, norm='rms')
            
        elif args.model == 'mlp':
            model = MLP_uni(input_size=28*28, hidden_size=100, output_size=10)
        
        elif args.model == 'basic_rnn':
            model = RNN_uni(vocab_size=30522)
            
        else:
            raise ValueError('Model not found')
        
    else:
        raise NotImplementedError("Model Parsing: Not implemented yet")
    
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
        logger.info(colored(f"Model is wrapped by DDP", 'red'))
    
   
        
    if args.data == 'mnist':
        pass
        
    elif args.data =='cifar10':

        logger.info(colored(f"{args.model} will be used.", 'red'))
        logger.info(colored(f"{args.type} Conv1 input channel is changed to 3", 'red'))
    
    elif args.data == 'cifar100':
        
        if args.model == 'mobilenetv2':
            
            if args.type == 'dnn':
                model.classifier = torch.nn.Linear(1280, 100)
                
            elif args.type =='uni':
                raise NotImplementedError("Not implemented yet")
        
        elif args.model == 'resnet18':
            
            if args.type =='dnn':
                model.base_model.fc = torch.nn.Linear(512, 100)
                
        elif args.model == 'resnet20':
            
            if args.type == 'dnn':
                model.linear = torch.nn.Linear(64, 100)
        
            if args.type == 'uni':
                model.linear = LinearReparameterization(64, 100)
                
        elif args.model == 'vit-tiny':
              model.head = torch.nn.Linear(model.head.in_features, 100, bias=True)

        elif args.model == 'densenet30':
            
            if args.type == 'dnn':
                model.classifier = torch.nn.Linear(model.classifier.in_features, 100, bias = True)
            if args.type == 'uni':
                model.classifier = LinearReparameterization(model.classifier.in_features, 100, bias = True, prior_type=args.prior_type)
        else:
            
            raise NotImplementedError("Not implemented yet")
                
    elif args.data == 'tinyimagenet':
        
        if args.model == 'mobilenetv2':
            
            if args.type == 'dnn':
                model.classifier = torch.nn.Linear(1280, 200)
        
        elif args.model == 'resnet18':
            
            if args.type == 'dnn':
                model.base_model.fc = torch.nn.Linear(512, 200)
        
        elif args.model == 'resnet20':
            if args.type == 'dnn':
                model.linear = torch.nn.Linear(64, 200)
            elif args.type == 'uni':
                model.linear = LinearReparameterization(64, 200)
        
        elif args.model == 'resnet20_h':
            if args.type == 'dnn':
                model.linear = torch.nn.Linear(64, 200)
            elif args.type == 'uni':
                model.linear = LinearReparameterizationHierarchical(64, 200)
                
        elif args.model == 'densenet30':
            if args.type == 'dnn':
                model.classifier = torch.nn.Linear(model.classifier.in_features, 200, bias = True)
            elif args.type == 'uni':
                model.classifier = LinearReparameterization(model.classifier.in_features, 200, bias = True, prior_type=args.prior_type)
                
        else:
            raise NotImplementedError("Not implemented yet")
    
    elif args.data == 'imagenet':
        
        if args.model == 'resnet18':
            # model.base_model.fc = torch.nn.Linear(512, 1000)
            print(model)
            
    
    elif args.data == 'svhn':
        pass

    elif args.data == 'ag_news':
        pass
    
    else:
        raise NotImplementedError("Data Parsing: Not implemented yet")
    
    return model

def get_dataset(args, logger):
    
    if args.data == 'mnist':
        
        logger.info(colored(f"MNIST dataset is loaded", 'green'))

        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(root='./data/', train=True, transform=transform_train, download=True)
        test_dataset = datasets.MNIST(root='./data/', train=False, transform=transform_test)
        
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
        
    elif args.data == 'cifar10':
        
        logger.info(colored(f"CIFAR-10 dataset is loaded", 'green'))
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        train_dataset = datasets.CIFAR10(root='./data/', train=True, transform=transform_train, download=True)
        test_dataset = datasets.CIFAR10(root='./data/', train=False, transform=transform_test)
        
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
    
    elif args.data == 'cifar100':
        
        img_size = 32
        logger.info(colored(f"CIFAR-100 dataset is loaded, Size: {img_size}x{img_size}", 'green'))
        
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        train_dataset = datasets.CIFAR100(root='./data/', train=True, transform=transform_train, download=True)
        test_dataset = datasets.CIFAR100(root='./data/', train=False, transform=transform_test)
        
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
        
    elif args.data == 'tinyimagenet':
        
        img_size = 64
        logger.info(colored(f"Tiny ImageNet dataset is loaded, Size: {img_size}x{img_size}", 'green'))
        
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))

        ])
        
        transform_test = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),

        ])
        
        train_dataset = ImageFolder(root='data/tiny-imagenet-200/train/', transform = transform_train)
        test_dataset = ImageFolder(root='data/tiny-imagenet-200/val/', transform = transform_test)
        
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
         
    elif args.data == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        traindir = os.path.join('imagenet', 'train')
        valdir = os.path.join('imagenet', 'val')
        
        transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        
        transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        
        train_dataset = datasets.ImageFolder(
            traindir,
            transform=transform_train)

        val_dataset = datasets.ImageFolder(
            valdir,
            transform=transform_test
            )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.bs, shuffle=True,
            num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.bs, shuffle=False,
            num_workers=4, pin_memory=True)
        
    elif args.data == 'svhn':
        
        logger.info(colored(f"SVHN dataset is loaded", 'green'))
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        
        train_dataset = datasets.SVHN(root='./data/', split='train', transform=transform_train, download=True)
        test_dataset = datasets.SVHN(root='./data/', split='test', transform=transform_test, download=True)
        
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
        
    elif args.data == 'ag_news':
        from transformers import AutoTokenizer
        from datasets import load_dataset
        MAX_LENGTH = 50
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        dataset = load_dataset("ag_news")

        def tokenize(example):
            return tokenizer(example["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

        dataset = dataset.map(tokenize)
        dataset.set_format(type="torch", columns=["input_ids", "label"])
        transform_train = None
        transform_test = None
        
        train_loader = torch.utils.data.DataLoader(dataset["train"], batch_size=args.bs, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset["test"], batch_size=args.bs)
        
    else:
        raise ValueError('Dataset not found')
    
    logger.info(colored(f"Train Transforms:"))
    logger.info(colored(f"{transform_train}", 'green'))
    
    logger.info(colored(f"Test Transforms:"))
    logger.info(colored(f"{transform_test}", 'green'))
        
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
    
class EarlyStopping:
    
    """
    Validation Loss가 개선되지 않으면 일정 patience 만큼 기다렸다가 학습을 조기 종료합니다.
    """
    def __init__(self, patience=5, min_delta=0.0):
        """
        Args:
            patience (int): 성능이 개선되지 않는 Epoch가 patience를 초과하면 학습 중단
            min_delta (float): Loss가 이전 최저값 대비 어느 정도(=min_delta) 이하로 내려가야 '개선'으로 판단
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model_state = None  # 최적 모델 가중치 저장

    def __call__(self, val_loss, model):
        """
        val_loss (float): 현재 Epoch에서 측정한 Validation Loss
        model (nn.Module): 학습 중인 모델 객체
        """
        if val_loss < self.best_loss - self.min_delta:
            # 성능이 개선된 경우
            self.best_loss = val_loss
            self.counter = 0
            # 모델의 가중치(파라미터) 복사해 저장
            self.best_model_state = copy.deepcopy(model.state_dict())
        else:
            # 성능 개선 없음
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
