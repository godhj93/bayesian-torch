import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import os 
from termcolor import colored
from bayesian_torch.models.dnn_to_bnn import get_kl_loss
    
def train_BNN(epoch, model, train_loader, test_loader, optimizer, writer, args, mc_runs, bs, device, moped=False):

    model.to(device)
    best_loss = torch.inf
    
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
                if not moped:
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
            
            pbar.set_description(colored(f"[Train] Epoch: {e}/{epoch}, Acc: {acc:.5f}, NLL: {np.mean(nll_total):.5f} KL: {np.mean(kl_total):.5f}", 'blue'))
            
        acc, nll, kl = test_BNN(model, test_loader, mc_runs, bs, device, moped)
        print(colored(f"[Test] Acc: {acc:.5f}, NLL: {nll:.5f}, KL: {kl:.5f}", 'yellow'))
        
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

def train_DNN(epoch, model, train_loader, test_loader, optimizer, device, writer):
    
    
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
            pbar.set_description(colored(f"[Train] Epoch: {e}/{epoch}, Acc: {acc_train:.3f}, NLL: {np.mean(nlls):.3f}", 'blue'))
        
        acc_test, nll_test = test_DNN(model, test_loader)
        
        print(colored(f"[Test] Acc: {acc_test:.3f}, NLL: {nll_test:.3f}", 'yellow'))
        
        writer.add_scalar('Train/accuracy', acc_train, e)
        writer.add_scalar('Train/loss/NLL', np.mean(nlls), e)
        writer.add_scalar('Test/accuracy', acc_test, e)
        writer.add_scalar('Test/loss/NLL', np.mean(nlls), e)
        
        if best_loss > nll_test:
            best_loss = nll_test
            torch.save(model.state_dict(), os.path.join(writer.log_dir, 'best_model.pth'))
            print(colored(f"Best model saved at epoch {e}", 'green'))
        
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

