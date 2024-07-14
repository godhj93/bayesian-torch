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
        model.train()
        nnls = []
        kls = []
        correct = 0
        total = 0
        
        pbar = tqdm(enumerate(train_loader))
        for batch_idx, (data, target) in pbar:

            data, target = data.to(device), target.to(device)
            output_ =[]
            kl_ = []
            
            for _ in range(mc_runs):
                if not moped:
                    output, kl = model(data)
                    output_.append(output)
                    kl_.append(kl)
                else:
                    output = model(data)
                    kl = get_kl_loss(model)
                    output_.append(output)
                    kl_.append(kl)
                
            output = torch.mean(torch.stack(output_), dim=0).to(device)
            kl = torch.mean(torch.stack(kl_), dim=0).mean().to(device)
            
            _, predicted = torch.max(output.data, 1)
            
            nnl = F.cross_entropy(output, target)
            
            loss = nnl * (1/args.t) + kl / bs # args.t: Cold posterior temperature
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            nnls.append(nnl.item())
            kls.append(kl.item() / bs)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            acc = correct / total
            
            pbar.set_description(colored(f"[Train] Epoch: {e}/{epoch}, Acc: {acc:.5f}, NNL: {np.mean(nnls):.5f} KL: {np.mean(kls):.5f}", 'blue'))
            
        acc, nnl, kl = test_BNN(model, test_loader, mc_runs, bs, device, moped)
        print(colored(f"[Test] Acc: {acc:.5f}, NNL: {nnl:.5f}, KL: {kl:.5f}", 'yellow'))
        
        # Tensorboard
        writer.add_scalar('Train/accuracy', acc, e)
        writer.add_scalar('Train/loss/NNL', np.mean(nnls), e)
        writer.add_scalar('Train/loss/KL', np.mean(kls), e)
        writer.add_scalar('Train/loss/total', np.mean(nnls) + np.mean(kls), e)

        writer.add_scalar('Test/accuracy', acc, e)
        writer.add_scalar('Test/loss/NNL', nnl, e)
        writer.add_scalar('Test/loss/KL', kl, e)
        writer.add_scalar('Test/loss/total', nnl + kl, e)
        
        # Evaluate the best model by the total loss (test)
        if best_loss > nnl + kl:
            best_loss = nnl + kl
            
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
    nnls = []
    kls = []
    with torch.no_grad():
        
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output_ = []
            kl_ = []
            for _ in range(mc_runs):
                if not moped:
                    output, kl = model(data)
                    output_.append(output)
                    kl_.append(kl)
                else:
                    output = model(data)
                    kl = get_kl_loss(model)
                    output_.append(output)
                    kl_.append(kl)
                    
            output = torch.mean(torch.stack(output_), dim=0).to(device)
            kl = torch.mean(torch.stack(kl_), dim=0).mean().to(device)

            _, predicted = torch.max(output.data, 1)
            
            nnl = F.cross_entropy(output, target) 
            
            nnls.append(nnl.item())
            kls.append(kl.item() / bs)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
    return correct / total, np.mean(nnls), np.mean(kls)

def train_DNN(epoch, model, train_loader, test_loader, optimizer, device, writer):
    
    
    model.to(device)    
    model.train()
    nnls = []
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
            
            nnls.append(loss.item())
            correct += (predicted == target).sum().item()
            total += target.size(0)
            acc_train = correct / total
            pbar.set_description(colored(f"[Train] Epoch: {e}/{epoch}, Acc: {acc_train:.3f}, NNL: {np.mean(nnls):.3f}", 'blue'))
        
        acc_test, nnl_test = test_DNN(model, test_loader)
        
        print(colored(f"[Test] Acc: {acc_test:.3f}, NNL: {nnl_test:.3f}", 'yellow'))
        
        writer.add_scalar('Train/accuracy', acc_train, e)
        writer.add_scalar('Train/loss/NNL', np.mean(nnls), e)
        writer.add_scalar('Test/accuracy', acc_test, e)
        writer.add_scalar('Test/loss/NNL', np.mean(nnls), e)
        
        if best_loss > nnl_test:
            best_loss = nnl_test
            torch.save(model.state_dict(), os.path.join(writer.log_dir, 'best_model.pth'))
            print(colored(f"Best model saved at epoch {e}", 'green'))
        
def test_DNN(model, test_loader):

    model.cuda()
    model.eval()
    correct = 0
    total = 0
    nnls = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            loss = F.cross_entropy(output, target)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            nnls.append(loss.item())
    return correct / total, np.mean(nnls)

