import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import os 
from termcolor import colored

def train_BNN(epoch, model, train_loader, test_loader, optimizer, writer, mc_runs=10, bs=512, device='cuda'):

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
            for mc_run in range(mc_runs):
                output, kl = model(data)
                output_.append(output)
                kl_.append(kl)
            output = torch.mean(torch.stack(output_), dim=0)
            kl = torch.mean(torch.stack(kl_), dim=0)
            
            _, predicted = torch.max(output.data, 1)
            
            nnl = F.cross_entropy(output, target)
            loss = nnl + kl / bs # batch size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            nnls.append(nnl.item())
            kls.append(kl.item() / bs)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            acc = correct / total
            
            pbar.set_description(f"Train Accuracy: {acc:.5f}, NNL: {np.mean(nnls):.5f} KL: {np.mean(kls):.5f} Epoch: {e}")
            
        acc, nnl, kl = test_BNN(model, test_loader, mc_runs, bs)
        print(f"Test accuracy: {acc:.5f}, NNL: {nnl:.5f}, KL: {kl:.5f}")
        
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
            
            torch.save(model.state_dict(), os.path.join(writer.log_dir, 'best_model.pth'))
            print(colored(f"Best model saved at epoch {e}", 'green'))
    
        
def test_BNN(model, test_loader, mc_runs, bs):
    model.eval()
    correct = 0
    total = 0
    nnls = []
    kls = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            
            output_ = []
            kl_ = []
            for mc_run in range(mc_runs):
                output, kl = model(data)
                output_.append(output)
                kl_.append(kl)
            output = torch.mean(torch.stack(output_), dim=0)
            kl = torch.mean(torch.stack(kl_), dim=0)

            _, predicted = torch.max(output.data, 1)
            
            nnl = F.cross_entropy(output, target) 
            
            nnls.append(nnl.item())
            kls.append(kl.item() / bs)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total, np.mean(nnls), np.mean(kls)

def train_DNN(epoch, model, train_loader, test_loader, optimizer, device):
    
    
    model.to(device)    
    model.train()
    
    for e in range(epoch):
        
        pbar = tqdm(enumerate(train_loader))
        for batch_idx, (data, target) in pbar:
            
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            pbar.set_description(f"NNL: {loss.item():.3f} Epoch: {epoch}")
            
        print(f"Test accuracy: {test_DNN(model, test_loader):.3f}")
        
def test_DNN(model, test_loader):

    model.cuda()
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            _, predicted = torch.max(output.data, 1)
        
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total

