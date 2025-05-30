from tqdm import tqdm
from termcolor import colored
from torch.distributions import LowRankMultivariateNormal
from torch.optim import Adam
import torch.nn as nn
from bayesian_torch.layers.variational_layers.conv_variational import Conv2dReparameterization, Conv2dReparameterization_Multivariate
from bayesian_torch.layers.variational_layers.linear_variational import LinearReparameterization

import numpy as np
import copy 
from torch.nn import functional as F
import torch
import os 

def Multivariate_MOPED(dnn, bnn, device = 'cuda'):
    
    dnn_conv_layers = get_conv_layers(dnn)
    bnn_conv_layers = get_conv_layers(bnn)
    
    for dnn_layer, bnn_layer in zip(dnn_conv_layers, bnn_conv_layers):
        
        bnn_layer.distill = True
        
        # Set the prior
        bnn_layer.prior_mean = dnn_layer.weight.data.view(-1).detach()
       
        # Set the variational parameters
        bnn_layer.mu_kernel = nn.Parameter(dnn_layer.weight.data.view(-1).detach().clone())
        
    print(colored(f"Set a prior w_MLE from DNN to BNN", 'red'))
    print(colored(f"Set a variational parameter mu from DNN", 'red'))
    print(colored(f"Disabled copying weights from DNN to BNN", 'red'))
        
    return bnn

def distill(dnn, bnn, data_loader, writer, alpha, args, device = 'cuda'):
    
    bnn_conv_layers = get_conv_layers(bnn)
    dnn_conv_layers = get_conv_layers(dnn)
    for layer_idx, (dnn_layer, bnn_layer) in tqdm(enumerate(zip(dnn_conv_layers,bnn_conv_layers)), total = len(bnn_conv_layers)):
        
        ckpt = torch.load(f'./simple_cifar_activations/layer_{layer_idx}/bnn_conv_quarter.pth')
        
        bnn_layer.prior_mean = dnn_layer.weight.data.view(-1).detach()
        
        upper_traiangular_matrix = torch.zeros((bnn_layer.weight_size, bnn_layer.weight_size))
        L_indices = torch.triu_indices(bnn_layer.weight_size, bnn_layer.weight_size, offset=0)
        upper_traiangular_matrix[L_indices[0], L_indices[1]] = ckpt['L_param'].cpu()
        
        bnn_layer.prior_cov_L = upper_traiangular_matrix
        bnn_layer.prior_cov_D = F.softplus(ckpt['mu_kernel'] * 1e-30).cpu() 
        
        print(colored(f"Set the prior for layer {layer_idx}", 'blue'))
        
    return bnn

def distill_old(dnn, bnn, steps, writer, alpha, args, device = 'cuda'):
    
    bnn_good_prior = copy.deepcopy(bnn)
    dnn_conv_layers = get_conv_layers(dnn)
    bnn_conv_layers = get_conv_layers(bnn)
    
    # Check the precomputed prior exists
    if os.path.exists(args.weight.replace('best_model.pth', f"Distilled_BNN_3210.0.pt")): # This condition never satisfies
        
        print(colored(f"Loading distilled BNN from {args.weight.replace('best_model.pth', f'Distilled_BNN.pt')}", 'red'))
        ckpt = torch.load(args.weight.replace('best_model.pth', f"Distilled_BNN.pt"))
        bnn_good_prior.load_state_dict(ckpt)
        return bnn_good_prior
      
    else:
        
        MSE = nn.MSELoss()

        optimizer = Adam(bnn.parameters(), lr = 0.01)
        
        pbar = tqdm(enumerate(range(steps)), total = steps)
        for idx, _ in pbar:
            
            loss = 0
                
            for dnn_layer, bnn_layer in zip(dnn_conv_layers, bnn_conv_layers):
                
                w_dnn = dnn_layer.weight.data.view(-1).to(device)
                
                mu_flat = bnn_layer.mu_kernel.view(-1).to(device)
                
                L, D = bnn_layer.get_covariance_param()
                
                L, D = L.to(device), D.to(device)
                
                w_sample = LowRankMultivariateNormal(mu_flat, L, D).rsample().reshape(w_dnn.size())
                
                loss += MSE(w_sample, w_dnn)  +  alpha / L.norm(p=1)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            pbar.set_description(f"Distillation Loss: {loss.item():.2f}, alpha: {alpha:.2f}")
                
            writer.add_scalar('Distillation Loss', loss.item(), idx)
                
        # Set the prior and variational parameters 
        bnn_good_prior_conv_layers = get_conv_layers(bnn_good_prior)
        
        for bnn_layer, bnn_good_prior_layer in zip(bnn_conv_layers, bnn_good_prior_conv_layers):
            
            bnn_good_prior_layer.distill = True
            # Set the prior
            bnn_good_prior_layer.prior_mean = bnn_layer.mu_kernel.detach().clone().flatten()
            # print(colored(f"Disabled copying of prior mean", 'red'))
            bnn_good_prior_layer.prior_cov_L = bnn_layer.get_covariance_param()[0].detach()
            bnn_good_prior_layer.prior_cov_D = bnn_layer.get_covariance_param()[1].detach()
            
        print(colored(f"Disabled copying weights from DNN to BNN", 'red'))
            
        # Save the model
        path_to_save = args.weight.replace('best_model.pth', f"Distilled_BNN_{alpha}.pt")
        torch.save(bnn_good_prior.state_dict(), path_to_save)
        print(colored(f"Distilled BNN saved at {path_to_save}", 'blue'))
        return bnn_good_prior

def set_martern_prior(dnn, bnn, device = 'cuda'):
    
    bnn_good_prior = copy.deepcopy(bnn)
            
    # Set the prior and variational parameters 
    bnn_good_prior_conv_layers = get_conv_layers(bnn_good_prior)
    # bnn_good_prior_linear_layers = get_linear_layers(bnn_good_prior)
    # dnn_linear_layers = get_linear_layers(dnn)
    
    for bnn_good_prior_layer in tqdm(bnn_good_prior_conv_layers, total = len(bnn_good_prior_conv_layers), ncols=0, desc="Setting Martern prior"):
        
        bnn_good_prior_layer.martern_prior = True
        
    print(colored(f"Martern prior set for Conv layers", 'red'))
    print(colored(f"Disabled copying weights from DNN to BNN", 'red'))
        
    return bnn_good_prior
  
def get_conv_layers(model):
    conv_layers = []
    conv_types = (nn.Conv2d, Conv2dReparameterization, Conv2dReparameterization_Multivariate)
    
    def find_conv_layers(module):
        # Check if the module is an instance of the convolutional layers we're interested in
        if isinstance(module, conv_types):
            conv_layers.append(module)
        # Recursively go through the children of the module
        for child in module.children():
            find_conv_layers(child)

    # Start the recursive search from the given model
    find_conv_layers(model)
    
    return conv_layers

def get_linear_layers(model):
    linear_layers = []
    linear_types = (nn.Linear, LinearReparameterization)
    def find_linear_layers(module):
        if isinstance(module, linear_types):
            linear_layers.append(module)
        for child in module.children():
            find_linear_layers(child)
    
    find_linear_layers(model)
    
    return linear_layers

# Marter Kernel 
# Ref: Bayesian Neural Network Priors Revisited (ICLR, 2022)

def block_diagonal_covariance(conv_layer_weights, sigma=1.0, lamb=1.0):
    """
    Conv layer의 필터별로 블록 대각 공분산 행렬을 계산하는 함수.
    필터 크기가 고정된 경우 공분산 행렬을 재사용하고, 미리 큰 배열을 할당하여 성능을 최적화.
    
    Parameters:
    conv_layer_weights: Convolution layer의 weight (예: [N, C, H, W])
    sigma: 공분산의 스케일링 값 (분산)
    lamb: 거리 감소 파라미터 (length scale)
    
    Returns:
    블록 대각 공분산 행렬
    """
    N = conv_layer_weights.shape[0]  # 필터 개수 (N)
    C = conv_layer_weights.shape[1]  # 채널 개수 (C)
    H = conv_layer_weights.shape[2]  # 필터 높이 (H)
    W = conv_layer_weights.shape[3]  # 필터 너비 (W)
    
    # 고정된 필터 크기에 대한 공분산 행렬을 미리 계산
    fixed_cov_matrix = covariance_matrix_by_filter((H, W), sigma, lamb)
    
    # 각 필터와 채널에 대한 공분산 행렬 크기
    block_size = fixed_cov_matrix.shape[0]
    
    # 최종적으로 필요한 블록 대각 행렬 크기를 계산
    total_blocks = N * C  # 전체 필터와 채널의 조합 개수
    block_diag_size = total_blocks * block_size
    
    # 최종 블록 대각 행렬을 미리 할당 (큰 행렬)
    block_diag_cov = np.zeros((block_diag_size, block_diag_size))
    
    # 블록을 복사할 시작 인덱스
    start_idx = 0
    
    # 블록 대각 행렬을 채워 넣음
    for filter_idx in range(N):
        for channel_idx in range(C):
            end_idx = start_idx + block_size
            block_diag_cov[start_idx:end_idx, start_idx:end_idx] = fixed_cov_matrix
            start_idx = end_idx  # 다음 블록 위치로 이동
    
    return block_diag_cov