from tqdm import tqdm
from termcolor import colored
from torch.distributions import LowRankMultivariateNormal
from torch.optim import Adam
import torch.nn as nn
from bayesian_torch.layers.variational_layers.conv_variational import Conv2dReparameterization, Conv2dReparameterization_Multivariate
import numpy as np
import copy 
from torch.nn import functional as F
import torch
import os 
def distill(dnn, bnn, steps, writer, alpha, args, device = 'cuda'):
    
    bnn_good_prior = copy.deepcopy(bnn)
    dnn_conv_layers = get_conv_layers(dnn)
    bnn_conv_layers = get_conv_layers(bnn)
    
    # Check the precomputed prior exists
    if os.path.exists(args.weight.replace('best_model.pth', f"Distilled_BNN.pt")):
        
        print(colored(f"Loading distilled BNN from {args.weight.replace('best_model.pth', f'Distilled_BNN.pt')}", 'red'))
        ckpt = torch.load(args.weight.replace('best_model.pth', f"Distilled_BNN.pt"))
        bnn_good_prior.load_state_dict(ckpt)
        return bnn_good_prior
      
    else:
        MSE = nn.MSELoss()

        optimizer = Adam(bnn.parameters(), lr = 0.001)
        
        pbar = tqdm(enumerate(range(steps)), total = steps)
        for idx, _ in pbar:
            
            loss = 0
                
            for dnn_layer, bnn_layer in zip(dnn_conv_layers, bnn_conv_layers):
                # print(f"Distilling from {dnn_layer} to {bnn_layer}")
                w_dnn = dnn_layer.weight.data.view(-1).to(device)
                
                mu_flat = bnn_layer.mu_kernel.view(-1).to(device)
                
                L, D = bnn_layer.get_covariance_param()
                
                L, D = L.to(device), D.to(device)
                
                # Check var is on the same device 
                w_sample = LowRankMultivariateNormal(mu_flat, L, D).rsample().reshape(w_dnn.size())
                
                
                loss += MSE(w_sample, w_dnn)  +  alpha / L.norm(p=1)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            pbar.set_description(f"Distillation Loss: {loss.item():.2f}, alpha: {alpha:.2f}")
                
            writer.add_scalar('Distillation Loss', loss.item(), idx)
                
        # Set the prior and variational parameters 
        bnn_good_prior_conv_layers = get_conv_layers(bnn_good_prior)
        bnn_good_prior_linear_layers = get_linear_layers(bnn_good_prior)
        dnn_linear_layers = get_linear_layers(dnn)
        
        for bnn_layer, bnn_good_prior_layer in zip(bnn_conv_layers, bnn_good_prior_conv_layers):
            
            bnn_good_prior_layer.distill = True
            # Set the prior
            bnn_good_prior_layer.prior_mean = bnn_layer.mu_kernel.detach().clone().flatten()
            # print(colored(f"Disabled copying of prior mean", 'red'))
            bnn_good_prior_layer.prior_cov_L = bnn_layer.get_covariance_param()[0].detach()
            bnn_good_prior_layer.prior_cov_D = bnn_layer.get_covariance_param()[1].detach()
            
            # Set the variational parameters
            # bnn_good_prior_layer.mu_kernel = bnn_layer.mu_kernel
            # bnn_good_prior_layer.L_param.data = bnn_layer.get_covariance_param()[0].detach().clone()
            # bnn_good_prior_layer.B.data = bnn_layer.get_covariance_param()[1].detach().clone()
            
        print(colored(f"Disabled copying weights from DNN to BNN", 'red'))
        # for dnn_layer, bnn_good_prior_layer in zip(dnn_linear_layers, bnn_good_prior_linear_layers):
            
        #     # Set the prior
        #     bnn_good_prior_layer.weight.data = dnn_layer.weight.data.clone()
        #     print(colored(f"Linear weight copied from DNN to BNN", 'red'))
            
        # Save the model
        path_to_save = args.weight.replace('best_model.pth', f"Distilled_BNN.pt")
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
        
        # Set the prior
        # bnn_good_prior_layer.prior_mean = # Set mean to zero
        
        # covariance_matrix = block_diagonal_covariance(bnn_good_prior_layer.mu_kernel)
        # Cholesky decomposition for efficient training
        bnn_good_prior_layer.martern_prior = True
        # bnn_good_prior_layer.block_diagonal_matrix = covariance_matrix_by_filter((bnn_good_prior_layer.mu_kernel.shape[-2:]), sigma=1.0, lamb=1.0)
        
    print(colored(f"Martern prior set for Conv layers", 'red'))
    
    print(colored(f"Disabled copying weights from DNN to BNN", 'red'))
    # for dnn_layer, bnn_good_prior_layer in zip(dnn_linear_layers, bnn_good_prior_linear_layers):
        
    #     # Set the prior
    #     bnn_good_prior_layer.weight.data = dnn_layer.weight.data.clone()
    #     print(colored(f"Linear weight copied from DNN to BNN", 'red'))
        
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
    
    def find_linear_layers(module):
        if isinstance(module, nn.Linear):
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