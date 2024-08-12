from tqdm import tqdm
from termcolor import colored
from torch.distributions import LowRankMultivariateNormal
from torch.optim import Adam
import torch.nn as nn
from bayesian_torch.layers.variational_layers.conv_variational import Conv2dReparameterization, Conv2dReparameterization_Multivariate
import numpy as np
import copy 

def distill(dnn, bnn, steps, writer, device = 'cuda'):
    
    bnn_good_prior = copy.deepcopy(bnn)
    
    dnn_conv_layers = get_conv_layers(dnn)
    bnn_conv_layers = get_conv_layers(bnn)
    
    MSE = nn.MSELoss()

    optimizer = Adam(bnn.parameters(), lr = 0.001)
    
    pbar = tqdm(enumerate(range(steps)), total = steps)
    for idx, _ in pbar:
        
        loss = 0
            
        for dnn_layer, bnn_layer in zip(dnn_conv_layers, bnn_conv_layers):
            # print(f"Distilling from {dnn_layer} to {bnn_layer}")
            w_dnn = dnn_layer.weight.data.view(-1).to(device)
            
            mu_flat = bnn_layer.mu_kernel.view(-1).to(device)
            
            L, B = bnn_layer.get_covariance_param()
            
            L, B = L.to(device), B.to(device)
            
            # Check var is on the same device 
            w_sample = LowRankMultivariateNormal(mu_flat, L, B).rsample().reshape(w_dnn.size())
            
            loss += (w_dnn - w_sample).pow(2).mean()
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        pbar.set_description(f"Distillation Loss: {loss.item():.2f}")
            
        writer.add_scalar('Distillation Loss', loss.item(), idx)
            
    # Set the prior and variational parameters 
    bnn_good_prior_conv_layers = get_conv_layers(bnn_good_prior)
    bnn_good_prior_linear_layers = get_linear_layers(bnn_good_prior)
    dnn_linear_layers = get_linear_layers(dnn)
    
    for bnn_layer, bnn_good_prior_layer in zip(bnn_conv_layers, bnn_good_prior_conv_layers):
        
        # Set the prior
        bnn_good_prior_layer.prior_mean = bnn_layer.mu_kernel.detach().clone().flatten()
        bnn_good_prior_layer.prior_cov_L = bnn_layer.get_covariance_param()[0].detach()
        bnn_good_prior_layer.prior_cov_B = bnn_layer.get_covariance_param()[1].detach()
        # bnn_good_prior_layer.prior_variance = bnn_layer.get_covariance_param()
        
        # Set the variational parameters
        bnn_good_prior_layer.mu_kernel = bnn_layer.mu_kernel
        
    for dnn_layer, bnn_good_prior_layer in zip(dnn_linear_layers, bnn_good_prior_linear_layers):
        
        # Set the prior
        bnn_good_prior_layer = dnn_layer.clone()
        print(colored(f"Linear weight copied from DNN to BNN", 'red'))
        
        
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