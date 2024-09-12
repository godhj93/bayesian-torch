from models import SimpleCNN, SimpleCNN_uni, SimpleCNN_multi, LeNet5, LeNet5_uni, LeNet5_multi, VGG7, VGG7_uni, VGG7_multi, resnet20_multi
from bayesian_torch.models.deterministic.resnet import resnet20 as resnet20_deterministic
from utils import get_model, get_dataset, test_DNN, test_BNN
import argparse
from termcolor import colored
import torch
import torch.nn.functional as F
import numpy as np
def main(args):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Get the model
    model = get_model(args).to(device)
    model.load_state_dict(torch.load(args.weight))
    
    # Get the dataset
    train_loader, test_loader = get_dataset(args)
    
    if args.type == 'dnn':
        acc, nll = test_DNN(
            model = model,
            test_loader = test_loader,
        )
        
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                outputs = []
                x, y = x.to(device), y.to(device)
                for _ in range(args.mc_runs):
                    logits = model(x)
                    outputs.append(logits)
                    
                logits = torch.stack(outputs, dim=0).mean(dim=0)
                ece = expected_calibration_error(logits, y, n_bins=15)
        
        print(colored(f'Accuracy: {acc:.5f}, NLL: {nll:.5f}, ECE: {ece:.5f}', 'green'))
        
    else:
        acc, nll, kl = test_BNN(
            model = model,
            test_loader = test_loader,
            mc_runs = args.mc_runs,
            bs = args.bs,
            device = device   
        )
        
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                outputs = []
                x, y = x.to(device), y.to(device)
                for _ in range(args.mc_runs):
                    logits, _ = model(x)
                    outputs.append(logits)
                    
                logits = torch.stack(outputs, dim=0).mean(dim=0)
                ece = expected_calibration_error(logits, y, n_bins=15)
                
        print(colored(f'Accuracy: {acc:.5f}, NLL: {nll:.5f}, KL: {kl:.5f}, ECE: {ece:.5f}', 'green'))            
    
    

def expected_calibration_error(logits, labels, n_bins=15) -> float:
    """
    Calculate Expected Calibration Error (ECE).

    Args:
    - logits (torch.Tensor): Model's predicted probabilities, shape (N, num_classes).
    - labels (torch.Tensor): Ground truth labels, shape (N,).
    - n_bins (int): Number of bins to use for calibration.

    Returns:
    - ece (float): The Expected Calibration Error.
    """

    # Get the predicted probabilities and the corresponding predicted labels
    y_hat = F.softmax(logits, dim=1)
    confidences, predicted_labels = torch.max(y_hat, 1)

    # Convert to numpy for easier handling
    confidences = confidences.detach().cpu().numpy()
    predicted_labels = predicted_labels.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    # Initialize bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0

    # Calculate accuracy and confidence within each bin
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find indices where confidence is within the current bin range
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(predicted_labels[in_bin] == labels[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            
            # Calculate ECE contribution for the current bin
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece
    
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Bayesian Neural Networks Benchmarks')
    parser.add_argument('--model', type=str, default='resnet20', help='Model to train [simple, lenet, vgg7, resnet20]')
    parser.add_argument('--type', type=str, default='multi', help='Type of model [dnn, uni, multi]')
    parser.add_argument('--bs', type=int, default=1024, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--mc_runs', type=int, default=50, help='Number of Monte Carlo runs')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--distill', action='store_true', help='Distill the model')
    parser.add_argument('--alpha', type=float, default=0.1, help='Distillation parameter')
    parser.add_argument('--moped', action='store_true', help='Use MoPeD')
    parser.add_argument('--data', type=str, default='cifar', help='Dataset to train on [mnist, cifar]')
    parser.add_argument('--weight', type=str, help='Path to load weights')
    parser.add_argument('--multi-gpu', action='store_true', help='Use multi-GPU')
    args = parser.parse_args()
    
    '''
    Proposed (Ours): runs/cifar/resnet20/reference/bs128_lr0.001_mc50_temp_1.0_ep300_kd_True_alpha_0.0_moped_False_20240820-114307/best_model.pth
    
    '''
    
    assert args.weight is not None, 'Please provide a path to load weights'
    
    print(colored(args, 'green'))
    main(args)
