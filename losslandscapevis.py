import argparse, random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from utils.models.densenet_uni import densenet_bc_30_uni  

def _save_npz(path: str, X, Y, ZA, ZB):
    np.savez_compressed(path, X=X, Y=Y, ZA=ZA, ZB=ZB)

def _make_filterwise_direction(model: torch.nn.Module, device: str = "cuda") -> torch.Tensor:
    """
    Filter Normalization Method With Random Directions
    https://github.com/tomgoldstein/loss-landscape/blob/master/net_plotter.py
    ! Default of BN and Bias is zero
    """
    dirs = []
    for W in model.parameters():               # [conv1_weight, conv1_bias, BN, conv2_weight, ...]
        if W.dim() >= 2:   
            d = torch.randn_like(W, device=device) # [output channel, input channel, height, width]
            Wf = W.reshape(W.size(0), -1)          # [output channel, input channel * height * width]
            df = d.reshape(W.size(0), -1)          # [output channel, input channel * height * width]
            d  = (df / (df.norm(dim=1, keepdim=True) + 1e-12) * (Wf.norm(dim=1, keepdim=True) + 1e-12)).view_as(W) 
            # [output channel, input channel * height * width] / [output channel, 1] * [output channel, 1] -> #[output channel, input channel, height, width]
            dirs.append(d)
        else:
            d = torch.zeros_like(W, device=device)
            dirs.append(d)  
    return parameters_to_vector(dirs)

@torch.no_grad()
def _compute_landscape_Z(model: torch.nn.Module, loader: DataLoader, *,
                         device="cuda", alpha=1.0, grid=21, planes=10,
                         mc_runs=30, max_samples=1024) -> np.ndarray:
    model.eval()
    """
    Model weight to vector
    """
    theta0 = parameters_to_vector([p.detach().clone() for p in model.parameters()]).to(device) 

    """
    Subset of the dataset
    """
    base_ds = loader.dataset
    sub_loader = DataLoader(Subset(base_ds, list(range(min(len(base_ds), max_samples)))),
                            batch_size=1, shuffle=False)
    
    axes = np.linspace(-alpha, alpha, grid)
    planes_Z = []

    for k in range(planes):
        """
        Basis
        """
        d1 = _make_filterwise_direction(model, device)
        d2 = _make_filterwise_direction(model, device)
        Z = np.zeros((grid, grid), dtype=np.float32)
        pbar = tqdm(total=grid*grid, desc=f"plane {k+1}/{planes}")
        """
        Span
        """
        for i, a in enumerate(axes):
            for j, b in enumerate(axes):
                vector_to_parameters(theta0 + a*d1 + b*d2, model.parameters())
                loss_sum = n = 0.0
                for xb, yb in sub_loader:
                    xb = xb.to(device); yb = yb.to(device)
                    outs = [model(xb)[0] for _ in range(mc_runs)] # [[logits1, logits2, ...], [logits1, logits2, ...], ...]
                    logits = torch.stack(outs).mean(0) # [logits1, logits2, ...] (mean over MC runs)
                    nll = F.cross_entropy(logits, yb, reduction='sum')
                    loss_sum += nll.item(); n += xb.size(0)
                Z[j,i] = loss_sum / n
                pbar.update(1)
        pbar.close()
        planes_Z.append(Z)

    vector_to_parameters(theta0, model.parameters())
    return np.mean(planes_Z, axis=0)

def visualize_overlay(modelA: torch.nn.Module, modelB: torch.nn.Module, loader: DataLoader, *,
                       device="cuda", alpha=1.0, grid=21, planes=10, mc_runs=30, max_samples=1024,
                       contour=True, alpha_plot=0.6, labels=("A","B") , save_path=None):

    ZA = _compute_landscape_Z(modelA, loader, device=device, alpha=alpha, grid=grid,
                              planes=planes, mc_runs=mc_runs, max_samples=max_samples)
    ZB = _compute_landscape_Z(modelB, loader, device=device, alpha=alpha, grid=grid,
                              planes=planes, mc_runs=mc_runs, max_samples=max_samples)

    axes = np.linspace(-alpha, alpha, grid)
    X, Y = np.meshgrid(axes, axes)
    _save_npz(save_path, X, Y, ZA, ZB)

    plt.figure(figsize=(6,5))
    if contour:
        cs1 = plt.contour(X, Y, ZA, levels=20, colors='red',  alpha=alpha_plot)
        cs2 = plt.contour(X, Y, ZB, levels=20, colors='blue', alpha=alpha_plot)
        plt.clabel(cs1, fmt="%.2f", fontsize=6)
        plt.clabel(cs2, fmt="%.2f", fontsize=6)
    else:
        ax = plt.axes(projection='3d')  
        ax.plot_surface(X, Y, ZA, color='red',  alpha=alpha_plot, edgecolor='none')
        ax.plot_surface(X, Y, ZB, color='blue', alpha=alpha_plot, edgecolor='none')
        ax.set_zlabel('Loss')

    plt.xlabel('d1'); plt.ylabel('d2')
    plt.title('Loss‑Landscape Overlay (red vs blue)')
    plt.legend(handles=[plt.Line2D([0],[0],color='red',label=labels[0],linewidth=4,alpha=alpha_plot),
                        plt.Line2D([0],[0],color='blue',label=labels[1],linewidth=4,alpha=alpha_plot)])
    plt.tight_layout(); plt.show()

def get_parser():
    p = argparse.ArgumentParser("Overlay two loss landscapes (CIFAR‑10, BNN)")
    p.add_argument('--weightA', required=True)
    p.add_argument('--weightB', required=True)
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--grid', type=int, default=21)
    p.add_argument('--planes', type=int, default=10)
    p.add_argument('--mc', type=int, default=30)
    p.add_argument('--max_samples', type=int, default=2000)
    p.add_argument('--contour', action='store_true')
    p.add_argument('--save', type=str, default=None,   # ← 추가
                   help='경로를 주면 X,Y,ZA,ZB를 npz로 저장')
    p.add_argument('--prior_type', type=str, help='Prior type [normal, laplace]')
    return p


def main():
    args = get_parser().parse_args()
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    """
    CIFAR 10 
    Test Dataset Augmentation
    """
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)),
    ])
    test_set = datasets.CIFAR10('~/.torch/datasets', train=False, transform=tf, download=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

    """
    Load Model
    """
    modelA = densenet_bc_30_uni().to(device) 
    modelB = densenet_bc_30_uni().to(device)
    modelA.load_state_dict(torch.load(args.weightA, map_location=device), strict=False)
    modelB.load_state_dict(torch.load(args.weightB, map_location=device), strict=False)
    modelA.eval(); modelB.eval()

    visualize_overlay(modelA, modelB, test_loader, device=device, alpha=args.alpha, grid=args.grid,
                      planes=args.planes, mc_runs=args.mc, max_samples=args.max_samples,
                      contour=args.contour, labels=('A','B'), save_path=args.save)

if __name__ == '__main__':
    main()
