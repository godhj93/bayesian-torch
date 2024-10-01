from models import SimpleCNN, SimpleCNN_uni, SimpleCNN_multi, LeNet5, LeNet5_uni, LeNet5_multi, VGG7, VGG7_uni, VGG7_multi, resnet20_multi
from bayesian_torch.models.deterministic.resnet import resnet20 as resnet20_deterministic
from utils import get_model, get_dataset, test_DNN, test_BNN
import argparse
from termcolor import colored
import torch
import torch.nn.functional as F
import numpy as np
from torchmetrics.classification import AveragePrecision
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def main(args):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Get the model
    model = get_model(args).to(device)
    model.load_state_dict(torch.load(args.weight))
    
    # Get the dataset
    _, test_loader = get_dataset(args)
    args.data = 'tinyimagenet'
    _, tiny_imagenet_loader = get_dataset(args)
    # args.data = 'mnist'
    # _, mnist_loader = get_dataset(args)
    
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
        # acc, nll, kl = test_BNN(
        #     model = model,
        #     test_loader = test_loader,
        #     mc_runs = args.mc_runs,
        #     bs = args.bs,
        #     device = device   
        # )
        
        # with torch.no_grad():
        #     for i, (x, y) in enumerate(test_loader):
        #         outputs = []
        #         x, y = x.to(device), y.to(device)
        #         for _ in range(args.mc_runs):
        #             logits, _ = model(x)
        #             outputs.append(logits)
                    
        #         logits = torch.stack(outputs, dim=0).mean(dim=0)
        #         ece = expected_calibration_error(logits, y, n_bins=15)
        
        # ece, auroc, aupr = ece_and_ood(test_loader, test_loader, model, device, args)
        # print(colored(f'Accuracy: {acc:.5f}, NLL: {nll:.5f}, KL: {kl:.5f}, ECE: {ece:.5f}, AUROC: {auroc:.5f}, AUPR: {aupr:.5f}', 'green'))
        
        thresholds = np.linspace(0.1, 1.0, 91)
        print(thresholds)
        tpr = []  # True Positive Rate
        fpr = []  # False Positive Rate
        
        
        for thres in thresholds:
            in_correct, out_correct = test_ood_detection(model, test_loader, tiny_imagenet_loader, thres)
            
            # TPR (In-distribution의 정확한 탐지 비율)
            tpr.append(in_correct / len(test_loader.dataset))
            
            # FPR (Out-of-distribution의 잘못된 탐지 비율)
            fpr.append(1 - (out_correct / len(tiny_imagenet_loader.dataset)))

        fpr, tpr = zip(*sorted(zip(fpr, tpr)))

        auroc = np.trapz(tpr, fpr)

        # AUROC 커브 그리기
        plt.plot(fpr, tpr, label=f'AUROC = {auroc:.2f}')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('AUROC Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'auroc_curve_{args.weight.split("/")[-2]}.png', dpi=300)

# 테스트 함수 정의
def test_ood_detection(model, in_loader, out_loader, threshold):
    import torch.nn as nn
    print(f'Threshold: {threshold}')
    model.eval().cuda()
    in_distribution_scores = []
    out_distribution_scores = []

    with torch.no_grad():
        for images, _ in in_loader:
            outputs = []
            for _ in range(args.mc_runs):
                
                output, _ = model(images.cuda())
                outputs.append(output)
            outputs = torch.stack(outputs, dim=0).mean(dim=0)
            softmax_probs = nn.Softmax(dim=1)(outputs)
            max_probs, _ = torch.max(softmax_probs, dim=1)
            in_distribution_scores.extend(max_probs.cpu().numpy())
    
    # OOD 데이터 생성 (예: Uniform Noise)
    # ood_data = torch.rand((len(in_distribution_scores), 3, 32, 32))
    with torch.no_grad():
        for ood_data, _ in out_loader:
            outputs = []
            for _ in range(args.mc_runs):
                output, _ = model(ood_data.cuda())
                outputs.append(output)
            
            outputs = torch.stack(outputs, dim=0).mean(dim=0)
            softmax_probs = nn.Softmax(dim=1)(outputs)
            max_probs, _ = torch.max(softmax_probs, dim=1)
            out_distribution_scores.extend(max_probs.cpu().numpy())

    # 결과 계산
    in_distribution_scores = np.array(in_distribution_scores)
    out_distribution_scores = np.array(out_distribution_scores)

    # Threshold에 따라 OOD Detection
    in_correct = np.sum(in_distribution_scores > threshold)
    out_correct = np.sum(out_distribution_scores < threshold)

    
    print(f'In-distribution samples correctly detected: {in_correct}/{len(in_distribution_scores)}')
    print(f'Out-of-distribution samples correctly detected: {out_correct}/{len(out_distribution_scores)}')
    return in_correct, out_correct

def ece_and_ood(id_loader, ood_loader, model, device, args):
    # ID와 OOD 데이터를 저장할 리스트
    id_scores = []
    ood_scores = []
    ece_scores = []
    # ID 데이터에 대한 예측 수행
    with torch.no_grad():
        for i, (x, y) in enumerate(id_loader):
            outputs = []
            x, y = x.to(device), y.to(device)
            for _ in range(args.mc_runs):
                logits, _ = model(x)
                outputs.append(logits)

            logits = torch.stack(outputs, dim=0).mean(dim=0)
            # ECE 계산 (ID 데이터만 사용)
            ece = expected_calibration_error(logits, y, n_bins=15)
            ece_scores.append(ece)
            # Softmax를 통해 최대 확률 값을 얻어 ID 점수로 사용
            max_prob = torch.max(torch.softmax(logits, dim=1), dim=1).values
            id_scores.extend(max_prob.cpu().tolist())

    # OOD 데이터에 대한 예측 수행
    with torch.no_grad():
        for i, (x, _) in enumerate(ood_loader):  # OOD 데이터에서는 레이블 사용 안 함
            outputs = []
            x = x.to(device)
            for _ in range(args.mc_runs):
                logits, _ = model(x)
                outputs.append(logits)

            logits = torch.stack(outputs, dim=0).mean(dim=0)
            # Softmax를 통해 최대 확률 값을 얻어 OOD 점수로 사용
            max_prob = torch.max(torch.softmax(logits, dim=1), dim=1).values
            ood_scores.extend(max_prob.cpu().tolist())

    # ID와 OOD 점수 결합
    scores = id_scores + ood_scores
    labels = [0] * len(id_scores) + [1] * len(ood_scores)  # 0: ID, 1: OOD

    # AUROC 계산
    auroc = roc_auc_score(labels, scores)

    # AUPR 계산 using torchmetrics
    aupr_metric = AveragePrecision(task='binary')
    scores_tensor = torch.tensor(scores)
    labels_tensor = torch.tensor(labels)
    aupr = aupr_metric(scores_tensor, labels_tensor).item()
    
    # ROC 곡선 계산 및 그리기
    fpr, tpr, _ = roc_curve(labels, scores)

    # ROC 곡선 그리기 및 저장
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUROC = {auroc:.4f}', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve, ECE: {np.mean(ece_scores):.4f}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('roc_curve.png')
    plt.close()

    return ece, auroc, aupr

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
    parser.add_argument('--martern', action='store_true', help='Use Marternal')
    args = parser.parse_args()
    
    '''
    Proposed (Ours): runs/cifar/resnet20/reference/bs128_lr0.001_mc50_temp_1.0_ep300_kd_True_alpha_0.0_moped_False_20240820-114307/best_model.pth
    
    '''
    
    assert args.weight is not None, 'Please provide a path to load weights'
    
    print(colored(args, 'green'))
    main(args)
