from models import *
from utils.utils import test_BNN, test_DNN, get_model, get_dataset
import argparse
from termcolor import colored
import torch
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, auc
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import label_binarize

def test_ood_detection_dnn(model, in_loader, out_loader, n_bins=15, args=None):
    
    model.eval().cuda()
    
    results = {
        'msp': {'scores': [], 'labels': []},
        'entropy': {'scores': [], 'labels': []}
    }
    
    all_confidences = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        # ──────────────────────────────────────────────
        # In-Distribution 처리
        # ──────────────────────────────────────────────
        for images, y in tqdm(in_loader, desc='In-distribution'):
            images = images.cuda()
            output = model(images)  # 단일 forward pass
            probs = F.softmax(output, dim=1)  # softmax 확률 계산
            
            # 최대 softmax 확률 (MSP)과 예측값 계산
            confidences, predictions = torch.max(probs, dim=1)
            all_confidences.extend(confidences.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
            # MSP 점수 저장 (ID는 label 0)
            results['msp']['scores'].extend(confidences.cpu().numpy())
            results['msp']['labels'].extend([0] * images.size(0))
            
            # 예측 엔트로피 계산: -sum(p * log(p))
            predictive_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            results['entropy']['scores'].extend(predictive_entropy.cpu().numpy())
            results['entropy']['labels'].extend([0] * images.size(0))
            
        # ──────────────────────────────────────────────
        # Out-of-Distribution 처리
        # ──────────────────────────────────────────────
        for images, _ in tqdm(out_loader, desc='Out-of-distribution'):
            images = images.cuda()
            output = model(images)
            probs = F.softmax(output, dim=1)
            
            confidences, _ = torch.max(probs, dim=1)
            
            # MSP 점수 저장 (OOD는 label 1)
            results['msp']['scores'].extend(confidences.cpu().numpy())
            results['msp']['labels'].extend([1] * images.size(0))
            
            # 예측 엔트로피 저장 (OOD는 label 1)
            predictive_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            results['entropy']['scores'].extend(predictive_entropy.cpu().numpy())
            results['entropy']['labels'].extend([1] * images.size(0))
    
    # ──────────────────────────────────────────────────────
    # In-Distribution 데이터에 대한 ECE 및 calibration plot 계산
    # ──────────────────────────────────────────────────────
    ece, avg_confidence_per_bin, accuracy_per_bin = compute_ece_and_plot_confidence_vs_accuracy_batches(
        torch.tensor(all_confidences),
        torch.tensor(all_preds),
        torch.tensor(all_labels),
        n_bins=n_bins,
    )
    print(f"ECE: {ece:.4f}")
    
    # ──────────────────────────────────────────────────────
    # OOD 검출을 위한 AUROC 계산 (MSP와 Entropy)
    # ──────────────────────────────────────────────────────
    for method in ['msp', 'entropy']:
        scores = np.array(results[method]['scores'])
        labels = np.array(results[method]['labels'])
        
        # MSP의 경우, in-distribution일수록 큰 값이므로 OOD 검출을 위해 음수 반전
        if method == 'msp':
            scores = -scores
        
        fpr, tpr, thresholds = roc_curve(labels, scores)
        auroc = roc_auc_score(labels, scores)
        print(f"AUROC ({method.upper()}): {auroc:.4f}")
    
    return results

def test_ood_detection_bnn(model, in_loader, out_loader, mc_runs=30, n_bins=15, args=None):
    model.eval().cuda()
    
    results = {
        'msp': {'scores': [], 'labels': []},
        'entropy': {'scores': [], 'labels': []},
        'mi': {'scores': [], 'labels': []}
    }
    
    all_confidences = []
    all_preds = []
    all_labels = []
    
    all_mean_probs_in = []
    all_labels_in = []
    
    with torch.no_grad():
        # ──────────────────────────────────────────────
        # In-Distribution 처리
        # ──────────────────────────────────────────────
        for images, y in tqdm(in_loader, desc='In-distribution'):
            images = images.cuda()
            mc_outputs = []
            for _ in range(mc_runs):
                output, _ = model(images)
                mc_outputs.append(F.softmax(output, dim=1))
            
            mc_outputs = torch.stack(mc_outputs, dim=0)  # [MC, Batch, Classes]
            mean_probs = torch.mean(mc_outputs, dim=0)   # [Batch, Classes]
            
            all_mean_probs_in.append(mean_probs.cpu().numpy())
            all_labels_in.append(y.cpu().numpy())
            
            confidences, predictions = torch.max(mean_probs, dim=1)
            
            all_confidences.extend(confidences.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        
            msp_scores, _ = torch.max(mean_probs, dim=1)

            results['msp']['scores'].extend(msp_scores.cpu().numpy())
            results['msp']['labels'].extend([0] * images.size(0))  
            
            mi_scores, predictive_entropy = compute_mutual_information(mc_outputs)
            results['mi']['scores'].extend(mi_scores.cpu().numpy())
            results['mi']['labels'].extend([0] * images.size(0))
            
            results['entropy']['scores'].extend(predictive_entropy.cpu().numpy())
            results['entropy']['labels'].extend([0] * images.size(0))
        
        # ──────────────────────────────────────────────
        # Out-of-Distribution 처리
        # ──────────────────────────────────────────────
        for images, _ in tqdm(out_loader, desc='Out-of-distribution'):
            images = images.cuda()
            mc_outputs = []
            for _ in range(mc_runs):
                output, _ = model(images)
                mc_outputs.append(F.softmax(output, dim=1))
            
            mc_outputs = torch.stack(mc_outputs, dim=0)
            mean_probs = torch.mean(mc_outputs, dim=0)
            
            msp_scores, _ = torch.max(mean_probs, dim=1)
            results['msp']['scores'].extend(msp_scores.cpu().numpy())
            results['msp']['labels'].extend([1] * images.size(0))  # out-dist -> 라벨 1
            
            mi_scores, predictive_entropy = compute_mutual_information(mc_outputs)
            results['mi']['scores'].extend(mi_scores.cpu().numpy())
            results['mi']['labels'].extend([1] * images.size(0))
            
            results['entropy']['scores'].extend(predictive_entropy.cpu().numpy())
            results['entropy']['labels'].extend([1] * images.size(0))
    
    # ──────────────────────────────────────────────────────
    # In-Distribution 데이터 ECE & Calibration plot
    # ──────────────────────────────────────────────────────
    ece, avg_confidence_per_bin, accuracy_per_bin = compute_ece_and_plot_confidence_vs_accuracy_batches(
                torch.tensor(all_confidences),
                torch.tensor(all_preds),
                torch.tensor(all_labels),
            )
    
    # ──────────────────────────────────────────────────────
    # OOD 검출용 AUROC (MSP/Entropy/MI) + ECE subplot
    # ──────────────────────────────────────────────────────
    centi_meter = 1/2.54
    palette = ['blue','red','green']  # 색상 예시( msp=blue, entropy=red, mi=green )

    
    method_colors = {'msp': 0, 'entropy': 1, 'mi': 2}
    
    for i, method in enumerate(['msp', 'entropy', 'mi']):
        scores = np.array(results[method]['scores'])
        labels = np.array(results[method]['labels'])
        
        # MSP는 ID일수록 값이 크므로, OOD 점수로 쓰기 위해 음수 반전
        if method == 'msp':
            scores = -scores
        
        fpr, tpr, thresholds = roc_curve(labels, scores)
        auroc = roc_auc_score(labels, scores)
        
        print(f"AUROC ({method.upper()}): {auroc:.4f}")
    
    return results

def compute_ece_and_plot_confidence_vs_accuracy_batches(confidences_batches, preds_batches, labels_batches, n_bins=15):
    
    # tensor라면 numpy 배열로 변환
    if torch.is_tensor(confidences_batches):
        all_confidences = confidences_batches.detach().cpu().numpy()
    else:
        all_confidences = np.array(confidences_batches)
    
    if torch.is_tensor(preds_batches):
        all_preds = preds_batches.detach().cpu().numpy()
    else:
        all_preds = np.array(preds_batches)
    
    if torch.is_tensor(labels_batches):
        all_labels = labels_batches.detach().cpu().numpy()
    else:
        all_labels = np.array(labels_batches)
    
    # Confidence bins 설정
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    N = len(all_confidences)  # 전체 샘플 수

    accuracy_per_bin = []
    avg_confidence_per_bin = []
    bin_fractions = []  # 각 bin별 (|Bin|/N)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        bin_mask = (all_confidences > bin_lower) & (all_confidences <= bin_upper)
        bin_size = np.sum(bin_mask)
        
        if bin_size > 0:
            acc = np.mean(all_preds[bin_mask] == all_labels[bin_mask])
            conf = np.mean(all_confidences[bin_mask])
            
            # ECE 누적 계산
            fraction = bin_size / N
            ece += np.abs(conf - acc) * fraction
            
            accuracy_per_bin.append(acc)
            avg_confidence_per_bin.append(conf)
            bin_fractions.append(fraction)
        else:
            # bin에 샘플이 없는 경우를 어떻게 처리할지는 옵션
            accuracy_per_bin.append(0.0)
            avg_confidence_per_bin.append((bin_lower + bin_upper)/2.0)
            bin_fractions.append(0.0)

    # 이후 ece, accuracy_per_bin, avg_confidence_per_bin 등을 반환
    return ece, avg_confidence_per_bin, accuracy_per_bin

def compute_mutual_information(mc_probabilities):
    """
    상호 정보량과 예측 엔트로피를 계산합니다.

    Parameters:
        mc_probabilities (torch.Tensor): [MC Samples, Batch Size, Num Classes] 형태의 예측 확률
    Returns:
        mutual_information (torch.Tensor): [Batch Size] 형태의 상호 정보량 값
        predictive_entropy (torch.Tensor): [Batch Size] 형태의 예측 엔트로피 값
    """
    # 평균 예측 확률 계산 (Mean of MC probabilities)
    mean_probabilities = torch.mean(mc_probabilities, dim=0)  # [Batch Size, Num Classes]
    
    # 예측 엔트로피 계산 (H[y | x])
    predictive_entropy = -torch.sum(mean_probabilities * torch.log(mean_probabilities + 1e-8), dim=1)  # [Batch Size]

    # 샘플별 엔트로피 계산 및 평균 (E[H[y | x, θ]])
    sample_entropies = -torch.sum(mc_probabilities * torch.log(mc_probabilities + 1e-8), dim=2)  # [MC Samples, Batch Size]
    expected_entropy = torch.mean(sample_entropies, dim=0)  # [Batch Size]

    # 상호 정보량 계산 (I[y, θ | x])
    mutual_information = predictive_entropy - expected_entropy  # [Batch Size]

    return mutual_information, predictive_entropy

def main(args):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    model = get_model(args = args, logger = logger)
    model.load_state_dict(torch.load(args.weight))
    print(colored(f"Pretrained weight is loaded from {args.weight}", 'green'))
    
    _, test_loader = get_dataset(args = args, logger = logger)
    
    if args.type == 'dnn':
        # ──────────────────────────────────────────────
        # ID Evaluation
        # ──────────────────────────────────────────────

        acc, nll = test_DNN(model, test_loader)
        print(colored(f"Acc: {acc:.4f}, NLL: {nll:.4f}", 'blue'))

        # ──────────────────────────────────────────────
        # OOD Evaluation
        # ──────────────────────────────────────────────
        args.data = 'svhn'
        _, svhn_loader = get_dataset(args, logger = logger)
        test_ood_detection_dnn(model, test_loader, svhn_loader)
        
        args.data = 'tinyimagenet'
        _, tiny_imagenet_loader = get_dataset(args, logger = logger)
        test_ood_detection_dnn(model, test_loader, tiny_imagenet_loader)
            
    elif args.type == 'uni':
        
        # ──────────────────────────────────────────────
        # ID Evaluation
        # ──────────────────────────────────────────────
        acc, nll, kld = test_BNN(model = model, test_loader = test_loader, bs = 128, device = device, mc_runs = args.mc_runs, args = args)
        
        print(colored(f"Acc: {acc:.4f}, NLL: {nll:.4f}, KLD: {kld:.4f}", 'blue'))

        # ──────────────────────────────────────────────
        # OOD Evaluation
        # Predictive Entropy(Total Uncertainty) = Model Uncertainty(Mutual Information) + Input Uncertainty(Expected Uncertainty)
        #! In MODEP Paper, they used the "predictive entropy" and "mutual information" for OOD detection
        # ──────────────────────────────────────────────
        args.data = 'svhn'
        _, svhn_loader = get_dataset(args, logger = logger)
        test_ood_detection_bnn(model, test_loader, svhn_loader, mc_runs=args.mc_runs)
        
        args.data = 'tinyimagenet'
        _, tiny_imagenet_loader = get_dataset(args, logger = logger)
        test_ood_detection_bnn(model, test_loader, tiny_imagenet_loader, mc_runs=args.mc_runs)

    else:
        
        raise NotImplementedError("Not implemented yet")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Test a Pretrained Model')
    parser.add_argument('--type', type=str, help='[dnn, uni, multi]')
    parser.add_argument('--model', type=str, help='Model to train [resnet20, densenet30, vgg7]')
    parser.add_argument('--bs', type=int, default=128, help='Batch size')
    parser.add_argument('--data', type=str, default='cifar10', help='Dataset to use [cifar10]')
    parser.add_argument('--mc_runs', type=int, default=30, help='Monte Carlo runs')
    parser.add_argument('--weight', type=str, help='Path to load weights')
    parser.add_argument('--moped', action='store_true', help='Use mode posterior')
    parser.add_argument('--multi_moped', action='store_true', help='Use mode posterior')
    parser.add_argument('--multi_gpu', action='store_true', help='Use mode posterior')
    
    args = parser.parse_args()
    
    print(colored(args, 'green'))
    main(args)