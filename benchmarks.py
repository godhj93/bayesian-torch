from utils import get_model, get_dataset
import argparse
from termcolor import colored
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os 
import torch.nn as nn

def main(args):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Get the model
    model = get_model(args).to(device)
    model.load_state_dict(torch.load(args.weight))
    model.eval()
    
    args.bs = 10000
    print(colored(f"Calculating ECE for {args.bs} data", 'green'))
    _, test_loader = get_dataset(args)

    for x,y in tqdm(test_loader):
        with torch.no_grad():
            logit_list = []
            for _ in tqdm(range(args.mc_runs), desc = 'MC Sampling'):
                logits, _ = model(x.to(device))
                logit_list.append(logits)
            logits = torch.stack(logit_list, dim=0).mean(dim=0)
        compute_ece_and_plot_confidence_vs_accuracy_batches(logits, y, args = args, n_bins=15)

    args.data = 'tinyimagenet'
    _, tiny_imagenet_loader = get_dataset(args)
    
    tpr = []  # True Positive Rate
    fpr = []  # False Positive Rate
    
    # for thres in thresholds:
    ood = test_ood_detection(model, test_loader, tiny_imagenet_loader)
        
    for thres, (in_correct, out_correct) in ood.items():
        
        # TPR (In-distribution의 정확한 탐지 비율)
        tpr.append(in_correct / len(test_loader.dataset))
        # FPR (Out-of-distribution의 잘못된 탐지 비율)
        fpr.append(1 - (out_correct / len(tiny_imagenet_loader.dataset)))

    fpr, tpr = zip(*sorted(zip(fpr, tpr)))

    auroc = np.trapz(tpr, fpr)

    # AUROC 커브 그리기
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUROC = {auroc:.2f}')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('AUROC Curve')
    plt.legend()
    plt.grid(True)
    save_dir = os.path.dirname(args.weight)
    save_path = os.path.join(save_dir, f'auroc_{auroc:.2f}.png')
    plt.savefig(save_path, dpi=300)

# 테스트 함수 정의
def test_ood_detection(model, in_loader, out_loader):
    model.eval().cuda()
    in_distribution_scores = []
    out_distribution_scores = []
    
    with torch.no_grad():
        for images, targets in tqdm(in_loader, desc='In-distribution'):
            outputs = []
            for _ in range(args.mc_runs):
                
                output, _ = model(images.cuda())
                outputs.append(output)
            outputs = torch.stack(outputs, dim=0).mean(dim=0)
            softmax_probs = nn.Softmax(dim=1)(outputs)
            max_probs, _ = torch.max(softmax_probs, dim=1)
            in_distribution_scores.extend(max_probs.cpu().numpy())
    
    # OOD 데이터 생성 (예: Uniform Noise)
    with torch.no_grad():
        for ood_data, _ in tqdm(out_loader, desc='Out-of-distribution'):
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
    thresholds = np.linspace(0.01, 1.0, 100)
    ood = {}
    
    pbar = tqdm(thresholds, desc='Thresholds')
    for threshold in pbar:
        in_correct = np.sum(in_distribution_scores > threshold)
        out_correct = np.sum(out_distribution_scores < threshold)   
        
        ood[threshold] = (in_correct, out_correct)

        pbar.set_description(f"Threshold: {threshold:.2f}, In-correct: {in_correct/len(in_distribution_scores)}, Out-correct: {out_correct}/{len(out_distribution_scores)}")
    
    return ood


def compute_ece_and_plot_confidence_vs_accuracy_batches(logits_batches, labels_batches, args, n_bins=15):
    """
    여러 배치에 대해 Confidence vs Accuracy 그래프를 그리며, ECE를 계산하는 함수.
    
    logits_batches: logits의 배치 리스트
    labels_batches: labels의 배치 리스트
    n_bins: ECE를 계산할 때 사용할 bin의 개수
    """
    all_confidences = []
    all_preds = []
    all_labels = []

    # 각 배치에 대해 처리
    # softmax를 적용하여 확률로 변환
    probs = torch.softmax(logits_batches, dim=1)
    preds = torch.argmax(probs, dim=1)
    confidences, _ = torch.max(probs, dim=1)

    # 결과 저장
    all_confidences.append(confidences.detach().cpu().numpy())
    all_preds.append(preds.detach().cpu().numpy())
    all_labels.append(labels_batches.detach().cpu().numpy())

    # numpy 배열로 변환
    all_confidences = np.concatenate(all_confidences)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Confidence bins 설정
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    accuracy_per_bin = []
    avg_confidence_per_bin = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # 현재 bin에 속하는 샘플들을 선택
        bin_mask = (all_confidences > bin_lower) & (all_confidences <= bin_upper)
        bin_size = np.sum(bin_mask)

        if bin_size > 0:
            # bin 내에서 정확도와 평균 확신도 계산
            accuracy = np.mean(all_preds[bin_mask] == all_labels[bin_mask])
            avg_confidence = np.mean(all_confidences[bin_mask])
            accuracy_per_bin.append(accuracy)
            avg_confidence_per_bin.append(avg_confidence)
        else:
            accuracy_per_bin.append(0)
            avg_confidence_per_bin.append((bin_lower + bin_upper) / 2)

    # ECE 계산
    ece = 0.0
    for acc, conf, bin_size in zip(accuracy_per_bin, avg_confidence_per_bin, np.diff(bin_boundaries)):
        ece += np.abs(conf - acc) * bin_size

    # 그래프 그리기
    plt.figure(figsize=(8, 6))
    plt.plot(avg_confidence_per_bin, accuracy_per_bin, marker='o', label="Model Calibration")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfect Calibration")
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f'Confidence vs Accuracy (ECE: {ece:.4f})')
    plt.legend()
    plt.grid(True)
    save_dir = os.path.dirname(args.weight)
    save_path = os.path.join(save_dir, f'ece_{ece:.4f}_in_n_bins_{n_bins}.png')
    plt.savefig(save_path, dpi=300)
    # plt.show()
    print(colored(f"The figure is saved at {save_path}", 'green'))

    return ece

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Bayesian Neural Networks Benchmarks')
    parser.add_argument('--model', type=str, default='resnet20', help='Model to train [simple, lenet, vgg7, resnet20]')
    parser.add_argument('--type', type=str, default='multi', help='Type of model [dnn, uni, multi]')
    parser.add_argument('--bs', type=int, default=10000, help='Batch size')
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
    parser.add_argument('--ece', action='store_true', help='Calculate ECE')
    args = parser.parse_args()
    
    '''
    Proposed (Ours): runs/cifar/resnet20/reference/bs128_lr0.001_mc50_temp_1.0_ep300_kd_True_alpha_0.0_moped_False_20240820-114307/best_model.pth
    '''
    
    assert args.weight is not None, 'Please provide a path to load weights'
    
    print(colored(args, 'green'))
    main(args)
