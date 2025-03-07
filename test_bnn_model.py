import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from utils import get_dataset, get_model, test_DNN, test_BNN, train_BNN
from distill import get_conv_layers
from torch.distributions import Normal
import argparse
from termcolor import colored
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("talk")
palette = sns.color_palette("bright", 3)  

def test_ood_detection(model, in_loader, out_loader, mc_runs=1, n_bins=30, args=None):
    model.eval().cuda()
    
    results = {
        'msp': {'scores': [], 'labels': []},
        'entropy': {'scores': [], 'labels': []},
        'mi': {'scores': [], 'labels': []}
    }
    
    all_confidences = []
    all_preds = []
    all_labels = []
    
    # (추가) Multi-class ROC/PR을 위해 in-distribution의 예측 확률을 전부 저장할 리스트
    all_mean_probs_in = []
    all_labels_in = []
    
    with torch.no_grad():
        # ──────────────────────────────────────────────
        # 1) In-Distribution 처리
        # ──────────────────────────────────────────────
        for images, y in tqdm(in_loader, desc='In-distribution'):
            images = images.cuda()
            mc_outputs = []
            for _ in range(mc_runs):
                output, _ = model(images)
                mc_outputs.append(F.softmax(output, dim=1))
            
            mc_outputs = torch.stack(mc_outputs, dim=0)  # [MC, Batch, Classes]
            mean_probs = torch.mean(mc_outputs, dim=0)   # [Batch, Classes]
            
            # (추가) Multi-class ROC/PR을 위해 따로 저장
            all_mean_probs_in.append(mean_probs.cpu().numpy())
            all_labels_in.append(y.cpu().numpy())
            
            # confidence와 예측값 계산 (ECE 용)
            confidences, predictions = torch.max(mean_probs, dim=1)
            
            # 저장
            all_confidences.extend(confidences.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        
            # 나머지 (MSP, MI, entropy) 계산
            msp_scores, _ = torch.max(mean_probs, dim=1)
            # OOD 검출 시 MSP 점수는 "값이 낮을수록 OOD"이므로 음수 반전
            # 하지만 여기서는 'results'에 원본(MSP) 형태로 저장 후,
            # 나중에 AUROC 계산 시에만 -scores를 넣는 방식(코드 아래 참고)
            results['msp']['scores'].extend(msp_scores.cpu().numpy())
            results['msp']['labels'].extend([0] * images.size(0))  # in-dist -> 라벨 0
            
            mi_scores, predictive_entropy = compute_mutual_information(mc_outputs)
            results['mi']['scores'].extend(mi_scores.cpu().numpy())
            results['mi']['labels'].extend([0] * images.size(0))
            
            results['entropy']['scores'].extend(predictive_entropy.cpu().numpy())
            results['entropy']['labels'].extend([0] * images.size(0))
        
        # ──────────────────────────────────────────────
        # 2) Out-of-Distribution 처리
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
    # (A) In-Distribution 데이터 ECE & Calibration plot
    # ──────────────────────────────────────────────────────
    ece, avg_confidence_per_bin, accuracy_per_bin = compute_ece_and_plot_confidence_vs_accuracy_batches(
                torch.tensor(all_confidences),
                torch.tensor(all_preds),
                torch.tensor(all_labels),
            )
    
    # ──────────────────────────────────────────────────────
    # (B) OOD 검출용 AUROC (MSP/Entropy/MI) + ECE subplot
    # ──────────────────────────────────────────────────────
    centi_meter = 1/2.54
    palette = ['blue','red','green']  # 색상 예시( msp=blue, entropy=red, mi=green )
    
    plt.figure(figsize=(30*centi_meter, 30*centi_meter))
    
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
        
        plt.subplot(2, 2, i+1)
        plt.plot(
            fpr, tpr,
            label=f'ROC curve (area = {auroc:.4f})',
            lw=2,
            color=palette[method_colors[method]]
        )
        plt.plot([0, 1], [0, 1], 'k--', label="Random", lw=1.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate (FPR)', fontweight='bold')
        plt.ylabel('True Positive Rate (TPR)', fontweight='bold')
        plt.title(f'[OOD] ROC - {method.upper()}', fontweight='bold')
        plt.legend(loc="lower right")
    
    # ECE Plot
    plt.subplot(2, 2, 4)
    plt.plot(
        avg_confidence_per_bin, accuracy_per_bin,
        marker='o', label="Model Calibration",
        lw=2, color="purple"
    )
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfect Calibration", lw=1.5)
    plt.xlabel('Confidence', fontweight='bold')
    plt.ylabel('Accuracy', fontweight='bold')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title(f'Confidence vs Accuracy (ECE: {ece:.4f})', fontweight='bold')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # ──────────────────────────────────────────────────────
    # (C) Multi-class ROC / PR (In-Distribution 전용)
    # ──────────────────────────────────────────────────────
    # 1) in-distribution의 (mean_probs, 라벨) 최종 집계
    all_mean_probs_in = np.concatenate(all_mean_probs_in, axis=0)  # [N, n_classes]
    all_labels_in = np.concatenate(all_labels_in, axis=0)          # [N,]
    
    # n_classes: 모델 혹은 데이터셋에서 확인 (예: CIFAR-10이면 10)
    # 여기서는 예시로 n_classes=10이라 가정
    n_classes = all_mean_probs_in.shape[1]
    
    # label_binarize로 이진화 (shape = [N, n_classes])
    y_test_bin = label_binarize(all_labels_in, classes=range(n_classes))
    
    # 클래스별 ROC/PR을 담을 dict
    fpr, tpr, roc_auc = dict(), dict(), dict()
    precision, recall, pr_auc = dict(), dict(), dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], all_mean_probs_in[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], all_mean_probs_in[:, i])
        pr_auc[i] = auc(recall[i], precision[i])
    
    # (C-1) micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), all_mean_probs_in.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_bin.ravel(),
                                                                    all_mean_probs_in.ravel())
    pr_auc["micro"] = auc(recall["micro"], precision["micro"])
    
    # (C-2) macro-average
    # 직접 fpr/tpr을 평균내는 방식도 있지만, 일반적으론 roc_auc_score / average_precision_score를 쓰면 간편
    roc_auc["macro"] = roc_auc_score(y_test_bin, all_mean_probs_in, average='macro', multi_class='ovr')
    pr_auc["macro"] = average_precision_score(y_test_bin, all_mean_probs_in, average='macro')
    
    # 플롯을 위한 Figure 생성
    plt.figure(figsize=(40*centi_meter, 20*centi_meter))
    
    # ─────────────────────────────────────────────────
    # (C-3) Multi-class ROC
    # ─────────────────────────────────────────────────
    plt.subplot(1, 2, 1)
    # micro-average
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average (AUC={roc_auc["micro"]:.4f})',
             color='deeppink', linestyle=':', linewidth=4)
    # 무작위 분류선
    plt.plot([0,1],[0,1], 'k--', label='Random', lw=1.5)
    
    # 각 클래스별
    colors = plt.cm.get_cmap('tab10', n_classes)  # n_classes개 색상
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i],
                 label=f'Class {i} (AUC={roc_auc[i]:.4f})',
                 color=colors(i), lw=2)
    
    plt.title(f'Multi-class ROC (macro-AUC={roc_auc["macro"]:.4f})', fontweight='bold')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    
    # ─────────────────────────────────────────────────
    # (C-4) Multi-class PR
    # ─────────────────────────────────────────────────
    plt.subplot(1, 2, 2)
    # micro-average
    plt.plot(recall["micro"], precision["micro"],
             label=f'micro-average (AUC={pr_auc["micro"]:.4f})',
             color='deeppink', linestyle=':', linewidth=4)
    
    for i in range(n_classes):
        plt.plot(recall[i], precision[i],
                 label=f'Class {i} (AUC={pr_auc[i]:.4f})',
                 color=colors(i), lw=2)
    
    plt.title(f'Multi-class PR (macro-AUC={pr_auc["macro"]:.4f})', fontweight='bold')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    
    plt.tight_layout()
    plt.show()
    
    return results


#def test_ood_detection(model, in_loader, out_loader, mc_runs=1, n_bins=30, args=None):
    
    model.eval().cuda()
    
    results = {
        'msp': {'scores': [], 'labels': []},
        'entropy': {'scores': [], 'labels': []},
        'mi': {'scores': [], 'labels': []}
    }
    
    all_confidences = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
       # In-Distribution 데이터 처리
        for images, y in tqdm(in_loader, desc='In-distribution'):
            images = images.cuda()
            mc_outputs = []
            for _ in range(mc_runs):
                output, _ = model(images)
                mc_outputs.append(F.softmax(output, dim=1))
            
            mc_outputs = torch.stack(mc_outputs, dim=0)  # [MC, Batch, Classes]
            
            mean_probs = torch.mean(mc_outputs, dim=0) # [Batch, Classes]
            
            # confidence와 예측값 계산
            confidences, predictions = torch.max(mean_probs, dim=1)
            
            # 저장
            all_confidences.extend(confidences.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        
            # 나머지 계산 (MSP, MI, entropy 등)
            msp_scores, _ = torch.max(mean_probs, dim=1)
            results['msp']['scores'].extend(msp_scores.cpu().numpy())
            results['msp']['labels'].extend([0] * images.size(0))
            
            mi_scores, predictive_entropy = compute_mutual_information(mc_outputs)
            results['mi']['scores'].extend(mi_scores.cpu().numpy())
            results['mi']['labels'].extend([0] * images.size(0))
            
            results['entropy']['scores'].extend(predictive_entropy.cpu().numpy())
            results['entropy']['labels'].extend([0] * images.size(0))
        
        # Out-of-Distribution 데이터 처리
        for images, _ in tqdm(out_loader, desc='Out-of-distribution'):
            images = images.cuda()
            mc_outputs = []
            for _ in range(mc_runs):
                output, _ = model(images)
                mc_outputs.append(F.softmax(output, dim=1))
            mc_outputs = torch.stack(mc_outputs, dim=0)  # [MC, Batch, Classes]
            mean_probs = torch.mean(mc_outputs, dim=0)
            
            msp_scores, _ = torch.max(mean_probs, dim=1)
            results['msp']['scores'].extend(msp_scores.cpu().numpy())
            results['msp']['labels'].extend([1] * images.size(0))
            
            mi_scores, predictive_entropy = compute_mutual_information(mc_outputs)
            results['mi']['scores'].extend(mi_scores.cpu().numpy())
            results['mi']['labels'].extend([1] * images.size(0))
            
            results['entropy']['scores'].extend(predictive_entropy.cpu().numpy())
            results['entropy']['labels'].extend([1] * images.size(0))
    
    # in-distribution 데이터에 대한 ECE 및 calibration plot 계산
    ece, avg_confidence_per_bin, accuracy_per_bin = compute_ece_and_plot_confidence_vs_accuracy_batches(
                torch.tensor(all_confidences),
                torch.tensor(all_preds),
                torch.tensor(all_labels),
            )
    
    #결과 계산 및 출력
    centi_meter = 1/2.54  # centimeters in inches
    
    # plt.subplots(...) 대신 plt.figure(...) 만 사용 -> 불필요한 기본축이 생성되지 않음
    plt.figure(figsize=(30*centi_meter, 30*centi_meter))
    # plt.suptitle(f"Results of {args.weight.split('/')[-2]}", fontsize=10)

    # color 팔레트 인덱스( msp=0, entropy=1, mi=2 )를 위한 dict
    method_colors = {'msp': 0, 'entropy': 1, 'mi': 2}

    for i, method in enumerate(['msp', 'entropy', 'mi']):
        scores = np.array(results[method]['scores'])
        labels = np.array(results[method]['labels'])
        
        if method == 'msp':
            # MSP는 값이 클수록 ID 데이터이므로, OOD 점수로 사용하기 위해 부호를 반전합니다.
            scores = -scores
        
        fpr, tpr, thresholds = roc_curve(labels, scores)
        auroc = roc_auc_score(labels, scores)
        
        print(f"AUROC ({method.upper()}): {auroc:.4f}")
        plt.subplot(2, 2, i+1)
        plt.plot(
            fpr, tpr,
            label=f'ROC curve (area = {auroc:.4f})',
            lw=2,
            color=palette[method_colors[method]]
        )
        plt.plot([0, 1], [0, 1], 'k--', label="Random", lw=1.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate (FPR)', fontweight='bold')
        plt.ylabel('True Positive Rate (TPR)', fontweight='bold')
        plt.title(f'Receiver Operating Characteristic (ROC) - {method.upper()}', fontweight='bold')
        plt.legend(loc="lower right")
    
    # Plot ECE
    plt.subplot(2, 2, 4)
    plt.plot(
        avg_confidence_per_bin, accuracy_per_bin,
        marker='o', label="Model Calibration",
        lw=2, color="purple"
    )
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfect Calibration", lw=1.5)
    plt.xlabel('Confidence', fontweight='bold')
    plt.ylabel('Accuracy', fontweight='bold')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title(f'Confidence vs Accuracy (ECE: {ece:.4f})', fontweight='bold')
    plt.legend()
    plt.grid(True)
    
    # save_dir = os.path.dirname(args.weight)
    # save_path = os.path.join(save_dir, f'RESULT_of_{args.weight.split("/")[-2]}.png')
    # plt.savefig(save_path, dpi=300)

    plt.tight_layout()
    plt.show()
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

def main():
    
    #*──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    #*────────────────────────────────────────────── Have to Change the model to BNN ───────────────────────────────────────────────────────────
    #*──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    
    class opt:
        type = 'dnn'
        model = 'lenet'
        moped = False
        multi_moped = False
        multi_gpu = False
        data = 'cifar'
        bs = 128
        
    args_bnn = opt()
    args_bnn.type = 'uni'
    args_bnn.train_sampler = False
    args_bnn.t = 1.0
    args_bnn.bs = 128
    args_bnn.weight = args.weight

    bnn = get_model(args_bnn)

    ckpt = torch.load(args.weight)
    bnn.load_state_dict(ckpt)
    train_loader, test_loader = get_dataset(args_bnn)

    #! ID Dataset
    acc, nnl, kl = test_BNN(model = bnn,
                 test_loader = test_loader,
                 mc_runs = args.mc_runs,
                 bs = args.bs,
                 device = 'cuda')

    print(colored(f"Test accuracy: {acc:.3f}, NNL: {nnl:.3f}, KL: {kl:.3f}", 'green'))

    #! OOD Dataset 가져오기
    args_bnn.data = 'svhn'
    svhn_train_loader, svhn_test_loader = get_dataset(args_bnn)
    args_bnn.data = 'tinyimagenet'
    tinyimagenet_train_loader, tinyimagenet_test_loader = get_dataset(args_bnn)
    
    test_ood_detection(bnn, test_loader, svhn_test_loader, mc_runs=args.mc_runs)
    test_ood_detection(bnn, test_loader, tinyimagenet_test_loader, mc_runs=args.mc_runs)
     
if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description='Testing a Bayesian Neural Network')
    parser.add_argument('--weight', type=str, help='DNN weight path for distillation')
    parser.add_argument('--mc_runs', type=int, default=30, help='Number of Monte Carlo runs')
    parser.add_argument('--bs', type=int, default=128, help='Batch size')
    args = parser.parse_args()
    
    main()