import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------------
# 1) 데이터 입력 (사용자 제공)
# --------------------------------------------------------------------------------
# Sparse BNN (희소도 0%~77%)에 대한 데이터
sparsities = np.array([0, 10, 20, 30, 40, 50, 60, 70, 77])

acc_sparse = np.array([0.8723, 0.8784, 0.8829, 0.8851, 0.8897, 0.8928, 0.9042, 0.8992, 0.9016])
nll_sparse = np.array([0.3979, 0.3729, 0.3621, 0.3505, 0.3396, 0.3313, 0.3199, 0.3108, 0.3087])
ece_sparse = np.array([0.0102, 0.0192, 0.0273, 0.0408, 0.0551, 0.0641, 0.0529, 0.0720, 0.0861])

ood_svhn_msp_sparse     = np.array([0.8740, 0.8888, 0.8965, 0.8868, 0.8819, 0.8908, 0.8886, 0.8837, 0.8881])
ood_svhn_entropy_sparse = np.array([0.8837, 0.8961, 0.9036, 0.8970, 0.8883, 0.8908, 0.8965, 0.8711, 0.8830])
ood_svhn_mi_sparse      = np.array([0.8620, 0.8701, 0.8633, 0.8563, 0.8009, 0.8366, 0.8558, 0.8350, 0.7596])

ood_tiny_msp_sparse     = np.array([0.7996, 0.8391, 0.8583, 0.8742, 0.8437, 0.8394, 0.8988, 0.8908, 0.8696])
ood_tiny_entropy_sparse = np.array([0.8164, 0.8694, 0.8931, 0.9100, 0.8750, 0.8692, 0.9146, 0.8977, 0.8988])
ood_tiny_mi_sparse      = np.array([0.8189, 0.8379, 0.8550, 0.8466, 0.7960, 0.8069, 0.8711, 0.8444, 0.7245])

# DNN (Sparsity 0%)
acc_dnn = 0.8662
nll_dnn = 0.4278
ece_dnn = 0.0439
ood_svhn_msp_dnn     = 0.8580
ood_svhn_entropy_dnn = 0.8551
# DNN은 MI값이 미제공
ood_tiny_msp_dnn     = 0.7753
ood_tiny_entropy_dnn = 0.7722

# Prior N(0,1)
acc_prior = 0.8972
nll_prior = 0.3286
ece_prior = 0.0891
ood_svhn_msp_prior     = 0.7424
ood_svhn_entropy_prior = 0.7498
ood_svhn_mi_prior      = 0.6872
ood_tiny_msp_prior     = 0.8256
ood_tiny_entropy_prior = 0.8389
ood_tiny_mi_prior      = 0.7857

# MOPED (Sparsity 0%)
acc_moped = 0.8912
nll_moped = 0.3299
ece_moped = 0.0923
ood_svhn_msp_moped     = 0.8445
ood_svhn_entropy_moped = 0.8404
ood_svhn_mi_moped      = 0.7265
ood_tiny_msp_moped     = 0.8150
ood_tiny_entropy_moped = 0.8738
ood_tiny_mi_moped      = 0.7102

# --------------------------------------------------------------------------------
# 2) 서브플롯(3×3) 설정
# --------------------------------------------------------------------------------
fig, axs = plt.subplots(3, 3, figsize=(16, 12))
fig.subplots_adjust(wspace=0.3, hspace=0.3)

# 편의 함수: metric_name을 subplot의 Title로 표시 + y축 라벨로도 사용
def plot_metric(ax, 
                y_sparse, 
                dnn_val=None, moped_val=None, prior_val=None,
                metric_name="", ylim=None, show_legend=True):
    # Sparse BNN 곡선
    ax.plot(sparsities, y_sparse, marker='o', label='Sparse BNN')
    
    # DNN
    if dnn_val is not None:
        ax.axhline(dnn_val, color='black', linestyle='--', label='DNN')
    # MOPED
    if moped_val is not None:
        ax.axhline(moped_val, color='green', linestyle='--', label='MOPED')
    # Prior
    if prior_val is not None:
        ax.axhline(prior_val, color='red', linestyle='--', label='Prior N(0,1)')
    
    # subplot 제목
    ax.set_title(metric_name, fontsize=11)
    # y축 라벨
    ax.set_ylabel(metric_name)
    # x축 라벨
    ax.set_xlabel('Sparsity (%)')
    # grid 추가
    ax.grid(True)
    
    if ylim is not None:
        ax.set_ylim(ylim)
    
    if show_legend:
        ax.legend(fontsize=9)

# --------------------------------------------------------------------------------
# 3) 실제 그래프 그리기
# --------------------------------------------------------------------------------

# (1) Accuracy
plot_metric(axs[0, 0], 
            y_sparse=acc_sparse,
            dnn_val=acc_dnn, 
            moped_val=acc_moped,
            prior_val=acc_prior,
            metric_name="Accuracy")

# (2) NLL
plot_metric(axs[0, 1], 
            y_sparse=nll_sparse,
            dnn_val=nll_dnn, 
            moped_val=nll_moped,
            prior_val=nll_prior,
            metric_name="NLL")

# (3) ECE
plot_metric(axs[0, 2], 
            y_sparse=ece_sparse,
            dnn_val=ece_dnn, 
            moped_val=ece_moped,
            prior_val=ece_prior,
            metric_name="ECE")

# ------------------------ OOD (SVHN) ------------------------
# (4) OOD(SVHN, MSP)
plot_metric(axs[1, 0],
            y_sparse=ood_svhn_msp_sparse,
            dnn_val=ood_svhn_msp_dnn,
            moped_val=ood_svhn_msp_moped,
            prior_val=ood_svhn_msp_prior,
            metric_name="OOD(SVHN, MSP)")

# (5) OOD(SVHN, Entropy)
plot_metric(axs[1, 1],
            y_sparse=ood_svhn_entropy_sparse,
            dnn_val=ood_svhn_entropy_dnn,
            moped_val=ood_svhn_entropy_moped,
            prior_val=ood_svhn_entropy_prior,
            metric_name="OOD(SVHN, Entropy)")

# (6) OOD(SVHN, MI)
plot_metric(axs[1, 2],
            y_sparse=ood_svhn_mi_sparse,
            dnn_val=None,  # DNN은 MI 값 없음
            moped_val=ood_svhn_mi_moped,
            prior_val=ood_svhn_mi_prior,
            metric_name="OOD(SVHN, MI)")

# ------------------------ OOD (Tiny) ------------------------
# (7) OOD(Tiny, MSP)
plot_metric(axs[2, 0],
            y_sparse=ood_tiny_msp_sparse,
            dnn_val=ood_tiny_msp_dnn,
            moped_val=ood_tiny_msp_moped,
            prior_val=ood_tiny_msp_prior,
            metric_name="OOD(Tiny, MSP)")

# (8) OOD(Tiny, Entropy)
plot_metric(axs[2, 1],
            y_sparse=ood_tiny_entropy_sparse,
            dnn_val=ood_tiny_entropy_dnn,
            moped_val=ood_tiny_entropy_moped,
            prior_val=ood_tiny_entropy_prior,
            metric_name="OOD(Tiny, Entropy)")

# (9) OOD(Tiny, MI)
plot_metric(axs[2, 2],
            y_sparse=ood_tiny_mi_sparse,
            dnn_val=None,  # DNN은 MI 값 없음
            moped_val=ood_tiny_mi_moped,
            prior_val=ood_tiny_mi_prior,
            metric_name="OOD(Tiny, MI)")

# 전체 제목 (suptitle)
plt.suptitle("Sparse BNN vs. DNN / MOPED / Prior(N(0,1)) on CIFAR-10", fontsize=16, y=0.98)
plt.show()
