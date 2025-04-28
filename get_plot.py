import re
import matplotlib.pyplot as plt

# 로그 문자열 (파일에서 읽어도 되고, 문자열로 직접 넣어도 됩니다)
log_text = """
DNN (Sparsity : 0%) → Acc: 0.8698, NLL: 0.4295, ECE: 0.0452,  
- OOD(SVHN, MSP): 0.8854,OOD(SVHN, Entropy): 0.8839,
- OOD(TINYIMAGENET, MSP): 0.8253 , OOD(TINYIMAGENET, Entropy): 0.8297

Prior N(0, 1) → Acc: 0.8922, NLL: 0.3315, KLD: 0.4078, ECE: 0.0605
- OOD(SVHN, MSP): 0.8411,OOD(SVHN, Entropy): 0.8320,OOD(SVHN, MI): 0.7683,
- OOD(Tiny, MSP): 0.8319 ,OOD(Tiny, Entropy):0.7910 ,OOD(Tiny, MI): 0.7723,

Sprasity : 0 % → Acc: 0.7699, NLL: 0.7078, KLD: 0.4245, ECE: 0.3038
- OOD(SVHN, MSP): 0.7289, OOD(SVHN, Entropy): 0.6316, OOD(SVHN, MI): 0.3488,
- OOD(TINYIMAGENET, MSP): 0.6718, OOD(TINYIMAGENET, Entropy): 0.7129, OOD(TINYIMAGENET, MI): 0.6823,

Sparsity : 0% (MOPED) →Acc: 0.8484, NLL: 0.4326, KLD: 0.4091, ECE: 0.1492
- OOD(SVHN, MSP): 0.8277, OOD(SVHN, Entropy): 0.7966, OOD(SVHN, MI): 0.6503,
- OOD(TINYIMAGENET, MSP): 0.8850, OOD(TINYIMAGENET, Entropy): 0.8976, OOD(TINYIMAGENET, MI): 0.7986,

Sparsity : 10% → Acc: 0.8055, NLL: 0.5835, KLD: 0.4256, ECE: 0.2675
- OOD(SVHN, MSP): 0.7591, OOD(SVHN, Entropy): 0.7067, OOD(SVHN, MI): 0.4098,
- OOD(TINYIMAGENET, MSP): 0.7464, OOD(TINYIMAGENET, Entropy): 0.7842, OOD(TINYIMAGENET, MI): 0.7144,

Sparsity : 20% →Acc: 0.8225, NLL: 0.5196, KLD: 0.4307, ECE: 0.2343
- OOD(SVHN, MSP): 0.7581, OOD(SVHN, Entropy): 0.7263, OOD(SVHN, MI): 0.4599,
- OOD(TINYIMAGENET, MSP): 0.8011, OOD(TINYIMAGENET, Entropy): 0.8329, OOD(TINYIMAGENET, MI): 0.7273,

Sparsity : 30% → Acc: 0.8293, NLL: 0.5023, KLD: 0.4301, ECE: 0.2143
- OOD(SVHN, MSP): 0.7744, OOD(SVHN, Entropy): 0.7460, OOD(SVHN, MI): 0.5306,
- OOD(TINYIMAGENET, MSP): 0.8196, OOD(TINYIMAGENET, Entropy): 0.8335, OOD(TINYIMAGENET, MI): 0.7386,

Sparsity : 40% → Acc: 0.8309, NLL: 0.4879, KLD: 0.4291, ECE: 0.2012
- OOD(SVHN, MSP): 0.7386, OOD(SVHN, Entropy): 0.7356, OOD(SVHN, MI): 0.4815,
- OOD(TINYIMAGENET, MSP): 0.8201, OOD(TINYIMAGENET, Entropy): 0.8480, OOD(TINYIMAGENET, MI): 0.7889,

Sparsity : 50% → Acc: 0.8429, NLL: 0.4667, KLD: 0.4274, ECE: 0.1917
- OOD(SVHN, MSP): 0.7980, OOD(SVHN, Entropy): 0.7861, OOD(SVHN, MI): 0.5271,
- OOD(TINYIMAGENET, MSP): 0.8217, OOD(TINYIMAGENET, Entropy): 0.8369, OOD(TINYIMAGENET, MI): 0.7754,

Sparsity : 60% →Acc: 0.8453, NLL: 0.4613, KLD: 0.4263, ECE: 0.1850
- OOD(SVHN, MSP): 0.8189, OOD(SVHN, Entropy): 0.7897, OOD(SVHN, MI): 0.6204,
- OOD(TINYIMAGENET, MSP): 0.7967, OOD(TINYIMAGENET, Entropy): 0.8145, OOD(TINYIMAGENET, MI): 0.7568,

Sparsity : 70% → Acc: 0.8504, NLL: 0.4373, KLD: 0.4245, ECE: 0.1666
- OOD(SVHN, MSP): 0.8081, OOD(SVHN, Entropy): 0.7961, OOD(SVHN, MI): 0.6122,
- OOD(TINYIMAGENET, MSP): 0.8105, OOD(TINYIMAGENET, Entropy): 0.8292, OOD(TINYIMAGENET, MI): 0.8101,

Sparsity : 80% → Acc: 0.8530, NLL: 0.4331, KLD: 0.4225, ECE: 0.1645
- OOD(SVHN, MSP): 0.7917, OOD(SVHN, Entropy): 0.7964, OOD(SVHN, MI): 0.6113,
- OOD(TINYIMAGENET, MSP): 0.8929, OOD(TINYIMAGENET, Entropy): 0.8975, OOD(TINYIMAGENET, MI): 0.7826,

Sparsity : 90% → Acc: 0.8594, NLL: 0.4092, KLD: 0.4199, ECE: 0.1526
- OOD(SVHN, MSP): 0.8738, OOD(SVHN, Entropy): 0.8781, OOD(SVHN, MI): 0.6829,
- OOD(TINYIMAGENET, MSP): 0.8816, OOD(TINYIMAGENET, Entropy): 0.8816, OOD(TINYIMAGENET, MI): 0.7967,
"""

def parse_log_blocks(log_text):
    blocks = re.split(r"\n\s*\n", log_text.strip())
    results = []
    for block in blocks:
        entry = {}
        if "DNN" in block:
            entry["label"] = "DNN"
        elif re.search(r"Prior\s+N\(0,\s*1\)", block):
            entry["label"] = "Prior N(0,1)"
        elif "MOPED" in block.upper():
            entry["label"] = "MOPED"
        else:
            match = re.search(r"Sparsity\s*[:]?[\s]*(\d+)%", block)
            if match:
                val = int(match.group(1))
                entry["sparsity"] = val
                entry["label"] = f"Sparsity {val}%"
        for metric in ["Acc", "NLL", "ECE"]:
            m = re.search(fr"{metric}:\s*([0-9.]+)", block)
            if m:
                entry[metric] = float(m.group(1))
        for ood in ["SVHN", "Tiny"]:
            for method in ["MSP", "Entropy", "MI"]:
                m = re.search(fr"OOD\({ood}.*?{method}\):\s*([0-9.]+)", block, re.IGNORECASE)
                if m:
                    entry[f"OOD({ood.upper()}, {method})"] = float(m.group(1))
        if "label" in entry:
            results.append(entry)
    return results

def build_plot_data(parsed):
    sparse = sorted([r for r in parsed if "sparsity" in r], key=lambda x: x["sparsity"])
    sparsity = [r["sparsity"] for r in sparse]
    metrics = list(sparse[0].keys())
    # Remove non-metric keys
    for k in ["label", "sparsity"]:
        metrics.remove(k)
    metric_dict = {m: [r.get(m) for r in sparse] for m in metrics}
    baselines = {r["label"]: r for r in parsed if "sparsity" not in r}
    return sparsity, metric_dict, baselines

def plot_metrics(s, md, baselines):
    styles = {
        "DNN": {"color":"green","linestyle":"dashdot"},
        "Prior N(0,1)": {"color":"blue","linestyle":"dotted"},
        "MOPED": {"color":"red","linestyle":"dashed"},
    }
    plt.style.use('tableau-colorblind10')
    n = len(md)
    cols=3; rows=(n+cols-1)//cols
    fig, axs = plt.subplots(rows, cols, figsize=(5*cols,4*rows))
    axs = axs.flatten()
    for i,(metric,vals) in enumerate(md.items()):
        ax=axs[i]
        ax.plot(s, vals, marker='o', linewidth=2, label="Sparse BNN")
        for name,style in styles.items():
            if metric in baselines.get(name,{}):
                ax.axhline(baselines[name][metric],
                           color=style["color"], linestyle=style["linestyle"],
                           linewidth=1.8,label=name)
        ax.set_title(metric, fontsize=11, fontweight='bold')
        ax.set_xlabel("Sparsity (%)"); ax.grid(True); ax.legend()
    for j in range(n,len(axs)): fig.delaxes(axs[j])
    plt.tight_layout(rect=[0,0,1,0.97])
    plt.suptitle("ResNet20 CIFAR10 std = 1e-2", fontsize=16, fontweight='bold')
    plt.subplots_adjust(top=0.9)
    plt.show()

parsed = parse_log_blocks(log_text)
sparsity, metric_dict, baselines = build_plot_data(parsed)
plot_metrics(sparsity, metric_dict, baselines)
