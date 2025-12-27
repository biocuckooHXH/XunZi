import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
)
import matplotlib.pyplot as plt

# ==============================
# 路径配置
# ==============================
# 已经算好的 all_others 指标
METRICS_OTHERS_PATH = r"E:\codes\PD\Metric_evaluation\Gene_disease_function\Disgenet_for_CTD\all_others_models\metrics_Disgenet_train_CTD_test.csv"

# XunZi / XunZiR 在 Disgenet_for_CTD 目录下的结果 (csv)
FOLDER_CTD_RESULT = r"E:\codes\PD\Metric_evaluation\Gene_disease_function\Disgenet_for_CTD"

# 输出合并后的表 & 图
OUT_METRICS_PATH = r"E:\codes\PD\Metric_evaluation\Gene_disease_function\Disgenet_for_CTD\metrics_with_Disgenet_for_CTD.csv"
FIG_DIR = r"E:\codes\PD\Metric_evaluation\Gene_disease_function\all_others\figures_with_Disgenet_for_CTD"

ID_COL = "id"
GROUP_COL = "group"

os.makedirs(FIG_DIR, exist_ok=True)

# ==============================
# 1. 读入已有的 all_others 指标，并统一列名
# ==============================
metrics_others = pd.read_csv(METRICS_OTHERS_PATH)

# 把之前脚本中的列名映射到你现在想要的命名
rename_map = {
    "precision": "precision_pos",
    "recall": "recall_pos",
    "specificity": "specificity_neg",
    "F1": "F1_pos",
}
metrics_others = metrics_others.rename(columns=rename_map)

# 确保都有这些列，没有的就补 NaN
needed_cols = [
    "model",
    "n_samples",
    "TP",
    "FN",
    "FP",
    "TN",
    "accuracy",
    "precision_pos",
    "recall_pos",
    "specificity_neg",
    "balanced_accuracy",
    "F1_pos",
    "MCC",
]
for col in needed_cols:
    if col not in metrics_others.columns:
        metrics_others[col] = np.nan

print("已有 all_others 指标：")
print(metrics_others.head())

# ==============================
# 2. 工具函数：给 Disgenet_for_CTD 目录里的 CSV 算同样的指标
# ==============================
def get_label_from_group(g):
    """Negative = 0, 其他 = 1"""
    if isinstance(g, str) and g.strip().lower() == "negative":
        return 0
    return 1

def parse_yes_no_from_text(text):
    """从一段文字里解析 Yes/No"""
    if not isinstance(text, str):
        return np.nan

    # 按行先看开头
    lines = text.splitlines()
    for line in lines:
        s = line.strip().lower()
        for prefix in ('**', '*', '"', "'", '>', '-'):
            if s.startswith(prefix):
                s = s[len(prefix):].lstrip()
        if s.startswith("yes"):
            return 1
        if s.startswith("no"):
            return 0

    s = text.strip().lower()
    if s.startswith("yes"):
        return 1
    if s.startswith("no"):
        return 0
    if "yes" in s and "no" not in s:
        return 1
    if "no" in s and "yes" not in s:
        return 0

    return np.nan

def get_pred_from_row(row):
    """
    从一行里找到预测列：
    优先：yesno / YesNo / pred / predict
    其次：response / answer（再从文本里解析）
    """
    # 1) 明确的 yes/no 列
    for col in ["yesno", "YesNo", "pred", "predict", "prediction"]:
        if col in row.index and pd.notna(row[col]):
            val = str(row[col]).strip().lower()
            if val.startswith("yes"):
                return 1
            if val.startswith("no"):
                return 0

    # 2) 文本回答列
    for col in ["response", "Response", "answer", "Answer"]:
        if col in row.index and pd.notna(row[col]):
            return parse_yes_no_from_text(str(row[col]))

    return np.nan

def compute_metrics_for_csv(fp):
    """给 Disgenet_for_CTD 目录下的一个 csv 计算性能指标"""
    df = pd.read_csv(fp)

    if ID_COL not in df.columns or GROUP_COL not in df.columns:
        raise ValueError(f"{fp} 缺少 {ID_COL} 或 {GROUP_COL} 列")

    df["label"] = df[GROUP_COL].apply(get_label_from_group)
    df["pred"] = df.apply(get_pred_from_row, axis=1)

    df_clean = df.dropna(subset=["pred"]).copy()
    if df_clean.empty:
        raise ValueError(f"{fp} 所有行都解析不到预测值")

    y_true = df_clean["label"].astype(int).values
    y_pred = df_clean["pred"].astype(int).values

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tp, fn, fp_, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    precision_pos, recall_pos, f1_pos, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1
    )
    specificity = tn / (tn + fp_) if (tn + fp_) > 0 else np.nan
    balanced_acc = (recall_pos + specificity) / 2
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    metrics = {
        "n_samples": len(df_clean),
        "TP": tp,
        "FN": fn,
        "FP": fp_,
        "TN": tn,
        "accuracy": acc,
        "precision_pos": precision_pos,
        "recall_pos": recall_pos,
        "specificity_neg": specificity,
        "balanced_accuracy": balanced_acc,
        "F1_pos": f1_pos,
        "MCC": mcc,
    }
    return metrics

# ==============================
# 3. 对 Disgenet_for_CTD 下的文件算指标
# ==============================
ctd_csv_files = glob.glob(os.path.join(FOLDER_CTD_RESULT, "*.csv"))
metrics_ctd = []

for fp in ctd_csv_files:
    model_name = os.path.splitext(os.path.basename(fp))[0]
    print(f"\n=== 计算 Disgenet_for_CTD 模型: {model_name} ===")
    try:
        met = compute_metrics_for_csv(fp)
        met["model"] = model_name
        metrics_ctd.append(met)
        print("  OK")
    except Exception as e:
        print(f"  跳过 {model_name}: {e}")

metrics_ctd_df = pd.DataFrame(metrics_ctd, columns=needed_cols)
print("\nDisgenet_for_CTD 指标：")
print(metrics_ctd_df)

# ==============================
# 4. 合并两个来源的指标 & 保存
# ==============================
all_metrics = pd.concat([metrics_others[needed_cols], metrics_ctd_df], ignore_index=True)
all_metrics = all_metrics.sort_values("accuracy", ascending=False)

all_metrics.to_csv(OUT_METRICS_PATH, index=False)
print(f"\n合并后的指标已保存到：{OUT_METRICS_PATH}")
print(all_metrics)

# ==============================
# 5. 画柱状图
# ==============================
metrics_to_plot = [
    "accuracy",
    "precision_pos",
    "recall_pos",
    "specificity_neg",
    "balanced_accuracy",
    "F1_pos",
    "MCC",
]

models = all_metrics["model"].tolist()

for metric in metrics_to_plot:
    plt.figure(figsize=(10, 4))
    plt.bar(models, all_metrics[metric].values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric)
    if metric == "MCC":
        plt.ylim(-1, 1)
    else:
        plt.ylim(0, 1)
    plt.title(f"{metric} (Disgenet train / CTD test)")
    plt.tight_layout()

    out_fig = os.path.join(FIG_DIR, f"{metric}_bar_Disgenet_for_CTD_all_models.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"保存柱状图: {out_fig}")
