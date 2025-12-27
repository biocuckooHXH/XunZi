import os
import gc
import numpy as np
import pandas as pd
from bert_score import score as bert_score
import torch
from transformers import AutoTokenizer

# =====================================================
# 1. 配置区 —— 你只要改这里
# =====================================================
data_dir = "/work/home/acssag9hf5/XunZi/XunZI-R/Metric/XunZI_independent_dataset"  # 这个文件夹
input_csv_name = "XunZiR_train_on_Disgenet.csv"  # 👈 改成你要算的那个 csv 文件名
input_csv_path = os.path.join(data_dir, input_csv_name)

ref_col = "Mechanism_answer"   # 参考解释列名
resp_col = "response"          # 模型输出列名

bert_model_path = "/work/home/acssag9hf5/XunZi/XunZI-R/Metric/Models/scibert_scivocab_uncased"

# 输出：在原 csv 的基础上加上 BERTScore，每一行都有
output_csv_name = input_csv_name.replace(".csv", "_with_bertscore.csv")
output_csv_path = os.path.join(data_dir, output_csv_name)

# BERTScore 分块设置
BERT_CHUNK_SIZE   = 20000   # 一次最多处理 2w 条样本
BERT_BATCH_SIZE   = 128     # BERTScore 内部 batch_size

# =====================================================
# 2. 选择设备：优先用 GPU
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(">>> BERTScore will run on device:", device)

# =====================================================
# 3. 加载 tokenizer
# =====================================================
print("Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
except Exception as e:
    print(f"Failed to load local tokenizer: {e}")
    print("Falling back to online SciBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")


def smart_truncate_with_tokenizer(text, tokenizer, max_tokens=400):
    """
    使用 SciBERT tokenizer 截断，保证 subword 长度不超过 max_tokens。
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    try:
        tokens = tokenizer.tokenize(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        truncated = tokenizer.convert_tokens_to_string(tokens)
        return truncated.strip()
    except Exception:
        # 兜底：按单词数截断
        words = text.split()
        return " ".join(words[:max_tokens // 2])


# =====================================================
# 4. 读取单个 csv，准备 refs 和 hyps
# =====================================================
print(f"\n>>> Loading file: {input_csv_path}")
df = pd.read_csv(input_csv_path)
print("Columns:", df.columns.tolist())

if ref_col not in df.columns:
    raise KeyError(f"{input_csv_path} does not contain reference column '{ref_col}'")
if resp_col not in df.columns:
    raise KeyError(f"{input_csv_path} does not contain response column '{resp_col}'")

# 填充空值
df[ref_col] = df[ref_col].fillna("").astype(str)
df[resp_col] = df[resp_col].fillna("").astype(str)

# 截断参考 & 预测文本
df["ref_trunc"] = df[ref_col].apply(
    lambda x: smart_truncate_with_tokenizer(x, tokenizer, max_tokens=400)
)
df["hyp_trunc"] = df[resp_col].apply(
    lambda x: smart_truncate_with_tokenizer(x, tokenizer, max_tokens=400)
)

# 过滤：参考文本太短的行可以选择过滤掉（可选）
valid_mask = df["ref_trunc"].str.len() > 5
valid_df = df[valid_mask].copy().reset_index(drop=True)

print(f"Total samples: {len(df)}, valid for BERTScore: {len(valid_df)}")

refs = valid_df["ref_trunc"].tolist()
hyps = valid_df["hyp_trunc"].tolist()
n_samples = len(refs)


# =====================================================
# 5. 分块计算 BERTScore 的函数
# =====================================================
def bertscore_chunked(hyps_list, refs_list, model_type, device="cuda",
                      batch_size=16, chunk_size=20000):
    """
    对长列表分块计算 BERTScore，避免一次性占用过多内存。
    返回拼接后的 P/R/F1 张量（长度 == len(hyps_list)）
    """
    assert len(hyps_list) == len(refs_list)
    n = len(hyps_list)
    all_P = []
    all_R = []
    all_F1 = []

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        hyps_chunk = hyps_list[start:end]
        refs_chunk = refs_list[start:end]

        print(f"  BERTScore chunk {start} : {end} ...")
        P, R, F1 = bert_score(
            hyps_chunk,
            refs_chunk,
            lang="en",
            model_type=model_type,
            num_layers=12,
            rescale_with_baseline=False,
            batch_size=batch_size,
            device=device,
        )
        all_P.append(P.cpu())
        all_R.append(R.cpu())
        all_F1.append(F1.cpu())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        del P, R, F1
        gc.collect()

    P_full = torch.cat(all_P, dim=0)
    R_full = torch.cat(all_R, dim=0)
    F1_full = torch.cat(all_F1, dim=0)

    return P_full, R_full, F1_full


# =====================================================
# 6. 计算该 csv 的 BERTScore
# =====================================================
print("Computing BERTScore (chunked) for this file...")

P_full, R_full, F1_full = bertscore_chunked(
    hyps_list=hyps,
    refs_list=refs,
    model_type=bert_model_path,
    device=device,
    batch_size=BERT_BATCH_SIZE,
    chunk_size=BERT_CHUNK_SIZE,
)

bert_P_mean  = float(P_full.mean())
bert_R_mean  = float(R_full.mean())
bert_F1_mean = float(F1_full.mean())

print(f"\nBERTScore P (mean):  {bert_P_mean:.4f}")
print(f"BERTScore R (mean):  {bert_R_mean:.4f}")
print(f"BERTScore F1 (mean): {bert_F1_mean:.4f}")

# =====================================================
# 7. 把逐行 P_full/R_full/F1_full 写回 DataFrame
#    注意要对齐：只对 valid_mask == True 的行写入
# =====================================================
# 初始化为 NaN，保证行数与原 df 一致
df["BERT_P"]  = np.nan
df["BERT_R"]  = np.nan
df["BERT_F1"] = np.nan

# 对 valid_df 的索引序列写回对应分数
valid_indices = df[valid_mask].index.tolist()  # 原 df 的行号
df.loc[valid_indices, "BERT_P"]  = P_full.numpy()
df.loc[valid_indices, "BERT_R"]  = R_full.numpy()
df.loc[valid_indices, "BERT_F1"] = F1_full.numpy()

# =====================================================
# 8. 保存带行级 BERTScore 的新 csv
# =====================================================
df.to_csv(output_csv_path, index=False)
print(f"\nSaved file with per-sample BERTScore to:\n  {output_csv_path}")
