import os
import gc
import numpy as np
import pandas as pd

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from transformers import AutoTokenizer


# =====================================================
# 1. 配置 —— 只需要修改这两行
# =====================================================
data_dir = "/work/home/acssag9hf5/XunZi/XunZI-R/Metric/XunZI_independent_dataset"      # 文件夹路径
input_csv_name = "XunZiR_train_on_Disgenet.csv"                         # 要评估的文件名

ref_col = "Mechanism_answer"   # 参考文本列
resp_col = "response"          # 模型输出列

bert_model_path = "/work/home/acssag9hf5/XunZi/XunZI-R/Metric/Models/scibert_scivocab_uncased"
MAX_TOKENS = 400   # 截断长度（subword 数）


# =====================================================
# 2. 自动构造输入 / 输出路径
# =====================================================
input_csv_path = os.path.join(data_dir, input_csv_name)
output_csv_path = os.path.join(
    data_dir,
    input_csv_name.replace(".csv", "_with_bleu_rouge.csv")
)

print(">>> Input file:", input_csv_path)
print(">>> Output file:", output_csv_path)


# =====================================================
# 3. 加载 tokenizer（用于截断）
# =====================================================
try:
    tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
    print("Loaded tokenizer from local model.")
except Exception:
    print("Failed to load local tokenizer. Falling back to SciBERT online.")
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")


def truncate_with_tokenizer(text, tokenizer, max_tokens=400):
    if not isinstance(text, str) or not text.strip():
        return ""
    try:
        tokens = tokenizer.tokenize(text)
        tokens = tokens[:max_tokens]
        return tokenizer.convert_tokens_to_string(tokens).strip()
    except:
        # fallback
        return " ".join(text.split()[: int(max_tokens/2)])


def simple_tokenize(text):
    if not isinstance(text, str):
        return []
    return text.strip().split()


# =====================================================
# 4. 读取 CSV 文件
# =====================================================
df = pd.read_csv(input_csv_path)
print("Columns:", df.columns.tolist())

if ref_col not in df.columns:
    raise KeyError(f"Missing reference column '{ref_col}'")
if resp_col not in df.columns:
    raise KeyError(f"Missing response column '{resp_col}'")

df[ref_col] = df[ref_col].fillna("").astype(str)
df[resp_col] = df[resp_col].fillna("").astype(str)


# =====================================================
# 5. 截断文本
# =====================================================
df["ref_trunc"] = df[ref_col].apply(lambda x: truncate_with_tokenizer(x, tokenizer, MAX_TOKENS))
df["hyp_trunc"] = df[resp_col].apply(lambda x: truncate_with_tokenizer(x, tokenizer, MAX_TOKENS))


# =====================================================
# 6. 配置 BLEU / ROUGE 计算器
# =====================================================
smoothing_fn = SmoothingFunction().method1
rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def compute_bleu(hyp, ref):
    hyp_tokens = simple_tokenize(hyp)
    ref_tokens = simple_tokenize(ref)
    if not hyp_tokens or not ref_tokens:
        return 0.0
    return float(
        sentence_bleu(
            [ref_tokens],
            hyp_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing_fn
        )
    )


def compute_rouge(hyp, ref):
    if not isinstance(hyp, str) or not isinstance(ref, str):
        return 0.0, 0.0, 0.0
    if hyp.strip() == "" or ref.strip() == "":
        return 0.0, 0.0, 0.0
    scores = rouge.score(ref, hyp)
    return (
        float(scores["rouge1"].fmeasure),
        float(scores["rouge2"].fmeasure),
        float(scores["rougeL"].fmeasure)
    )


# =====================================================
# 7. 逐行计算 BLEU + ROUGE
# =====================================================
bleu_scores = []
rouge1_scores = []
rouge2_scores = []
rougel_scores = []

print("\n>>> Computing BLEU + ROUGE per sample...")
for hyp, ref in zip(df["hyp_trunc"], df["ref_trunc"]):
    bleu = compute_bleu(hyp, ref)
    r1, r2, rl = compute_rouge(hyp, ref)

    bleu_scores.append(bleu)
    rouge1_scores.append(r1)
    rouge2_scores.append(r2)
    rougel_scores.append(rl)

df["BLEU"] = bleu_scores
df["ROUGE1_F"] = rouge1_scores
df["ROUGE2_F"] = rouge2_scores
df["ROUGEL_F"] = rougel_scores


# =====================================================
# 8. 保存结果
# =====================================================
df.to_csv(output_csv_path, index=False)
print("\n>>> Saved result to:", output_csv_path)
print("Done.")
