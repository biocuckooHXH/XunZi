import pandas as pd
from bert_score import score as bert_score
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import torch

# =====================================================
# 1. 配置区 —— 你只需要改文件路径即可
# =====================================================
gpt_csv   = "./Merge_for_gene_function_Start_GPT_format.csv"
xunzi_csv = "./Merge_for_gene_function_Start_GPT_all_XunZi_R.csv"

# 参考解释列名为 Mechanism_answer
ref_col = "Mechanism_answer"

# 本地 SciBERT 模型路径
bert_model_path = "/work/home/acssag9hf5/XunZi/XunZI-R/Metric/Models/scibert_scivocab_uncased"

# 输出文件
save_path = "text_eval_bertscore_bleu_summary.csv"

# =====================================================
# 2. 选择设备：优先用 GPU
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(">>> BERTScore will run on device:", device)

# =====================================================
# 3. 读取两个模型的数据表
# =====================================================
GPT_o3 = pd.read_csv(gpt_csv, sep=",")
XunZi_R = pd.read_csv(xunzi_csv, sep=",")

print("GPT_o3 columns:", GPT_o3.columns.tolist())
print("XunZi_R columns:", XunZi_R.columns.tolist())

# =====================================================
# 4. 合并两个表，保证同一行是同一个 id
# =====================================================
merged = (
    GPT_o3[['id', 'Mechanism_answer', 'content']]
    .merge(
        XunZi_R[['id', 'response']],
        on='id',
        how='inner'
    )
)

print("Merged shape:", merged.shape)

# =====================================================
# 5. 文本截断函数：防止超过 BERT 的 512 token 限制
# =====================================================
def truncate_words(text, max_words=256):
    if not isinstance(text, str):
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])

# 对参考解释和两个模型输出都做截断，保证公平
merged["Mechanism_answer_trunc"] = merged["Mechanism_answer"].fillna("").astype(str).apply(truncate_words)
merged["content_trunc"]          = merged["content"].fillna("").astype(str).apply(truncate_words)
merged["response_trunc"]         = merged["response"].fillna("").astype(str).apply(truncate_words)

# =====================================================
# 6. 准备 reference（人工机制解释）和 模型输出（截断版）
# =====================================================
refs = merged["Mechanism_answer_trunc"].tolist()

model_texts = {
    "GPT_o3": merged["content_trunc"].tolist(),
    "XunZi_R": merged["response_trunc"].tolist(),
}

results = []

# NLTK BLEU 需要 tokenized 的文本
def to_bleu_format(hyps, refs):
    """
    hyps: list[str]
    refs: list[str]
    返回:
      hyp_tokens: list[list[str]]
      ref_tokens: list[list[list[str]]]  # corpus_bleu 的格式：每个样本可以有多个参考，这里只有1个
    """
    hyp_tokens = [h.split() for h in hyps]
    ref_tokens = [[r.split()] for r in refs]
    return hyp_tokens, ref_tokens

smooth = SmoothingFunction().method1  # 避免句子太短 BLEU=0

# =====================================================
# 7. 对每个模型计算：BERTScore + BLEU
# =====================================================
for model_name, hyps in model_texts.items():
    print(f"\n==================== {model_name} ====================")

    # ----------- BERTScore（本地 SciBERT）-----------
    print("Computing BERTScore...")
    P, R, F1 = bert_score(
        hyps,
        refs,
        lang="en",
        model_type=bert_model_path,   # 本地 SciBERT 路径
        num_layers=12,                # SciBERT/BERT-base 是 12 层
        rescale_with_baseline=False,  # 不用 baseline，保持原始分数
        batch_size=128,                # 显存吃紧可以减小
        device=device
    )

    bert_P_mean = float(P.mean())
    bert_R_mean = float(R.mean())
    bert_F1_mean = float(F1.mean())

    print(f"BERTScore P:  {bert_P_mean:.4f}")
    print(f"BERTScore R:  {bert_R_mean:.4f}")
    print(f"BERTScore F1:{bert_F1_mean:.4f}")

    # ----------- BLEU（nltk 版）-----------
    print("Computing BLEU (nltk.corpus_bleu)...")
    hyp_tokens, ref_tokens = to_bleu_format(hyps, refs)
    bleu = corpus_bleu(
        ref_tokens,
        hyp_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),  # 标准 BLEU-4
        smoothing_function=smooth
    )
    bleu_score = bleu * 100.0  # 转换到 0–100 区间

    print(f"BLEU score: {bleu_score:.2f}")

    results.append({
        "model": model_name,
        "BERTScore_P": bert_P_mean,
        "BERTScore_R": bert_R_mean,
        "BERTScore_F1": bert_F1_mean,
        "BLEU_corpus": bleu_score
    })

# =====================================================
# 8. 保存结果
# =====================================================
res_df = pd.DataFrame(results)
print("\n======== Summary ========")
print(res_df)

res_df.to_csv(save_path, index=False)
print(f"\nSaved results to {save_path}")
