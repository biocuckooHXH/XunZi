# demo_xunzi.py
"""
End-to-end demo: XunZi-L (omics scoring) + XunZi-R (mechanistic reasoning)

Workflow
--------
1) Load graph_data + L checkpoint; compute XunZi_L_score for gene nodes.
2) Select top-K genes (or score >= threshold).
3) Build prompts and query XunZi-R (HF Hub or local).
4) Save a merged CSV with scores and hypotheses.

Examples
--------
From HF Hub for XunZi-R:
python demo_xunzi.py \
  --graph_data ./demo_data/graph_data.pth \
  --l_checkpoint ./demo_data/finetuned_model.pth \
  --model_id H2dddhxh/XunZi-R \
  --top_k 50 \
  --output_csv xunzi_end2end.csv

From local XunZi-R:
python demo_xunzi.py \
  --graph_data ./demo_data/graph_data.pth \
  --l_checkpoint ./demo_data/finetuned_model.pth \
  --local_dir ./models/XunZi-R \
  --score_threshold 0.85 \
  --output_csv xunzi_end2end.csv
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import GCNConv
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------- Utilities --------
def infer_dims_from_ckpt(ckpt_path: str):
    """
    Infer (input_dim, goid_input_dim) from checkpoint weights.
    Falls back to None if keys are missing.
    """
    state = torch.load(ckpt_path, map_location="cpu")
    in_dim = None
    goid_in = None
    # GCNConv keeps a Linear as 'conv1.lin.weight' in state_dict for PyG>=2.0
    if "conv1.lin.weight" in state:
        in_dim = state["conv1.lin.weight"].shape[1]
    # Our goid_dnn[0] is the first Linear: weight shape [1024, goid_input_dim]
    if "goid_dnn.0.weight" in state:
        goid_in = state["goid_dnn.0.weight"].shape[1]
    return in_dim, goid_in


def validate_graph_data(g, input_dim: int, goid_input_dim: int):
    assert hasattr(g, "x"), "graph_data missing attribute: x"
    assert hasattr(g, "edge_index"), "graph_data missing attribute: edge_index"
    assert hasattr(g, "mask"), "graph_data missing attribute: mask (bool mask for gene nodes)"
    assert hasattr(g, "geneid"), "graph_data missing attribute: geneid (IDs for gene nodes)"
    assert hasattr(g, "istj_predict"), "graph_data missing attribute: istj_predict ([N,1])"

    mask = g.mask
    assert mask.dtype == torch.bool, "graph_data.mask must be boolean"
    assert g.istj_predict.dim() == 2 and g.istj_predict.shape[1] == 1, \
        f"istj_predict should be [N,1], got {tuple(g.istj_predict.shape)}"

    x = g.x
    gene_cols = x[mask].shape[1]
    goid_cols = x[~mask].shape[1]
    assert gene_cols >= input_dim, f"Gene feature cols {gene_cols} < input_dim {input_dim}"
    assert goid_cols == goid_input_dim, f"GO feature cols {goid_cols} != goid_input_dim {goid_input_dim}"

    # Optional sanity: ensure mask has at least one True/False
    assert mask.any().item(), "mask has no True (no gene nodes)"
    assert (~mask).any().item(), "mask has no False (no GO nodes)"


# -------- XunZi-L model --------
class GCNWithAggregator_Resnet(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim,
                 goid_input_dim=4096, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.dropout = dropout

        self.conv1 = GCNConv(input_dim, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)

        self.goid_dnn = nn.Sequential(
            nn.Linear(goid_input_dim, 1024), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(1024, input_dim), nn.ReLU(), nn.Dropout(p=dropout)
        )

        self.linear_residual1 = nn.Linear(input_dim, hidden_dim1)
        self.linear_residual2 = nn.Linear(hidden_dim1, hidden_dim2)

        # fusion: [hidden_dim2] (gene nodes only) + [1] (istj) -> 2 classes
        self.fusion_fc = nn.Linear(hidden_dim2 + 1, 2)

    def forward(self, data):
        x, edge_index, mask, istj_results = data.x, data.edge_index, data.mask, data.istj_predict

        # Keep original node order. Build a full [N, input_dim] feature matrix.
        gene_x = x[mask][:, :self.input_dim]                 # [Ngene, input_dim]
        goid_x = x[~mask]                                    # [Ngo, goid_input_dim]
        goid_x_transformed = self.goid_dnn(goid_x)           # [Ngo, input_dim]

        N = x.shape[0]
        x_transformed = x.new_zeros((N, self.input_dim))
        x_transformed[mask] = gene_x
        x_transformed[~mask] = goid_x_transformed

        # GCN layers with residuals
        h1 = F.relu(self.conv1(x_transformed, edge_index))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h1 = h1 + self.linear_residual1(x_transformed)

        h2 = F.relu(self.conv2(h1, edge_index))
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        h2 = h2 + self.linear_residual2(h1)

        fusion_input = torch.cat([h2[mask], istj_results[mask]], dim=1)  # [Ngene, hidden_dim2+1]
        logits = self.fusion_fc(fusion_input)
        return logits


def parse_args():
    p = argparse.ArgumentParser(description="End-to-end XunZi demo (L + R)")

    # L inputs
    p.add_argument("--graph_data", type=str, required=True, help="Path to graph_data .pth")
    p.add_argument("--l_checkpoint", type=str, required=True, help="Path to XunZi-L checkpoint .pth")

    # Allow None so we can infer from checkpoint if not provided
    p.add_argument("--input_dim", type=int, default=112, help="Gene feature dim used by GCN (will infer from ckpt if None)")
    p.add_argument("--hidden_dim1", type=int, default=128)
    p.add_argument("--hidden_dim2", type=int, default=32)
    p.add_argument("--goid_input_dim", type=int, default=None, help="GO-node raw feature dim (will infer from ckpt if None)")
    p.add_argument("--dropout", type=float, default=0.2)

    # Selection
    p.add_argument("--top_k", type=int, default=50, help="Take top-K genes by XunZi_L_score")
    p.add_argument("--score_threshold", type=float, default=None, help="Alternatively, filter by score >= threshold")

    # R model source
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--model_id", type=str, default=None, help="HF repo id, e.g. H2dddhxh/XunZi-R")
    src.add_argument("--local_dir", type=str, default=None, help="Local dir for XunZi-R")

    # R generation params
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")

    # Prompt template for R
    p.add_argument("--system_prompt", type=str, default=(
        "You are XunZi-R, an AI biologist specializing in mechanism-guided reasoning. "
        "For each candidate gene in Parkinson's disease (PD), propose a concise, mechanistic, "
        "and testable hypothesis including pathways, regulators, and 2-3 validation experiments."
    ))
    p.add_argument("--assistant_prefix", type=str, default="Hypothesis:")
    p.add_argument("--disease_name", type=str, default="Parkinson's disease")

    # Output
    p.add_argument("--output_csv", type=str, default="xunzi_end2end.csv")

    return p.parse_args()


def build_gene_prompts(gene_ids, disease_name, system_prompt, assistant_prefix):
    prompts = []
    for gid in gene_ids:
        prompt = (
            f"{system_prompt}\n\n"
            f"Query: Candidate gene: {gid} in {disease_name}. "
            f"Summarize the most plausible mechanism and propose testable experiments.\n"
            f"{assistant_prefix} "
        )
        prompts.append(prompt)
    return prompts


@torch.inference_mode()
def run_xunzi_l(args, device):
    # Infer dims from checkpoint if not provided
    inferred_in_dim, inferred_goid_in = infer_dims_from_ckpt(args.l_checkpoint)
    if args.input_dim is None:
        if inferred_in_dim is None:
            raise ValueError("Cannot infer input_dim from checkpoint; please pass --input_dim explicitly.")
        args.input_dim = inferred_in_dim
        print(f"[XunZi-L] Inferred input_dim from ckpt: {args.input_dim}")
    if args.goid_input_dim is None:
        if inferred_goid_in is None:
            raise ValueError("Cannot infer goid_input_dim from checkpoint; please pass --goid_input_dim explicitly.")
        args.goid_input_dim = inferred_goid_in
        print(f"[XunZi-L] Inferred goid_input_dim from ckpt: {args.goid_input_dim}")

    graph_data = torch.load(args.graph_data, map_location=device)
    validate_graph_data(graph_data, args.input_dim, args.goid_input_dim)
    graph_data = graph_data.to(device)

    model = GCNWithAggregator_Resnet(
        input_dim=args.input_dim,
        hidden_dim1=args.hidden_dim1,
        hidden_dim2=args.hidden_dim2,
        output_dim=2,
        goid_input_dim=args.goid_input_dim,
        dropout=args.dropout
    ).to(device)

    state = torch.load(args.l_checkpoint, map_location=device)
    # Strict load to avoid silent mismatches
    model.load_state_dict(state, strict=True)
    model.eval()

    logits = model(graph_data)  # shape [Ngene, 2]
    scores = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    gene_ids = graph_data.geneid[graph_data.mask].detach().cpu().numpy()
    df = pd.DataFrame({"GeneID": gene_ids, "XunZi_L_score": scores})
    return df.sort_values("XunZi_L_score", ascending=False).reset_index(drop=True)


@torch.inference_mode()
def run_xunzi_r(args, texts):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model_id_or_dir = args.model_id or args.local_dir
    tokenizer = AutoTokenizer.from_pretrained(
        model_id_or_dir, trust_remote_code=args.trust_remote_code, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=args.trust_remote_code
    )

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    outputs = model.generate(**inputs, **gen_kwargs)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    cleaned = []
    marker = "Hypothesis:"
    for t in decoded:
        idx = t.rfind(marker)
        s = t[idx + len(marker):].strip() if idx != -1 else t.strip()
        cleaned.append(s)
    return cleaned


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: XunZi-L scoring
    l_df = run_xunzi_l(args, device)
    if args.score_threshold is not None:
        sel_df = l_df[l_df["XunZi_L_score"] >= args.score_threshold].copy()
    else:
        sel_df = l_df.head(args.top_k).copy()

    print(f"[XunZi] Selected {len(sel_df)} genes for reasoning.")

    # Step 2: build prompts for XunZi-R
    prompts = build_gene_prompts(
        sel_df["GeneID"].astype(str).tolist(),
        disease_name=args.disease_name,
        system_prompt=args.system_prompt,
        assistant_prefix=args.assistant_prefix
    )

    # Step 3: run XunZi-R
    responses = run_xunzi_r(args, prompts)

    # Step 4: merge and save
    sel_df["XunZi_R_hypothesis"] = responses
    sel_df.to_csv(args.output_csv, index=False)

    print(f"✅ XunZi end-to-end finished. Saved to {args.output_csv}")
    print("   Preview:")
    for _, row in sel_df.head(3).iterrows():
        print(f" - GeneID: {row['GeneID']}, Score: {row['XunZi_L_score']:.4f}")
        print(f"   Hypothesis: {row['XunZi_R_hypothesis'][:120]}…")


if __name__ == "__main__":
    main()

