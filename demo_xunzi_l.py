# demo_xunzi_l_infer.py
"""
Quickstart demo for XunZi-L (multi-omics learning module).

Features
- Load a pre-trained GNN checkpoint.
- Run inference on a graph_data .pth and produce per-gene scores.
- Save outputs to a CSV.

Usage
------
python demo_xunzi_l_infer.py \
  --graph_data ./demo_data/graph_data.pth \
  --checkpoint ./demo_data/finetuned_model.pth \
  --input_dim 64 --hidden_dim1 128 --hidden_dim2 32 --output_dim 2 \
  --output_csv demo_prediction.csv
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import GCNConv


class GCNWithAggregator_Resnet(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, goid_input_dim=4096, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.dropout = dropout

        # GCN layers
        self.conv1 = GCNConv(input_dim, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)

        # DNN for GOID features
        self.goid_dnn = nn.Sequential(
            nn.Linear(goid_input_dim, 1024), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(1024, self.input_dim), nn.ReLU(), nn.Dropout(p=dropout)
        )

        # Residual projections
        self.linear_residual1 = nn.Linear(input_dim, hidden_dim1)
        self.linear_residual2 = nn.Linear(hidden_dim1, hidden_dim2)

        # Fusion (omics + istj)
        self.fusion_fc = nn.Linear(hidden_dim2 + 1, output_dim)

    def forward(self, data):
        x, edge_index, mask, istj_results = data.x, data.edge_index, data.mask, data.istj_predict
        gene_x = x[mask][:, :self.input_dim]
        goid_x = x[~mask]
        goid_x_transformed = self.goid_dnn(goid_x)
        x_transformed = torch.cat([gene_x, goid_x_transformed], dim=0)

        h1 = F.relu(self.conv1(x_transformed, edge_index))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h1 = h1 + self.linear_residual1(x_transformed)

        h2 = F.relu(self.conv2(h1, edge_index))
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        h2 = h2 + self.linear_residual2(h1)

        # only gene nodes go into classifier; concatenate istj score
        fusion_input = torch.cat([h2[mask], istj_results[mask]], dim=1)
        logits = self.fusion_fc(fusion_input)
        return logits


def parse_args():
    p = argparse.ArgumentParser(description="XunZi-L demo inference")
    p.add_argument("--graph_data", type=str, required=True, help="Path to graph_data .pth")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to finetuned model checkpoint (.pth)")
    p.add_argument("--input_dim", type=int, default=64)
    p.add_argument("--hidden_dim1", type=int, default=128)
    p.add_argument("--hidden_dim2", type=int, default=32)
    p.add_argument("--output_dim", type=int, default=2)
    p.add_argument("--goid_input_dim", type=int, default=4096)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--output_csv", type=str, default="demo_prediction.csv")
    return p.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    graph_data = torch.load(args.graph_data, map_location=device)
    graph_data = graph_data.to(device)

    # init model & load weights
    model = GCNWithAggregator_Resnet(
        input_dim=args.input_dim,
        hidden_dim1=args.hidden_dim1,
        hidden_dim2=args.hidden_dim2,
        output_dim=args.output_dim,
        goid_input_dim=args.goid_input_dim,
        dropout=args.dropout
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # inference
    logits = model(graph_data)
    scores = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

    # save
    gene_ids = graph_data.geneid[graph_data.mask].detach().cpu().numpy()
    out_df = pd.DataFrame({"GeneID": gene_ids, "XunZi_L_score": scores})
    out_df.to_csv(args.output_csv, index=False)

    print(f"✅ XunZi-L inference finished. Saved to {args.output_csv}")
    print(f"   Scored genes: {len(out_df)}")


if __name__ == "__main__":
    main()
