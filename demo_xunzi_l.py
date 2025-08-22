# demo_xunzi_l_infer.py
import torch
import torch.nn.functional as F
import pandas as pd
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCNWithAggregator_Resnet(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, goid_input_dim=4096, dropout=0.2):
        super(GCNWithAggregator_Resnet, self).__init__()
        self.input_dim = input_dim
        self.dropout = dropout  # Dropout rate

        # GCN Layers
        self.conv1 = GCNConv(input_dim, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)

        # GOID DNN
        self.goid_dnn = nn.Sequential(
            nn.Linear(goid_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(1024, self.input_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        # Linear for residuals
        self.linear_residual1 = nn.Linear(input_dim, hidden_dim1)
        self.linear_residual2 = nn.Linear(hidden_dim1, hidden_dim2)

        # ✨ Fusion layer (final classification using omics + istj)
        self.fusion_fc = nn.Linear(hidden_dim2 + 1, output_dim)


    def forward(self, data):
        x, edge_index, mask, istj_results = data.x, data.edge_index, data.mask, data.istj_predict

        # Split gene nodes and GOID nodes
        gene_x = x[mask][:, :self.input_dim]  # Assuming the first input_dim features are gene-related
        goid_x = x[~mask]  # GOID nodes' features
        istj_x = istj_results

        # Transform GOID features
        goid_x_transformed = self.goid_dnn(goid_x)

        # Combine all input features
        x_transformed = torch.cat([gene_x, goid_x_transformed], dim=0)

        # First GCN Layer with Residual Connection
        x_hidden_dim1 = F.relu(self.conv1(x_transformed, edge_index))
        x_hidden_dim1_residual = self.linear_residual1(x_transformed)
        x_hidden_dim1 = F.dropout(x_hidden_dim1, p=self.dropout, training=self.training)
        x_hidden_dim1 = x_hidden_dim1 + x_hidden_dim1_residual

        # Second GCN Layer with Residual Connection
        x_hidden_dim2 = F.relu(self.conv2(x_hidden_dim1, edge_index))
        x_hidden_dim2_residual = self.linear_residual2(x_hidden_dim1)
        x_hidden_dim2 = F.dropout(x_hidden_dim2, p=self.dropout, training=self.training)
        x_hidden_dim2 = x_hidden_dim2 + x_hidden_dim2_residual

        # Only use gene node features for final classification
        gene_hidden_features = x_hidden_dim2
        # print(f"gene_hidden_features shape :{gene_hidden_features.shape}")
        # print(f"istj_x shape :{istj_x.shape}")
        # Concatenate istj score
        fusion_input = torch.cat([gene_hidden_features, istj_x], dim=1)  # shape: [num_gene_nodes, hidden_dim2 + 1]

        final_results = self.fusion_fc(fusion_input)  # shape: [num_gene_nodes, output_dim]

        return final_results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load data
graph_data = torch.load("./demo_data/graph_data.pth").to(device)

# 2. Initialize model & load weights
model = GCNWithAggregator_Resnet(input_dim=64, hidden_dim1=128, hidden_dim2=32, output_dim=2).to(device)
model.load_state_dict(torch.load("./demo_data/finetuned_model.pth", map_location=device))
model.eval()

# 3. Inference
with torch.no_grad():
    _, logits = model(graph_data)
    scores = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

# 4. Save results
gene_ids = graph_data.geneid[graph_data.mask].cpu().numpy()
result_df = pd.DataFrame({"GeneID": gene_ids, "XunZi_l_score": scores})
result_df.to_csv("demo_prediction.csv", index=False)

print("✅ XunZi-L inference finished. Results saved to demo_prediction.csv")

