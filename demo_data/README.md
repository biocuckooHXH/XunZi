# Demo Data for XunZi End-to-End

This folder contains the example input files used by `demo_xunzi.py`, which demonstrates the end-to-end pipeline of **XunZi-L** (omics scoring) and **XunZi-R** (mechanistic reasoning).

## Files

- **`graph_data.pth`**  
  Pre-processed graph structure containing **gene nodes**, **GO term nodes**, features, and edges.  
  ⚠️ **Note:** This file will not be hosted on GitHub due to size limits.  
  Please download it from HuggingFace:  
  👉 [HuggingFace Dataset: H2dddhxh/XunZi](https://huggingface.co/datasets/H2dddhxh/XunZi/tree/main)

- **`finetuned_model.pth`**  
  Checkpoint for **XunZi-L** (the GCN-based multi-omics scoring model).  
  Used to compute disease-relevance scores (`XunZi_L_score`) for gene nodes.

- **`PD_finetune_Kinase_function_model_full.pth`**  
  Specialized checkpoint of **XunZi-L** finetuned for **kinase functional prediction in Parkinson’s disease (PD)**.  
  Can be swapped in place of `finetuned_model.pth` for kinase-focused demos.

- **`XunZi_r_prompt.txt`**  
  Example system/assistant prompts for **XunZi-R**.  
  Used to guide the mechanistic reasoning LLM to generate concise, testable hypotheses.

- **`README.md`**  
  This file — explains the purpose of each data file.

## Example Usage

```bash
# Run with HuggingFace-hosted XunZi-R
python demo_xunzi.py \
  --graph_data ./demo_data/graph_data.pth \
  --l_checkpoint ./demo_data/finetuned_model.pth \
  --model_id H2dddhxh/XunZi-R \
  --top_k 50 \
  --output_csv xunzi_end2end.csv

# Run with local XunZi-R
python demo_xunzi.py \
  --graph_data ./demo_data/graph_data.pth \
  --l_checkpoint ./demo_data/finetuned_model.pth \
  --local_dir ./models/XunZi-R \
  --score_threshold 0.85 \
  --output_csv xunzi_end2end.csv
