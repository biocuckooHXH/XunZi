# 🧠 XunZi: An AI Biologist for Mechanism-Guided Therapeutic Target Discovery

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

## 🔍 Overview

XunZi is an AI-driven framework that bridges mechanistic insights from biomedical literature with multi-omics data for therapeutic target discovery. It consists of two synergistic modules:

- **XunZi-M**: Multi-omics learning module integrating >600TB of biological data
- **XunZi-R**: Mechanistic reasoning engine based on Mistral-7B, trained on 24M biomedical publications

## 📊 Key Features

- **Continual Pre-training**: Custom biomedical language model trained on 240,000+ curated publications
- **Multi-modal Integration**: Combines graph neural networks with transformer-based reasoning
- **Scalable Architecture**: Distributed training support with LoRA for efficient fine-tuning
- **Comprehensive Evaluation**: Multiple metrics including perplexity, ROUGE, BLEU, and entity extraction F1

## 🏗️ Repository Structure
```
XunZi/
├── src/
│   ├── data/
│   │   ├── dataset.py              # Dataset classes for biomedical text
│   │   ├── preprocessor.py         # Text preprocessing for gene-disease annotations
│   │   └── preprocess_data.py      # Main data preprocessing pipeline
│   ├── model/
│   │   ├── mistral_wrapper.py      # Mistral-7B model wrapper with LoRA
│   │   ├── XunZi_finetune.py       # XunZi fine-tuning implementation
│   │   ├── XunZi_full_train.py     # Full XunZi training (M+R modules)
│   │   └── XunZi_M_train.py        # XunZi-M multi-omics training
│   ├── training/
│   │   ├── trainer.py              # Distributed training orchestration
│   │   └── train.py                # Main training scriptr deployment
├── plots/                           # Draw the figures
├── dataset/                         # Supplementary data
├── evaluation/
│   ├── CTD_for_Disgenet_metric.py  # CTD database evaluation metrics
│   ├── Disgenet_for_CTD_metric.py  # DisGeNET cross-validation metrics
│   ├── XunZi_mechanism_BLEU_BertScore.py    # BLEU & BERTScore for mechanism generation
│   ├── XunZi_mechanism_BLEU_Rouge_single_csv.py  # BLEU & ROUGE metrics for single outputs
│   └── XunZi_mechanism_Bertscore_single_csv.py   # BERTScore for single outputs
├── models/
│   ├── XunZi-R                     # Reasoning module checkpoints
│   └── XunZi-L/                    # Multi-omics module checkpoints
├── demo_data/                       # Example datasets
├── demo_xunzi.py                    # End-to-end demonstration
├── demo_xunzi_r.py                  # XunZi-R standalone demo
├── demo_xunzi_l.py                  # XunZi-M standalone demo
├── requirements.txt
└── README.md
```

## 💻 System Requirements

| Component | Requirement |
|-----------|------------|
| **Python** | 3.8+ |
| **PyTorch** | 2.0+ |
| **CUDA** | 11.8+ (for GPU) |
| **RAM** | 64GB+ |
| **GPU** | 24GB+ VRAM (RTX 3090/4090, A100) |
| **Storage** | 100GB+ for full dataset |

## 🧠 XunZi-R Model Hub

The mechanistic reasoning engine is built on top of the BioMistral-7B series and is hosted at HuggingFace:

| Model | Description | Link |
|-------|------------|------|
| **XunZi-R-BioPre** | Pretrained on 24M biomedical abstracts | [🤗 HuggingFace](https://huggingface.co/H2dddhxh/XunZi-R-BioPre) |
| **XunZi-R** | Fine-tuned for mechanistic reasoning | [🤗 HuggingFace](https://huggingface.co/H2dddhxh/XunZi-R) |

⚠️ **Important**: To use XunZi-R, you must also download its base model XunZi-R-BioPre. After downloading, edit `models/XunZi-R/adapter_config.json` and update the field:
```json
"base_model_name_or_path": "/path/to/XunZi-R-BioPre"
```

This ensures the LoRA adapter (`XunZi-R`) can correctly load the pretrained weights.

## 🚀 Quick Start

### 1. Installation
```bash
# Clone repository
git clone https://github.com/biocuckooHXH/XunZi.git
cd XunZi

# Create conda environment
conda create -n xunzi python=3.9
conda activate xunzi

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Pre-trained Models
```bash
# Download XunZi-R base model from HuggingFace
huggingface-cli download H2dddhxh/XunZi-R-BioPre --local-dir ./models/XunZi-R-BioPre

# Download XunZi-R adapter
huggingface-cli download H2dddhxh/XunZi-R --local-dir ./models/XunZi-R

# Download demo data
huggingface-cli download H2dddhxh/XunZi graph_data.pth --local-dir ./demo_data

# Update adapter config
# Edit models/XunZi-R/adapter_config.json to point to XunZi-R-BioPre
```

### 3. Run Demo
```bash
# Quick demo with pre-trained models
python demo_xunzi.py \
  --graph_data ./demo_data/graph_data.pth \
  --model_id ./models/XunZi-R \
  --output_csv results/xunzi_predictions.csv
```

## 🔧 Training Pipeline

### Stage 1: Continual Pre-training on Biomedical Literature

Pre-train on 240,000 PubMed abstracts to adapt Mistral-7B to biomedical domain:
```bash
# Preprocess PubMed abstracts for continual pre-training
python scr/preprocess_data.py \
  --input-path /path/to/pubmed_abstracts.json \
  --output-dir data/cpt_processed \
  --tokenizer mistralai/Mistral-7B-Instruct-v0.1 \
  --task cpt \
  --max-length 2048 \
  --val-ratio 0.05

# Run continual pre-training
python scr/train.py \
  --config configs/cpt_config.yaml \
  --use-wandb
```

Input format for pre-training (PubMed abstracts):
```json
{
  "pmid": "12345678",
  "title": "Gene expression patterns in breast cancer",
  "abstract": "Full text of the abstract..."
}
```

### Stage 2: Fine-tuning for Mechanistic Reasoning

Fine-tune on curated gene-disease QA pairs with mechanistic explanations:
```bash
# Prepare instruction-tuning dataset
python scr/preprocess_data.py \
  --input-path /path/to/gene_disease_qa.json \
  --output-dir data/ft_processed \
  --tokenizer mistralai/Mistral-7B-Instruct-v0.1 \
  --task it \
  --max-length 2048

# Fine-tune for mechanistic reasoning
python scr/train.py \
  --config configs/finetune_config.yaml \
  --checkpoint outputs/cpt_model/best_model \
  --use-wandb
```

Input format for fine-tuning (QA pairs):
```json
{
  "question": "Is Gene CXCL10 (CXCL10) involved in disease Cystitides, Interstitial in a functional way?",
  "answer": "Yes. CXCL10, a chemokine, plays a significant role in Cystitis and Interstitial Cystitis by attracting Th1 cells, mast cells, NK cells, and NKT cells to the site of inflammation...\nImpacted Genes: CXCL9, CXCL11, IFN-gamma...\nImpacted Pathways: Chemokine signaling pathway, Th1 cell signaling pathway..."
}
```

### Stage 3: Multi-Omics Integration (XunZi-M)

Train graph neural networks on multi-omics data:
```bash
# Train XunZi-M module
python src/model/XunZi_M_train.py \
  --graph_data ./demo_data/graph_data.pth \
  --label_data ./demo_data/disease_labels.csv \
  --epochs 100 \
  --k_folds 5
```

### Stage 4: End-to-End Integration

Combine XunZi-M and XunZi-R for final predictions:
```bash
# Full XunZi training (M+R integration)
python src/model/XunZi_full_train.py \
  --graph_data ./demo_data/graph_data.pth \
  --reasoning_model outputs/ft_model/best_model \
  --llm_data ./demo_data/llm_predictions.csv \
  --output_dir outputs/xunzi_full
```


## 🔬 XunZi Modules

### XunZi-M: Multi-Omics Learning
- Graph Convolutional Networks for gene-gene interactions
- Integration of 600TB+ multi-omics data
- Cross-validation with DisGeNET and CTD databases
- 5-fold stratified cross-validation

### XunZi-R: Mechanistic Reasoning
- Fine-tuned Mistral-7B on 24M publications
- Specialized for biomedical entity recognition
- Generates mechanistic hypotheses for gene-disease associations
- Context-aware gene-disease relationship extraction


## 📝 Citation

If you use XunZi in your research, please cite:
```bibtex
@article{huang2024xunzi,
  title={XunZi, an AI biologist, reveals novel disease-modifying targets },
  author={Huang, Xinhe et al.},
  journal={bioRxiv},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Mistral AI for the base language model
- DisGeNET and CTD for validation databases
- The biomedical research community for open datasets

## 📧 Contact

For questions and support:
- Open an issue on GitHub
- Email: huangxinhe@hust.edu.cn

