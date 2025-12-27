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
│   │   └── XunZi_modules/           # XunZi-M and XunZi-R implementations
│   ├── training/
│   │   ├── trainer.py               # Distributed training orchestration
│   │   └── train.py                 # Main training script
│   └── evaluation/
│       ├── evaluator.py            # Comprehensive evaluation framework
│       ├── metrics.py               # Biomedical-specific metrics
│       └── inference.py             # Inference engine for deployment
├── scripts/
│   ├── preprocess_data.py          # Data preparation scripts
│   ├── train.py                     # Training launcher
│   └── evaluate/
│       ├── evaluate_model.py       # Model evaluation
│       └── benchmark.py             # Performance benchmarking
├── configs/
│   ├── training_config.yaml        # Training hyperparameters
│   └── eval_config.yaml            # Evaluation settings
├── models/
│   ├── XunZi-R/                    # Reasoning module checkpoints
│   └── XunZi-M/                    # Multi-omics module checkpoints
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
```

### 3. Run Demo
```bash
# Quick demo with pre-trained models
python demo_xunzi.py \
  --graph_data ./demo_data/graph_data.pth \
  --model_id ./models/XunZi-R \
  --output_csv results/xunzi_predictions.csv
```

## 🔧 Full Pipeline

### Data Preprocessing

Process raw biomedical literature (240,000 articles and gene-disease annotations):
```bash
python scripts/preprocess_data.py \
  --input-path /path/to/raw/articles.json \
  --output-dir data/processed \
  --tokenizer mistralai/Mistral-7B-Instruct-v0.1 \
  --task cpt \
  --max-length 2048 \
  --val-ratio 0.05
```

### Continual Pre-training

Train XunZi-R on biomedical corpus using LoRA:
```bash
python scripts/train.py \
  --config configs/training_config.yaml \
  --use-wandb
```


## 🔬 XunZi Modules

### XunZi-M: Multi-Omics Learning
- Graph Convolutional Networks for gene-gene interactions
- Integration of 600TB+ multi-omics data
- Cross-validation with DisGeNET and CTD databases

### XunZi-R: Mechanistic Reasoning
- Fine-tuned Mistral-7B on 24M publications
- Specialized for biomedical entity recognition
- Generates mechanistic hypotheses for gene-disease associations


## 📝 Citation

If you use XunZi in your research, please cite:
```bibtex
@article{huang2024xunzi,
  title={XunZi, a AI biologist, reveals novel disease-modifying targets},
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
