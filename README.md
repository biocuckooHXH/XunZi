# 🧠 XunZi: An AI Biologist for Mechanism-Guided Therapeutic Target Discovery

## 🔍 Introduction

Holistic hypothesis generation in biomedicine remains limited by the integration of literature-derived mechanistic insights with multi-omics data. While large language models (LLMs) have advanced textual analysis, few systems unify logical reasoning and omics learning at scale.

Here, we introduce **XunZi**, an AI biologist composed of two synergistic modules:
- **XunZi-L** (multi-omics learning)
- **XunZi-R** (mechanistic reasoning)

XunZi integrates >600 TB of multi-omics data and 24 million biomedical publications to prioritize disease-modifying genes and generate testable mechanistic hypotheses.

---

## 🧩 Repository Structure

```
XunZi/
├── models/
│   ├── XunZi-R/              # Scripts and configs for reasoning module
│   └── XunZi-L/              # Scripts and configs for multi-omics learning
├── demo_data/                # Example input files
├── demo_xunzi_r.py           # Quickstart demo for XunZi-R
├── xunzi_r_infer.py          # Batch inference script
├── requirements.txt
└── README.md
```

---

## 💻 1. System Requirements

| Component     | Requirement                              |
|---------------|-------------------------------------------|
| Language      | Python 3.8+                               |
| Dependencies  | `transformers`, `torch`, `pandas`, etc.   |
| OS            | Ubuntu 20.04 / macOS 13+ / Windows (WSL)  |
| Hardware      | GPU (>=24GB VRAM) recommended for XunZi-R |
| Tested on     | RTX 3090 / A100 / CPU (demo only)         |

> ⚠️ Note: The reasoning model `XunZi-R` is based on Mistral-7B. GPU with at least 24GB VRAM is recommended for inference.

---

## ⚙️ 2. Installation

```bash
git clone https://github.com/biocuckooHXH/XunZi.git
cd XunZi

conda create -n xunzi python=3.9
conda activate xunzi
pip install -r requirements.txt
```

Estimated setup time: ~5 minutes

---

## 🚀 3. Demo: XunZi-R Mechanistic Reasoning

### ▶️ Run the demo:

```bash
python demo_xunzi_r.py --query_file demo_data/sample_gene_disease.csv
```

### 📤 Expected Output

```text
[✔] Gene: CHEK2, Disease: Parkinson's disease
↪ Mechanism: CHEK2 modulates mitochondrial apoptosis and DNA damage response.
↪ Score: 0.94
```

### ⏱️ Runtime (per query)
| Hardware | Time |
|----------|------|
| CPU      | ~20s |
| GPU (3090/A100) | ~5s |

---

## 🧠 4. XunZi-R Model Hub

The mechanistic reasoning engine is built on top of the **BioMistral-7B** series and is hosted at HuggingFace:

| Model        | Description                             | Link |
|--------------|-----------------------------------------|------|
| XunZi-R-BioPre | Pretraining on 24M biomedical abstracts | [HuggingFace 🔗](https://huggingface.co/H2dddhxh/XunZi-R-BioPre) |
| XunZi-R       | Fine-tuned for mechanistic reasoning     | [HuggingFace 🔗](https://huggingface.co/H2dddhxh/XunZi-R) |

> These models power logical reasoning across 21,000 genes and 5,800 diseases.

---

## 📚 5. Instructions for Use (Custom Inference)

### Input Format

```csv
gene,disease
CHEK2,Parkinson's disease
TP53,NSCLC
...
```

### Run:

```bash
python xunzi_r_infer.py --query_file your_input.csv --output predictions.csv
```

---

## ♻️ 6. (Optional) Reproduce Figures

To reproduce results in the manuscript:

```bash
bash reproduce_figures.sh
```

---

## 🔐 7. License

This project is licensed under the MIT License.  
See `LICENSE` for full terms.
