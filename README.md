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
│   ├── XunZi-R/              # Reasoning module (adapter + configs)
│   └── XunZi-L/              # Multi-omics learning module
├── demo_data/                # Example input files (need manual download for graph_data.pth)
├── demo_xunzi_r.py           # Quickstart demo for XunZi-R only
├── demo_xunzi_l.py           # Quickstart demo for XunZi-L only
├── demo_xunzi.py             # End-to-end demo (L + R)
├── requirements.txt
└── README.md
```

---

## 💻 1. System Requirements

| Component     | Requirement                              |
|---------------|-------------------------------------------|
| Language      | Python 3.8+                               |
| Dependencies  | `transformers`, `torch`, `torch-geometric`, `pandas` |
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

---

## 📥 3. Prepare Demo Data

The demo requires a preprocessed graph file `graph_data.pth`.  
Due to size limits, this file is not included in the repository.  

👉 Please download it from HuggingFace and place it into the `demo_data/` folder:  
[HuggingFace Dataset: H2dddhxh/XunZi](https://huggingface.co/datasets/H2dddhxh/XunZi/tree/main)

```bash
# Example (requires huggingface-cli)
huggingface-cli download H2dddhxh/XunZi graph_data.pth --local-dir ./demo_data
```

---

## 🧠 4. XunZi-R Model Hub

The mechanistic reasoning engine is built on top of the **BioMistral-7B** series and is hosted at HuggingFace:

| Model          | Description                               | Link |
|----------------|-------------------------------------------|------|
| XunZi-R-BioPre | Pretrained on 24M biomedical abstracts    | [HuggingFace 🔗](https://huggingface.co/H2dddhxh/XunZi-R-BioPre) |
| XunZi-R        | Fine-tuned for mechanistic reasoning      | [HuggingFace 🔗](https://huggingface.co/H2dddhxh/XunZi-R) |

⚠️ **Important:**  
To use **XunZi-R**, you must also download its base model **XunZi-R-BioPre**.  
After downloading, edit `models/XunZi-R/adapter_config.json` and update the field:

```json
"base_model_name_or_path": "/path/to/XunZi-R-BioPre"
```

This ensures the LoRA adapter (`XunZi-R`) can correctly load the pretrained weights.

---

## 🚀 5. Run the End-to-End Demo

Make sure that:
1. `graph_data.pth` is placed in `./demo_data/`
2. `adapter_config.json` in `models/XunZi-R/` has the correct `base_model_name_or_path` pointing to your downloaded `XunZi-R-BioPre`

Then run:

```bash
python demo_xunzi.py \
  --graph_data ./demo_data/graph_data.pth \
  --l_checkpoint ./demo_data/finetuned_model.pth \
  --model_id H2dddhxh/XunZi-R \
  --top_k 50 \
  --output_csv xunzi_end2end.csv
```

---

## 📤 6. Expected Output

The output file `xunzi_end2end.csv` will contain both:
- **XunZi-L scores** (disease relevance of candidate genes)
- **XunZi-R generated hypotheses** (mechanism-guided reasoning)

Preview in terminal:

```text
✅ XunZi end-to-end finished. Saved to xunzi_end2end.csv
 - GeneID: CHEK2, Score: 0.9821
   Hypothesis: CHEK2 may regulate LRRK2 activity through DNA damage response pathways and checkpoint kinases…
```

---

## ♻️ 7. (Optional) Reproduce Figures

To reproduce results in the manuscript:

```bash
bash reproduce_figures.sh
```

---

## 🔐 8. License

This project is licensed under the MIT License.  
See `LICENSE` for full terms.
