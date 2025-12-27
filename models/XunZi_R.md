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
