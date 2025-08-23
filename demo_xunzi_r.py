# demo_xunzi_r.py
"""
Quickstart demo for XunZi-R (mechanistic reasoning module).

Features
- Load model from Hugging Face Hub (e.g. H2dddhxh/XunZi-R) or a local folder.
- Read a CSV (default column: 'input') and run batched generation.
- Save outputs to a CSV with 'response' column.

Usage
------
1) From Hugging Face Hub:
   python demo_xunzi_r.py --model_id H2dddhxh/XunZi-R --input_csv demo_data/xunzi_r_inputs.csv

2) From local folder (already downloaded):
   python demo_xunzi_r.py --local_dir ./models/XunZi-R --input_csv demo_data/xunzi_r_inputs.csv

Optional args:
   --text_column input --output_csv xunzi_r_predictions.csv
   --batch_size 4 --max_new_tokens 256 --temperature 0.2 --top_p 0.9

CSV format
----------
input
"Summarize PD kinase mechanisms linking CHK2 to LRRK2."
"Propose testable hypotheses for IRAK4 in neuroinflammation."
...
"""

import os
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer


def parse_args():
    parser = argparse.ArgumentParser(description="XunZi-R demo inference")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--model_id", type=str, default=None,
                     help="HF repo id, e.g. H2dddhxh/XunZi-R")
    src.add_argument("--local_dir", type=str, default=None,
                     help="Local model folder containing tokenizer + weights")

    parser.add_argument("--input_csv", type=str, required=True,
                        help="Path to input CSV")
    parser.add_argument("--text_column", type=str, default="input",
                        help="Column name containing prompts")
    parser.add_argument("--output_csv", type=str, default="xunzi_r_predictions.csv",
                        help="Path to save outputs")

    # Inference params
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true",
                        help="Enable sampling (default: False → greedy)")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="Pass through to transformers loaders if model uses custom code")

    # Prompt template (simple, can be customized)
    parser.add_argument("--system_prompt", type=str, default=(
        "You are XunZi-R, an AI biologist specializing in mechanism-guided reasoning. "
        "Given a biomedical query, provide a concise, mechanistic, and testable hypothesis "
        "including key pathways, regulators, and suggested validation experiments."
    ))
    parser.add_argument("--user_prefix", type=str, default="Query:")
    parser.add_argument("--assistant_prefix", type=str, default="Hypothesis:")

    return parser.parse_args()


def build_prompts(texts, system_prompt, user_prefix, assistant_prefix):
    """
    Very light template to keep things model-agnostic.
    If your chat template requires special tokens, adjust here.
    """
    prompts = []
    for t in texts:
        t = str(t).strip()
        prompt = (
            f"{system_prompt}\n\n"
            f"{user_prefix} {t}\n"
            f"{assistant_prefix} "
        )
        prompts.append(prompt)
    return prompts


@torch.inference_mode()
def generate_batches(model, tokenizer, prompts, device, batch_size=4,
                     max_new_tokens=512, temperature=0.2, top_p=0.9, do_sample=False):
    outputs = []
    # Optional live streamer for console preview (disabled by default)
    # streamer = TextStreamer(tokenizer, skip_special_tokens=True)

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = model.generate(**inputs, **gen_kwargs)  # , streamer=streamer
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

        # Extract only the assistant part after the assistant prefix if present
        # (keeps output clean in case the model echoes the prompt)
        cleaned = []
        for full_text in decoded:
            # Heuristic: take text after the last occurrence of "Hypothesis:" if present
            marker = "Hypothesis:"
            idx = full_text.rfind(marker)
            snippet = full_text[idx+len(marker):].strip() if idx != -1 else full_text.strip()
            cleaned.append(snippet)
        outputs.extend(cleaned)
    return outputs


def main():
    args = parse_args()

    assert os.path.exists(args.input_csv), f"Input CSV not found: {args.input_csv}"
    df = pd.read_csv(args.input_csv)
    assert args.text_column in df.columns, f"Column '{args.text_column}' not in CSV."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model_source = args.model_id if args.model_id else args.local_dir
    print(f"[XunZi-R] Loading model from: {model_source}")

    # Load tokenizer & model (works for most causal LMs; adjust if your arch differs)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id or args.local_dir,
        trust_remote_code=args.trust_remote_code,
        use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id or args.local_dir,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=args.trust_remote_code
    )

    prompts = build_prompts(
        df[args.text_column].tolist(),
        args.system_prompt,
        args.user_prefix,
        args.assistant_prefix
    )

    print(f"[XunZi-R] Running inference on {len(prompts)} rows "
          f"(batch_size={args.batch_size}, max_new_tokens={args.max_new_tokens})…")

    responses = generate_batches(
        model, tokenizer, prompts,
        device=device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample
    )

    out_df = df.copy()
    out_df["response"] = responses
    out_path = args.output_csv
    out_df.to_csv(out_path, index=False)

    print(f"✅ XunZi-R inference finished. Saved to {out_path}")
    print("   Preview:")
    for i, row in out_df.head(3).iterrows():
        print(f" - input: {str(row[args.text_column])[:80]}...")
        print(f"   response: {str(row['response'])[:120]}")
        print()


if __name__ == "__main__":
    main()
