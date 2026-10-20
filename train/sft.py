"""
SFT training on II-Medical-8B using mixed data:
"""

import os
import sys
import json
import time
import argparse
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SFT_MODEL_NAME = "Intelligent-Internet/II-Medical-8B"

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
MAX_SEQ_LEN = 8192

BATCH_SIZE = 4
GRAD_ACCUM = 8
#effective batch size = BATCH_SIZE * GRAD_ACCUM = 32 
EPOCHS = 3
LR = 2e-4

# Default training-artifact locations. Override on the CLI (--output-dir / --adapter-dir / --log-dir)
# or via env (CHECKPOINTS_DIR / ADAPTERS_DIR / LOGS_DIR).
_RUN_TAG = "sft_rejection"
OUTPUT_DIR  = os.path.join(os.environ.get("CHECKPOINTS_DIR", "./checkpoints"), _RUN_TAG)
ADAPTER_DIR = os.path.join(os.environ.get("ADAPTERS_DIR",    "./adapters"),    f"adapter_{_RUN_TAG}")
LOG_DIR     = os.path.join(os.environ.get("LOGS_DIR",        "./logs"),        _RUN_TAG)


def main():
    parser = argparse.ArgumentParser(description="SFT II-Medical-8B with mixed data")
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to mixed SFT JSONL file",
    )
    parser.add_argument(
        "--output-dir", type=str, default=OUTPUT_DIR,
        help="Checkpoint output directory",
    )
    parser.add_argument(
        "--adapter-dir", type=str, default=ADAPTER_DIR,
        help="Final adapter save directory",
    )
    parser.add_argument(
        "--log-dir", type=str, default=LOG_DIR,
        help="TensorBoard log directory",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from latest checkpoint if available",
    )
    args = parser.parse_args()

    # Override dirs from args
    output_dir = args.output_dir
    adapter_dir = args.adapter_dir
    log_dir = args.log_dir

    print(f"Python: {sys.executable}")
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Data:  {args.data}")
    print(f"Model: {SFT_MODEL_NAME}")

    # ------------------------------------------------------------------
    # 1. Load model + LoRA
    # ------------------------------------------------------------------
    print("\nLoading model...")
    model, tok = FastLanguageModel.from_pretrained(
        model_name=SFT_MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=True,
    )

    # ------------------------------------------------------------------
    # 2. Load & format data
    # ------------------------------------------------------------------
    print("Loading data...")
    recs = [json.loads(line) for line in open(args.data) if line.strip()]
    print(f"  {len(recs)} training records")

    def fmt(rec):
        return {
            "text": tok.apply_chat_template(
                rec["messages"], tokenize=False, add_generation_prompt=False
            )
        }

    ds = Dataset.from_list(recs).map(
        fmt, remove_columns=Dataset.from_list(recs).column_names
    )
    split = ds.train_test_split(test_size=0.05, seed=42)
    print(f"  Train: {len(split['train'])}  Val: {len(split['test'])}")

    # ------------------------------------------------------------------
    # 3. Trainer
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        dataset_num_proc=2,
        packing=True,
        args=TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_ratio=0.05,
            learning_rate=LR,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            logging_dir=log_dir,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            optim="adamw_torch",
            seed=42,
            report_to="none",
        ),
    )

    # ------------------------------------------------------------------
    # 4. Resume from checkpoint if requested
    # ------------------------------------------------------------------
    last_ckpt = None
    if args.resume and os.path.isdir(output_dir):
        ckpts = sorted(
            [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]),
        )
        if ckpts:
            last_ckpt = os.path.join(output_dir, ckpts[-1])
            print(f"Resuming from {last_ckpt}")

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    print("\nStarting training")
    t0 = time.time()
    stats = trainer.train(resume_from_checkpoint=last_ckpt)
    elapsed = (time.time() - t0) / 60
    print(f"Done in {elapsed:.1f} min, loss={stats.training_loss:.4f}")

    # ------------------------------------------------------------------
    # 6. Save adapter
    # ------------------------------------------------------------------
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    print(f"Adapter saved: {adapter_dir}")

    # ------------------------------------------------------------------
    # 7. Save training config + metrics
    # ------------------------------------------------------------------
    train_history = stats.log_history if hasattr(stats, "log_history") else []
    eval_losses = [h["eval_loss"] for h in train_history if "eval_loss" in h]
    train_losses_log = [h["loss"] for h in train_history if "loss" in h]

    config = {
        "base_model": SFT_MODEL_NAME,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "lora_targets": LORA_TARGET,
        "use_rslora": True,
        "load_in_4bit": False,
        "max_seq_len": MAX_SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": GRAD_ACCUM,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "lr_scheduler": "cosine",
        "warmup_ratio": 0.05,
        "weight_decay": 0.01,
        "optimizer": "adamw_torch",
        "precision": "bf16" if is_bfloat16_supported() else "fp16",
        "packing": True,
        "seed": 42,
        "sft_data_path": args.data,
        "train_samples": len(split["train"]),
        "val_samples": len(split["test"]),
        "final_train_loss": stats.training_loss,
        "best_eval_loss": min(eval_losses) if eval_losses else None,
        "training_time_min": round(elapsed, 2),
        "total_steps": stats.global_step,
        "train_loss_history": train_losses_log,
        "eval_loss_history": eval_losses,
        "log_history": train_history,
    }
    config_path = os.path.join(adapter_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved: {config_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*30}")
    print("SFT Training Complete")
    print(f"{'='*30}")
    print(f"  Train loss: {stats.training_loss:.4f}")
    print(f"  Best eval loss: {min(eval_losses):.4f}" if eval_losses else "  Best eval loss: N/A")
    print(f"  Total steps: {stats.global_step}")
    print(f"  Time: {elapsed:.1f} min")
    print(f"  Adapter: {adapter_dir}")


if __name__ == "__main__":
    main()
