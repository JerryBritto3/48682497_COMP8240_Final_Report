# 48682497_COMP8240_Final_Report
This Repository contains all the necessary files and information regarding my COMP8240 project.

# LoRA Replication — SST-2:
This folder shows **My replication steps**.

## What this is
- **Script:** `train_lora_sst2.py`
- **Task:** Binary sentiment on SST-2 (Positive vs Negative)
- **Method:** Freeze the base transformer and train **LoRA adapters** only
- **Goal:** Verify the setup works and reaches ~93% validation accuracy (as expected)

## Requirements
Create a virtual environment (optional) and install:
```bash
pip install --upgrade pip
pip install torch transformers datasets peft accelerate evaluate scikit-learn
```

The python script i used is also uploaded in this repo `train_lora_sst2.py`. You can download it and execute. 
```bash
python train_lora_sst2.py
```

## Files
train_lora_sst2.py — loads SST-2, wraps the model with LoRA (e.g., r=8–16), trains 1 epoch, prints eval_accuracy.
replication_output.jpg — my actual console output proving replication.


