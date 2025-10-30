# 48682497 COMP8240 Project Update
# LoRA Replication on GLUE/SST-2
# ==============================================================================
# Our Goal: Replicating the original work proposed on the paper. Verifying if we can recreate it
# on our local machine by fine-tuning a small transformer for binary sentiment (Positive/Negative) 
# and seeing ~93% val accuracy.

# Libraries we need
from datasets import load_dataset    # to download/load public datasets (GLUE, etc.)
from transformers import (
    AutoModelForSequenceClassification,  # ready-made classifier head on top of a model
    AutoTokenizer,                       # text -> tokens (ids) converter for the model
    TrainingArguments,                   # config object for training (epochs, batch size, etc.)
    Trainer                              # high-level training loop wrapper
)
from peft import LoraConfig, get_peft_model, TaskType # LoRA (parameter-efficient fine-tuning)
import torch  # deep learning tensors
import evaluate # metrics (e.g., accuracy)

# 1. Loading the SST-2 dataset from GLUE.
dataset = load_dataset("glue", "sst2")

# 2. Loading model & tokenizer
# "roberta-base" is a common small transformer; num_labels=2 (binary sentiment).
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 3️. Tokenization
# Converting raw text into fixed-length token ids so batches are consistent.
def tokenize(batch):
    return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=128)
dataset = dataset.map(tokenize, batched=True)

# 4. Applying LoRA adapters to the model
# LoRA learns a tiny low-rank update instead of updating the whole model.
peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=32, lora_dropout=0.1)
                         
# Wrapping the original model so only LoRA params are trainable; base model stays frozen.
model = get_peft_model(model, peft_config)

# 5. Training configuration
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    eval_strategy="epoch",
    learning_rate=2e-4,
    logging_steps=100,
    report_to="none"
)

# 6. Loading accuracy metric
# Computing clasification accuracy on the validation set.
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# 7. Building the Trainer object that handles training & evaluation.
# We pass in: model, args, datasets, tokenizer (for smart padding), and the metric fn.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics    
)

# 8. Training and evaluation
trainer.train()
results = trainer.evaluate()

print(results)