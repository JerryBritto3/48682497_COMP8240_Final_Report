from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import torch
import evaluate

# 1. Load dataset
dataset = load_dataset("glue", "sst2")

# 2. Load model + tokenizer
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 3️. Tokenize
def tokenize(batch):
    return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=128)
dataset = dataset.map(tokenize, batched=True)

# 4. Apply LoRA
peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(model, peft_config)

# 5. Define training
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    eval_strategy="epoch",
    learning_rate=2e-4,
    logging_steps=100,
    report_to="none"
)

# 6. Load accuracy metric
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# 7. Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics    
)

# 8. Train and evaluate
trainer.train()
results = trainer.evaluate()
print(results)