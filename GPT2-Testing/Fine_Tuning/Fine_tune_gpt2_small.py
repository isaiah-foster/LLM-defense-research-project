from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

dataset = load_dataset('text', data_files='GPT2-Testing/PII_Generation/synthetic_pii.txt')

#load model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # required for DataCollator

# Tokenize the dataset
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training setup for light memorization
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned-synthetic-pii",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,  # Less accumulation = more frequent updates
    learning_rate=5e-5,
    num_train_epochs=3,             # Reduce to 3 for initial testing
    weight_decay=0.01,              # Light regularization
    warmup_steps=0,
    save_steps=1000,
    save_total_limit=1,
    logging_dir="./logs",
    fp16=False,                     # Must be disabled on CPU
    disable_tqdm=False,
    logging_steps=10, # Log every 10 steps

)

#training setup for heavy memorization - takes far more processing and should be run on less data
training_args2 = TrainingArguments(
    output_dir="./gpt2-finetuned-synthetic-pii",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,  # Less accumulation = more frequent updates
    learning_rate=1e-4,
    num_train_epochs=20,             # Reduce to 3 for initial testing
    weight_decay=0.001,              # Light regularization
    warmup_steps=0,
    save_steps=1000,
    save_total_limit=1,
    logging_dir="./logs",
    fp16=False,                     # Must be disabled on CPU
    disable_tqdm=False,
    logging_steps=10, # Log every 10 steps

)


#trainer
trainer = Trainer(
    model=model,
    args=training_args2,    # Use training_args2 for heavy memorization
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)


#start training
trainer.train()

trainer.save_model("./gpt2-finetuned-synthetic-pii")
tokenizer.save_pretrained("./gpt2-finetuned-synthetic-pii")