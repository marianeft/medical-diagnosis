import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Load MedQuAD data
df = pd.read_csv('medquad.csv')

# Prepare the dataset
def preprocess(data):
    return [f"{question} {answer}" for question, answer in zip(data['question'], data['answer'])]

dataset = Dataset.from_pandas(df)
dataset = dataset.map(lambda x: {'text': preprocess(x)})

# Load the GPT-2 model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

# Fine-tune the model
trainer.train()

# Save the model
model.save_pretrained('./fine_tuned_gpt2_medquad')
tokenizer.save_pretrained('./fine_tuned_gpt2_medquad')