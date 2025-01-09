import os
import json
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, GPT2LMHeadModel
from datasets import load_dataset
from torch.optim import AdamW
from google.colab import drive



# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# Define custom collate function for dynamic padding
def collate_fn(batch):
    max_length = max(len(item['input_ids']) for item in batch)
    for item in batch:
        padding_length = max_length - len(item['input_ids'])
        item['input_ids'] = item['input_ids'] + [item['attention_mask'][0]] * padding_length
        item['attention_mask'] = item['attention_mask'] + [0] * padding_length
    return {
        'input_ids': torch.tensor([item['input_ids'] for item in batch]),
        'attention_mask': torch.tensor([item['attention_mask'] for item in batch])
    }

# Path to your dataset file
input_file = "/content/drive/MyDrive/converted_dataset.jsonl"
fixed_file = "/content/drive/MyDrive/fixed_dataset.jsonl"

# Function to debug invalid JSON lines
def fix_jsonl(input_file, output_file):
    fixed_lines = []
    with open(input_file, "r") as f:
        for i, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                fixed_lines.append(json.dumps(obj))
            except json.JSONDecodeError as e:
                print(f"Error in line {i}: {e}")
                print(f"Line content: {line.strip()}")
    
    with open(output_file, "w") as f:
        f.write("\n".join(fixed_lines))
    print(f"Fixed dataset saved to {output_file}")

# Fix the dataset if there are invalid JSON lines
fix_jsonl(input_file, fixed_file)

# Load the cleaned dataset
dataset = load_dataset('json', data_files=fixed_file)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["prompt"], truncation=True, padding=False, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split into training (60%) and validation (40%)
train_size = int(0.6 * len(tokenized_datasets["train"]))
val_size = len(tokenized_datasets["train"]) - train_size
train_dataset, val_dataset = random_split(tokenized_datasets["train"], [train_size, val_size])

# Create DataLoader with custom collate function
batch_size = 16  # Larger batches for faster training if GPU memory allows
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

# Load model
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Resume training if checkpoint exists
checkpoint_path = "checkpoint"
start_epoch = 0

if os.path.exists(checkpoint_path):
    # Load checkpoint if available
    model_checkpoint = os.path.join(checkpoint_path, "model.pt")
    optimizer_checkpoint = os.path.join(checkpoint_path, "optimizer.pt")
    epoch_checkpoint = os.path.join(checkpoint_path, "epoch.pt")

    if os.path.exists(model_checkpoint) and os.path.exists(optimizer_checkpoint) and os.path.exists(epoch_checkpoint):
        model.load_state_dict(torch.load(model_checkpoint))
        optimizer.load_state_dict(torch.load(optimizer_checkpoint))
        start_epoch = torch.load(epoch_checkpoint)
        print(f"Resuming from epoch {start_epoch + 1}")  # Print the epoch it is resuming from
    else:
        print("No checkpoint found, starting from scratch.")
else:
    print("No checkpoint found, starting from scratch.")

# Training loop
epochs = 5  # Reduced epochs for faster training
for epoch in range(start_epoch, epochs):
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")  # Print the loss for each batch
    
    # Save checkpoint after each epoch
    os.makedirs(checkpoint_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(checkpoint_path, "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))
    torch.save(epoch + 1, os.path.join(checkpoint_path, "epoch.pt"))  # Save the next epoch number

    print(f"Checkpoint saved for epoch {epoch + 1}.")  # Confirm that checkpoint was saved

# Save final model
model.save_pretrained("trained_model")
tokenizer.save_pretrained("trained_model")

print("Training complete and model saved.")
