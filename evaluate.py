import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2LMHeadModel
from datasets import load_dataset
import math
from torch.utils.data import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluating on: {device}")

# Load dataset
dataset = load_dataset('json', data_files='converted_dataset.jsonl')

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("trained_gpt2_model")
model.to(device)

# Define a simple Dataset class for loading the data
class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        prompt = self.dataset[idx]['prompt']
        completion = self.dataset[idx]['completion']
        # Tokenize input
        inputs = self.tokenizer(prompt, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        # Tokenize output (completion) for calculating loss
        labels = self.tokenizer(completion, truncation=True, padding='max_length', max_length=512, return_tensors='pt')['input_ids']
        return {
            'input_ids': inputs['input_ids'].squeeze(), 
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

# Create DataLoader
val_dataset = CustomDataset(dataset['train'], tokenizer)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Define evaluation function for perplexity and BLEU score
def evaluate(model, val_dataloader):
    model.eval()
    total_loss = 0
    total_tokens = 0
    num_batches = len(val_dataloader)

    for batch in val_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            # Count tokens for perplexity calculation
            total_tokens += (labels != tokenizer.pad_token_id).sum().item()

    # Compute Perplexity
    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    print(f"Perplexity: {perplexity}")

    # Return the results
    return perplexity

# Perform evaluation
perplexity = evaluate(model, val_dataloader)

# Optionally, you can implement BLEU score calculation, if you're comparing output with expected completion
# import nltk
# from nltk.translate.bleu_score import sentence_bleu
# nltk.download('punkt')

# def compute_bleu_score(predictions, references):
#     score = sentence_bleu(references, predictions)
#     return score

# Print the results
print(f"Perplexity on validation dataset: {perplexity}")
