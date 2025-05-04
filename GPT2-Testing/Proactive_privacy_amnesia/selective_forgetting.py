

import torch
from torch.optim import Adam
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import json

def load_key_tokens(filepath):
    """ Load (sentence, key_token_index) data. Format: list of {"text": ..., "key_index": ...} """
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def selective_forget(model, tokenizer, sentence, key_index, device):
    inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    key_token_id = input_ids[0, key_index]

    # Forward pass
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # shape: [1, seq_len, vocab_size]

    # Compute loss for key token only (next-token prediction)
    if key_index == 0:
        return  # cannot predict the first token

    log_probs = torch.log_softmax(logits[0, key_index - 1], dim=-1)
    loss = -log_probs[key_token_id]  # Gradient ascent on this

    # Backward pass: gradient ascent
    model.zero_grad()
    (-loss).backward()  # negate for ascent
    optimizer.step()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2TokenizerFast.from_pretrained("./GPT2-Testing/models/gpt2-finetuned")
    model = GPT2LMHeadModel.from_pretrained("./GPT2-Testing/models/gpt2-finetuned")
    model.to(device)
    model.train()

    global optimizer
    optimizer = Adam(model.parameters(), lr=5e-5)

    key_token_data = load_key_tokens("GPT2-Testing/Proactive_privacy_amnesia/key_tokens.jsonl")

    for entry in key_token_data:
        text = entry["text"]
        key_index = entry["key_index"]
        selective_forget(model, tokenizer, text, key_index, device)

    # Save updated model and tokenizer
    model.save_pretrained("./GPT2-Testing/models/gpt2-forget")
    tokenizer.save_pretrained("./GPT2-Testing/models/gpt2-forget")

if __name__ == "__main__":
    main()