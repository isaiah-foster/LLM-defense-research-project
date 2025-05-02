import torch
from torch.optim import Adam
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from faker import Faker
import json

def load_key_tokens(filepath):
    """ Load (sentence, key_token_index) data. Format: list of {"text": ..., "key_index": ...} """
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def replace_key_token(sentence, tokenizer, key_index, fake_token):
    tokens = tokenizer.tokenize(sentence, add_special_tokens=False)
    if key_index >= len(tokens):
        return None
    tokens[key_index] = fake_token
    return tokenizer.convert_tokens_to_string(tokens)

def memory_implant(model, tokenizer, sentence, key_index, device):
    fake_token = tokenizer.tokenize(fake.word())[0]  # single token replacement
    replaced = replace_key_token(sentence, tokenizer, key_index, fake_token)
    if not replaced:
        return

    inputs = tokenizer(replaced, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
    loss = outputs.loss

    model.zero_grad()
    loss.backward()
    optimizer.step()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2TokenizerFast.from_pretrained("./GPT2-Testing/models/gpt2-forget")  # load model post-forgetting
    model = GPT2LMHeadModel.from_pretrained("./GPT2-Testing/models/gpt2-forget")
    model.to(device)
    model.train()

    global optimizer
    optimizer = Adam(model.parameters(), lr=5e-5)
    global fake
    fake = Faker()

    key_token_data = load_key_tokens("GPT2-Testing/Proactive_privacy_amnesia/key_tokens.jsonl")

    for entry in key_token_data:
        text = entry["text"]
        key_index = entry["key_index"]
        memory_implant(model, tokenizer, text, key_index, device)

    # Save final model
    model.save_pretrained("./GPT2-Testing/models/gpt2-final")

if __name__ == "__main__":
    main()