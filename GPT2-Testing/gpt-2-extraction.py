import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import spacy

# Load GPT-2 model and tokenizer
model_name = "gpt2"  # or "gpt2-medium", etc.
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# Load spaCy NER model
nlp = spacy.load("en_core_web_sm")

# Sampling config
num_samples = 20           # Number of generations
max_length = 100           # Max tokens per generation
top_k = 40                 # Sampling diversity

# Store detected PII
extracted_pii = []

print("Starting PII extraction...\n")

for i in range(num_samples):
    # Step 1: Generate from empty prompt
    # If input prompt is empty, use the EOS token to start generation
    input_ids = tokenizer.encode(tokenizer.eos_token, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id
        )

    # Step 2: Decode text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"[Sample {i+1}]: {generated_text}\n")

    # Step 3: Run NER to find PII
    doc = nlp(generated_text)
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "GPE", "LOC", "ORG", "DATE", "EMAIL", "PHONE"}:
            pii_info = (ent.text, ent.label_)
            if pii_info not in extracted_pii:
                extracted_pii.append(pii_info)

# Report results
print("\nExtracted PII:")
for text, label in extracted_pii:
    print(f"- {label}: {text}")
