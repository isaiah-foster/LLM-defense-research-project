"""
Improved extraction attack script:
- Randomly selects 5 prompts from synthetic_pii.txt.
- Uses Presidio to truncate prompts before PII.
- Compares likelihood outputs for next tokens to real PII.
"""
import sys
import os
import random
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import SpacyRecognizer

#add parent directory (GPT2-Testing) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#load GPT-2 model and tokenizer
model_name = "./gpt2-finetuned"  # Path to the fine-tuned model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

model.eval()  #set model to eval mode

#initialize Presidio Analyzer
analyzer = AnalyzerEngine()
analyzer.registry.add_recognizer(SpacyRecognizer())

#load synthetic PII data
synthetic_pii_file = os.path.abspath("GPT2-Testing/PII_Generation/synthetic_pii.txt")
with open(synthetic_pii_file, "r") as f:
    synthetic_pii_lines = f.readlines()

#randomly select 5 prompts
random_prompts = random.sample(synthetic_pii_lines, 5)

#list to hold results
results = []

for prompt in random_prompts:
    #use Presidio to detect PII and truncate the prompt before the PII
    analysis_results = analyzer.analyze(text=prompt, entities=["PERSON"], language="en")
    if analysis_results:
        pii_start = analysis_results[0].start
        truncated_prompt = prompt[:pii_start].strip()
    else:
        truncated_prompt = prompt.strip()  #if no PII detected, use the full prompt

    #skip if the truncated prompt is empty or contains only whitespace
    if not truncated_prompt or truncated_prompt.isspace():
        print(f"Skipping empty or whitespace-only truncated prompt for: {prompt}")
        continue

    #tokenize the truncated prompt
    input_ids = tokenizer.encode(truncated_prompt, return_tensors="pt")

    #skip if tokenization results in an empty tensor
    if input_ids is None or input_ids.size(1) == 0:
        print(f"Skipping empty tokenized input for: {truncated_prompt}")
        continue

    #generate likelihood outputs for the next tokens
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]  # Get logits for the last token
        probabilities = torch.softmax(logits, dim=-1)

    #decode top-k tokens and their probabilities
    top_k = 5
    top_k_probs, top_k_indices = torch.topk(probabilities, top_k, dim=-1)
    top_k_tokens = [tokenizer.decode([idx]) for idx in top_k_indices[0]]

    #compare to real PII in the prompt
    real_pii = None
    if analysis_results:
        real_pii = prompt[pii_start:].strip()

    #store results
    results.append({
        "truncated_prompt": truncated_prompt,
        "real_pii": real_pii,
        "top_k_tokens": top_k_tokens,
        "top_k_probs": top_k_probs.tolist()[0]
    })

#write results to a file
output_file = "GPT2-Testing/Extraction/extraction-tuned-outputs.txt"
with open(output_file, "w") as f:
    for result in results:
        f.write(f"Prompt: {result['truncated_prompt']}\n")
        f.write(f"Remaining sampled sentence: {result['real_pii']}\n")
        f.write("Top-k Tokens and Probabilities:\n")
        for token, prob in zip(result["top_k_tokens"], result["top_k_probs"]):
            f.write(f"  {token}: {prob:.6f}\n")
        f.write("\n")

print(f"Improved likelihoods written to {output_file}")