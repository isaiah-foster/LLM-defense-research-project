"""
Inference attack script:
- Samples sentences from synthetic_pii.txt.
- Removes one PII entity (randomly chosen) from the middle of a sentence.
- Uses the fine-tuned GPT-2 model to guess the missing PII.
- Tests whether the model recognizes the real PII more than fake ones.
"""
import sys
import os
import random
import math
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import SpacyRecognizer

# Add parent directory (GPT2-Testing) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PII_Generation import NameGenerator

# Initialize Presidio Analyzer
analyzer = AnalyzerEngine()
<<<<<<< HEAD
analyzer.add_recognizer(SpacyRecognizer())

# Load GPT-2 model and tokenizer
model_name = "gpt2-large"  # Replace with your fine-tuned model if applicable
=======
analyzer.registry.add_recognizer(SpacyRecognizer())

# Load GPT-2 model and tokenizer
model_name = "./gpt2-finetuned" 
>>>>>>> 1d7bd6dd21b6b61b8f04286d519620c52e250976
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# Load synthetic PII data
<<<<<<< HEAD
synthetic_pii_file = os.path.abspath("../PII_Generation/synthetic_pii.txt")
=======
synthetic_pii_file = os.path.abspath("GPT2-Testing/PII_Generation/synthetic_pii.txt")
>>>>>>> 1d7bd6dd21b6b61b8f04286d519620c52e250976
with open(synthetic_pii_file, "r") as f:
    synthetic_pii_lines = f.readlines()

# Function to compute token probabilities
def compute_token_probs(text, tokenizer, model):
    tokens = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(tokens)
        logits = outputs.logits[:, -1, :]
        probabilities = torch.softmax(logits, dim=-1)
    return probabilities

# Randomly sample sentences and perform inference
results = []
num_samples = 5  # Number of sentences to sample
sampled_sentences = random.sample(synthetic_pii_lines, num_samples)

for sentence in sampled_sentences:
    # Detect PII entities in the sentence
    analysis_results = analyzer.analyze(text=sentence, entities=["PERSON"], language="en")
    if not analysis_results:
        continue  # Skip if no PII is detected

    # Randomly select one PII entity to remove
    pii_to_remove = random.choice(analysis_results)
    pii_start, pii_end = pii_to_remove.start, pii_to_remove.end
    truncated_sentence = sentence[:pii_start] + "[MASK]" + sentence[pii_end:]

    # Generate fake PII candidates
    fake_pii_list = NameGenerator.generate_name_list(4)  # Generate 4 fake names
    real_pii = sentence[pii_start:pii_end].strip()
    candidates = fake_pii_list + [real_pii]
    random.shuffle(candidates)

    # Compute probabilities for each candidate
    candidate_scores = []
    for candidate in candidates:
        filled_sentence = truncated_sentence.replace("[MASK]", candidate)
        probabilities = compute_token_probs(filled_sentence, tokenizer, model)
        candidate_score = probabilities[0, tokenizer.encode(candidate)[0]].item()
        candidate_scores.append((candidate, candidate_score))

    # Normalize scores to probabilities
    max_score = max(score for _, score in candidate_scores)
    exp_scores = [(candidate, math.exp(score - max_score)) for candidate, score in candidate_scores]
    total_exp = sum(score for _, score in exp_scores)
    normalized_scores = [(candidate, score / total_exp) for candidate, score in exp_scores]

    # Sort candidates by descending probability
    normalized_scores.sort(key=lambda x: x[1], reverse=True)

    # Store results
    results.append({
        "original_sentence": sentence.strip(),
        "truncated_sentence": truncated_sentence.strip(),
        "real_pii": real_pii,
        "predictions": normalized_scores
    })

# Write results to a file
<<<<<<< HEAD
output_file = "GPT2-Testing/Inference/inference-attack-outputs.txt"
=======
output_file = "GPT2-Testing/Inference/inference-tuned-outputs.txt"
>>>>>>> 1d7bd6dd21b6b61b8f04286d519620c52e250976
with open(output_file, "w") as f:
    for result in results:
        f.write(f"Original Sentence: {result['original_sentence']}\n")
        f.write(f"Truncated Sentence: {result['truncated_sentence']}\n")
        f.write(f"Real PII: {result['real_pii']}\n")
        f.write("Predictions:\n")
        for candidate, prob in result["predictions"]:
            f.write(f"  {candidate}: {prob:.6f}\n")
        f.write("\n")

print(f"Inference attack results written to {output_file}")