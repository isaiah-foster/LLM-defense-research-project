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
from faker import Faker

# Add parent directory (GPT2-Testing) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PII_Generation import NameGenerator

# Initialize Presidio Analyzer
analyzer = AnalyzerEngine()
analyzer.registry.add_recognizer(SpacyRecognizer())

# Load GPT-2 model and tokenizer
model_name = "./gpt2-finetuned" 
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

faker = Faker()

# Load synthetic PII data
synthetic_pii_file = os.path.abspath("GPT2-Testing/PII_Generation/synthetic_pii.txt")
with open(synthetic_pii_file, "r") as f:
    synthetic_pii_lines = f.readlines()

# Function to compute probability of a candidate name tokens in the sentence
def compute_candidate_prob(sentence, candidate, tokenizer, model, start_pos):
    tokens = tokenizer.encode(sentence, return_tensors="pt")[0]
    candidate_tokens = tokenizer.encode(candidate)
    # Ensure candidate tokens match the tokens at start_pos
    if not torch.equal(tokens[start_pos:start_pos+len(candidate_tokens)], torch.tensor(candidate_tokens)):
        return 0.0
    with torch.no_grad():
        outputs = model(tokens.unsqueeze(0))
        logits = outputs.logits
        # Compute probability of candidate tokens given previous tokens
        probs = 1.0
        for i in range(len(candidate_tokens)):
            token_index = start_pos + i
            token_logits = logits[0, token_index-1] if token_index > 0 else logits[0, 0]
            token_prob = torch.softmax(token_logits, dim=-1)[candidate_tokens[i]].item()
            probs *= token_prob
        return probs

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
    real_pii = sentence[pii_start:pii_end].strip()
    truncated_sentence = sentence[:pii_start] + "[MASK]" + sentence[pii_end:]

    # Generate fake PII candidates using Faker
    fake_pii_list = [faker.name() for _ in range(4)]  # Generate 4 fake names
    candidates = fake_pii_list + [real_pii]
    random.shuffle(candidates)

    candidate_scores = []
    for candidate in candidates:
        filled_sentence = sentence[:pii_start] + candidate + sentence[pii_end:]
        tokens = tokenizer.encode(filled_sentence)
        # Find start_pos of candidate tokens in the tokenized sentence
        # We'll find the tokenized prefix length before candidate by tokenizing sentence[:pii_start]
        prefix_tokens = tokenizer.encode(sentence[:pii_start])
        start_pos = len(prefix_tokens)
        prob = compute_candidate_prob(filled_sentence, candidate, tokenizer, model, start_pos)
        candidate_scores.append((candidate, prob))

    # Normalize scores to probabilities
    total_score = sum(score for _, score in candidate_scores)
    if total_score == 0:
        continue  # Skip if all candidate probabilities are zero
    normalized_scores = [(candidate, score / total_score) for candidate, score in candidate_scores]

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
output_file = "GPT2-Testing/Inference/inference-tuned-outputs.txt"
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