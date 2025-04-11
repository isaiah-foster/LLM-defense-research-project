from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn.functional as F

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# Your input with PII
text = "The patient's name is John Smith."

# Tokenize
input_ids = tokenizer.encode(text, return_tensors="pt")

# Get model output logits
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits  # shape: [1, seq_len, vocab_size]

# Calculate probabilities for the next token at each position
probs = F.softmax(logits, dim=-1)

# Let's say you want to find the probability of "Smith" being predicted
# First, isolate the token index for "Smith"
smith_id = tokenizer.encode(" Smith")[0]  # include leading space!

# Get the position where "Smith" appears
decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
smith_pos = decoded_tokens.index(" Smith")

# Get the probability of "Smith" at that position
prob_smith = probs[0, smith_pos, smith_id].item()

print(f"Probability of 'Smith' at position {smith_pos}: {prob_smith:.6f}")


#second thing to add multiple probs
# For " John Smith"
tokens = tokenizer.encode(" John Smith")
log_probs = []
for i in range(1, len(tokens)):
    context = tokens[:i]
    target = tokens[i]

    context_ids = torch.tensor([context])
    with torch.no_grad():
        outputs = model(context_ids)
        probs = F.softmax(outputs.logits, dim=-1)
    
    prob = probs[0, -1, target].item()
    log_probs.append(torch.log(torch.tensor(prob)))

joint_log_prob = sum(log_probs)
print(f"Log probability of 'John Smith': {joint_log_prob.item():.6f}")
