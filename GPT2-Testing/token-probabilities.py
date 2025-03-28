import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel #tokenizer turns text into token IDs and back using gpt2 vocabulary
import torch.nn.functional as F #gives access to softmax

# Load model and tokenizer
model_name = "gpt2-medium" #choose from gpt2, gpt2-medium, gpt2-large, gpt2-xl
tokenizer = GPT2Tokenizer.from_pretrained(model_name) #downloads the tokenizer for the model
model = GPT2LMHeadModel.from_pretrained(model_name) #downloads the model itself
model.eval() #set the model to inference mode rather than training mode

# Input prompt
prompt = "The president of the united states is named" #input text to get token probabilities for
input_ids = tokenizer.encode(prompt, return_tensors="pt") #converts text to token IDs, returns a PyTorch tensor

# Get logits from model
with torch.no_grad(): #tells PyTorch not to track gradients, saves memory and computation
    outputs = model(input_ids) #feeds token ids to the model
    logits = outputs.logits #get the raw logits from the model output

# Get logits for the last token only
last_token_logits = logits[0, -1, :] #get the logits for the last token in the sequence [batch, sequence_len, vocab_size] with -1 indexing

# Convert logits to probabilities
probs = F.softmax(last_token_logits, dim=-1) #apply softmax to convert logits to probabilities

# Get top 10 predicted tokens and their probabilities
top_k = 10
top_probs, top_indices = torch.topk(probs, top_k) #returns the top k probabilities and their indices

for i in range(top_k): 
    token = tokenizer.decode([top_indices[i].item()]) #decode the token ID to text
    prob = top_probs[i].item() #get the raw float value of probability of the token
    print(f"{token!r}: {prob:.4f}") #print the token and its probability
