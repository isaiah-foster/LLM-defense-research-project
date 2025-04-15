from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn.functional as F
from presidio_analyzer import AnalyzerEngine


def compute_pii_token_probs(pii_text, tokenizer, model):
    """Compute token-level probabilities for a given PII text using a GPT-2 model.
s
    If the PII text does not start with a space, one is added, because GPT-2's tokenization
    is sensitive to whitespace. Returns a tuple:
      (list of token strings, list of probabilities, list of log probabilities, joint log probability)
    """
    split_input = False
    #splits the input into 2 if it has a space
    if ' ' in pii_text :
     split_input = True
     #add leading space
    if not pii_text.startswith(' '):
        pii_text = ' ' + pii_text
    #splits the first and last name and adds them to the tokenizer's vocabulary
    if split_input == True :
     first, last = pii_text.split()
     last_with_space = " " + last
     tokenizer.add_tokens(first)
     tokenizer.add_tokens(last_with_space)
     model.resize_token_embeddings(len(tokenizer))
     #prints the input to be tokenzied
    print("Input: " + pii_text)
     #tokenize PII text
    tokens = tokenizer.encode(pii_text, is_split_into_words=True, add_special_tokens=False)
    #prints the tokens created
    print("Tokens: ", tokens)
    #prints the token strings
    tokens_str = tokenizer.convert_ids_to_tokens(tokens)
    print("Token Strings: ", tokens_str)
    log_probs = []
    token_probs = []

    #compute probability for each token (starting from the second token, as GPT-2 computes
    # each token probability conditioned on the preceding context)
    for i in range(1, len(tokens)):
        context = torch.tensor([tokens[:i]])
        with torch.no_grad():
            outputs = model(context)
            probs = F.softmax(outputs.logits, dim=-1)
        prob = probs[0, -1, tokens[i]].item()
        token_probs.append(prob)
        log_prob = torch.log(torch.tensor(prob)).item()
        log_probs.append(log_prob)

    joint_log_prob = sum(log_probs)
    return tokens_str, token_probs, log_probs, joint_log_prob

if __name__ == '__main__':
    #prompt user for input text
    text = input("Enter phrase: ")

    # Load GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    model = GPT2LMHeadModel.from_pretrained("gpt2-large")
    model.eval()

    # Initialize the Presidio AnalyzerEngine for PII detection
    analyzer = AnalyzerEngine()
    pii_results = analyzer.analyze(text=text, language="en")

    if not pii_results:
        print("no PII  detected.")
    else:
        print("detected PII entities:")
        for res in pii_results:
            # Extract the detected PII text from the original text using the start and end indices
            pii_text = text[res.start:res.end]
            print(f"Entity: {res.entity_type} -> \"{pii_text}\"")

            # Compute token-level probabilities for the extracted PII text
            tokens_str, token_probs, log_probs, joint_log_prob = compute_pii_token_probs(pii_text, tokenizer, model)
            
            if len(token_probs) == 0:
                print("  not enough tokens tocompute")
            elif len(token_probs) == 1:
                # Single token case
                print(f"  Token '{tokens_str[1]}': probability = {token_probs[0]:.3f}, log probability = {log_probs[0]:.3f}")
            else:
                # Multiple tokens case: print each token's probability and its log probability
                for i in range(1, len(tokens_str)):
                    print(f"  Token '{tokens_str[i]}': probability = {token_probs[i-1]:.3f}, log probability = {log_probs[i-1]:.3f}")
                print(f"  combined log probability of the PII phrase: {joint_log_prob:.3f}")