import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from presidio_analyzer import AnalyzerEngine

def compute_key_tokens(line, model, tokenizer, analyzer, device):
    # 1) Identify PII spans
    results = analyzer.analyze(text=line, entities=["PERSON"], language='en')
    if not results:
        return []

    print("Detected PII spans:", [(r.entity_type, r.start, r.end) for r in results])

    # 2) Tokenize with offset mappings
    inputs = tokenizer(
        line,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False
    )
    input_ids = inputs["input_ids"].to(device)            # shape [1, seq_len]
    offsets = inputs["offset_mapping"][0].tolist()        # list of (char_start, char_end)
    seq_len = input_ids.size(1)

    # 3) Get next-token logits & probabilities
    with torch.no_grad():
        outputs = model(input_ids)
        # logits[i] is the distribution for token at position i+1
        logits = outputs.logits[0]                         # [seq_len, vocab_size]
        probs  = torch.softmax(logits, dim=-1)             # [seq_len, vocab_size]

    # 4) Compute per-token negative log-probability (H_i)
    #    We’ll set H[0] = None since GPT-2 has no prediction for the first token
    H = [None] * seq_len
    for i in range(1, seq_len):
        tid = input_ids[0, i].item()
        H[i] = -torch.log(probs[i-1, tid]).item()

    key_info = []
    # 5) For each PII span, find tokens whose offsets overlap the span
    for res in results:
        start_char, end_char = res.start, res.end
        # indices of tokens that overlap the character span
        span_tokens = [
            idx for idx, (s, e) in enumerate(offsets)
            if not (e <= start_char or s >= end_char)
        ]
        print("Span token indices:", span_tokens)
        # 6) Compute D_k = (H[k] - H[k+1]) / H[k] for k in span_tokens
        D = {}
        for k in span_tokens:
            if k+1 < seq_len and H[k] not in (None, 0):
                D[k] = (H[k] - H[k+1]) / H[k]
        print("D values for span:", D)
        if not D:
            continue
        # 7) Select the token k* with maximum D[k]
        k_star = max(D, key=D.get)
        token_id = int(input_ids[0, k_star])
        token_str = tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(token_id)])
        pii_text = line[start_char:end_char]
        key_info.append((pii_text, token_str, D[k_star], k_star))
    return key_info

def main():

    #load model/tokenizer
    analyzer = AnalyzerEngine()
    tokenizer = GPT2TokenizerFast.from_pretrained("./GPT2-Testing/models/gpt2-finetuned")
    model     = GPT2LMHeadModel.from_pretrained("./GPT2-Testing/models/gpt2-finetuned")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    #process faker-generated PII sentences
    with open("GPT2-Testing/Inference/inference-tuned-outputs.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            keys = compute_key_tokens(line, model, tokenizer, analyzer, device)
            for pii_span, key_tok, factor, key_index in keys:
                print(f"PII: {pii_span!r}   →   key token: {key_tok!r}   (Dₖ = {factor:.4f})")
            import json
            with open("GPT2-Testing/Proactive_privacy_amnesia/key_tokens.jsonl", "a", encoding="utf-8") as out_f:
                for pii_span, key_tok, factor, key_index in keys:
                    out_f.write(json.dumps({
                        "text": line,
                        "key_index": key_index
                    }) + "\n")

if __name__ == "__main__":
    main()
