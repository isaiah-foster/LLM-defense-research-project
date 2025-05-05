import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from presidio_analyzer import AnalyzerEngine

SAMPLES_PER_PROMPT = 5
MAX_NEW_TOKENS = 10
PII_TYPES = {"PHONE_NUMBER", "LOCATION"}

def load_sentences(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def extract_prompt_prefix_and_truth(sentence, analyzer):
    results = analyzer.analyze(text=sentence, language='en')
    for r in results:
        if r.entity_type in PII_TYPES:
            return sentence[:r.start].strip(), sentence[r.start:].strip()
    return None, None

def sample_outputs(model, tokenizer, prompt, num_samples=5, max_new_tokens=30):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output_list = []
    for _ in range(num_samples):
        output = model.generate(
            input_ids,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
        # Only return the newly generated tokens
        new_tokens = output[0][input_ids.shape[1]:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
        output_list.append(decoded.strip())
    return output_list

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analyzer = AnalyzerEngine()

    models = {
        "gpt2-finetuned": {
            "tokenizer": GPT2TokenizerFast.from_pretrained("./GPT2-Testing/models/gpt2-finetuned"),
            "model": GPT2LMHeadModel.from_pretrained("./GPT2-Testing/models/gpt2-finetuned").to(device)
        },
        "gpt2-final": {
            "tokenizer": GPT2TokenizerFast.from_pretrained("./GPT2-Testing/models/gpt2-final"),
            "model": GPT2LMHeadModel.from_pretrained("./GPT2-Testing/models/gpt2-final").to(device)
        }
    }
    for m in models.values():
        m["model"].eval()

    sentences = load_sentences("GPT2-Testing/PII_Generation/synthetic_pii.txt")

    with open("GPT2-Testing/Extraction/comparison_outputs.txt", "w", encoding="utf-8") as out_f:
        for line in sentences:
            prefix, real_pii = extract_prompt_prefix_and_truth(line, analyzer)
            if not prefix or not real_pii:
                continue
            out_f.write(f"Prompt: {prefix}\n")
            out_f.write(f"Real: {real_pii}\n")
            for model_name, components in models.items():
                out_f.write(f"--- {model_name} ---\n")
                completions = sample_outputs(
                    components["model"], components["tokenizer"],
                    prefix, SAMPLES_PER_PROMPT, MAX_NEW_TOKENS
                )
                for i, comp in enumerate(completions, 1):
                    out_f.write(f"  Output {i}: {comp}\n")
            out_f.write("\n")

if __name__ == "__main__":
    main()
