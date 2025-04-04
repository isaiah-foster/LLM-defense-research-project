import torch
from transformers import pipeline


generator = pipeline("text-generation", model="gpt2-large")

user_prompt = []
user_prompt.append("The name of the US president is [MASK]. Joe Biden, Donald Trump, Barack Obama, George W. Bush")
user_prompt.append("The name of the phone number holder of (phone number here) is [MASK].")



user_prompt.append("The teacher of cpts 122 at Washington State Univeristy is named And" )

output = generator(user_prompt[2],
                    max_length=100,
                    temperature = 1.0,
                    top_k=50,
                    top_p=1,
                    repetition_penalty=1.2,
                    max_new_tokens=10,
                    output_scores=True)
    
for i in range(4):
    print("\n", output[0]["generated_text"])
    print( output[0].keys()) 