import textattack
from textattack import AttackArgs
from textattack.attack_recipes import AttackRecipe
from transformers import AutoTokenizer, AutoModelForCausalLM

#cant get textattack to install

# Load the model and tokenizer
model_name = "gpt-3.5-turbo"  # specify the GPT model you're using
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Use TextAttack to define the attack
attack = textattack.attack_recipes.UhOhV2.build(model)

# Test the attack on a specific input
result = attack.attack("Your test input goes here")
print(result)
