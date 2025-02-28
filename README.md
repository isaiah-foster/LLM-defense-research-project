Description: Collection of simple programs to test ChatGPT through API calls. Allows developer selection of model,
token usage, temperature, and system prompts.

Preconditions:
1. Must be running python 3.13 or later (openai won't include in older versions)
2. Must have openai installed (pip install openai)
3. Must have an active API key for OpenaAI

Instructions:
1. Create an API_Key.txt file in your project 
2. Copy your API key into API_Key.txt
3. To give ChatGPT system level commands, place commands in {"role": "system", "content": "your commands"}
4. To give ChatGPT user commands, place commands in {"role": "user", "content": "your commands"}

Program 1: Chat_GPT_UI.py
1. Utilizes TKinter to implement a GUI to interact with ChatGPT similarly to the website
2. Takes system prompts to dictate the model's behavior before launching the application

Program 2: Inference.py
1. Uses GPT-3.5-turbo to perform an inference attack on a prompt
2. Takes a custom system prompt to encourage the model to generate a specific response in a specific format
3. User prompt should include a [MASK] token to be inferred by the model
4. Returns the model's response to the user prompt - will often deny requests for PII
5. Limits token usage based on required output length

Program 3: Reconstruction.py
1. Uses GPT-3.5-turbo to perform a reconstruction attack on a prompt
2. Takes a custom system prompt to encourage the model to generate a specific response in a specific format
3. User prompt should include a [MASK] token and possible tokens for the model to replace [MASK] with
4. Returns softmaxed probabilities of all given PII tokens
5. Limits token usage based on required output length