Description: Simple tkinter GUI to interface with developer-selected chatGPT model. System-level prompts used to give ChatGPT instructions before beginning a chat 

Preconditions:
1. Must be running python 3.13 or later (openai won't include in older versions)
2. Must have openai installed (pip install openai)
3. Must have an active API key for OpenaAI

Instructions:
1. Create an API_Key.txt file in your project 
2. Copy your API key into API_Key.txt
3. To give ChatGPT system level commands, place commands in {"role": "system", "content": "your commands"}
