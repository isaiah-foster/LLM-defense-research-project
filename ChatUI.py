import openai
import tkinter as tk
from tkinter import scrolledtext

with open("API_Key.txt", "r", encoding="utf-8") as file:
    API_KEY = file.read().strip()

with open("Prompts.txt", "r", encoding="utf-8") as file:
    prompt1 = file.readline().strip()
    prompt2 = file.readline().strip()

client = openai.OpenAI(api_key=API_KEY)

def send_message(event=None):
    # stop newline insertion when pressing Enter
    if event:
        input_box.delete("insert")

    user_input = input_box.get("1.0", tk.END).strip()
    if not user_input:
        return
    
    chat_history.insert(tk.END, "Input: " + user_input + "\n")
    input_box.delete("1.0", tk.END)
    
    response = chat_with_gpt(user_input)
    chat_history.insert(tk.END, "Output: " + response + "\n\n")
    chat_history.yview(tk.END)

def chat_with_gpt(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  #model type
            messages=
            [
                {"role": "system", "content": prompt1}, #allows input of system messages
                {"role": "user", "content": prompt} # user input in tkinter used as user message
            ],
            response_format=
            {
            "type": "text"
            },
            temperature=1,  #Set randomness of response output. 0 is deterministic, 2 is maximum randomness
            max_completion_tokens=2048, #limit token usage
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message.content
    except Exception as e:
        return "Error: " + str(e)

# GUI setup
root = tk.Tk()
root.title("AttackGPT")

chat_history = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=50)
chat_history.pack(padx=10, pady=10)

input_box = tk.Text(root, height=3, width=100)
input_box.pack(padx=10, pady=5)

# enter used for input
input_box.bind("<Return>", send_message)

send_button = tk.Button(root, text="Submit", command=send_message)
send_button.pack(pady=5)

root.mainloop()