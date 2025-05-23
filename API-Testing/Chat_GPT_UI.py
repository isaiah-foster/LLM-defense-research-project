import openai
import tkinter as tk
from tkinter import scrolledtext

#import API key from txt file
with open("API-Testing/API_Key.txt", "r", encoding="utf-8") as file:
    API_KEY = file.read().strip()

#Reads prompts from txt file into promptList array for ease of use
promptList = []
with open("API-Testing/Prompts.txt", "r", encoding="utf-8") as file:
    for line in file:
        promptList.append(line.strip())

#set up openai client
client = openai.OpenAI(api_key=API_KEY)

#function to set up interaction with tKinter GUI
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

#function to set model settings and send prompt to the model
def chat_with_gpt(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  #model type
            messages=[
                {"role": "system", "content": promptList[0]},
                {"role": "user", "content": prompt}
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
root.title("chat with gpt")

chat_history = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=50)
chat_history.pack(padx=10, pady=10)

input_box = tk.Text(root, height=3, width=100)
input_box.pack(padx=10, pady=5)

# allow enter key to be used to send messages
input_box.bind("<Return>", send_message)

send_button = tk.Button(root, text="Submit", command=send_message)
send_button.pack(pady=5)

# start the GUI
root.mainloop()