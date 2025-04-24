#  Transformers Local Setup (VSCode + CMD)

This project uses Hugging Face's `transformers` library to run NLP models locally.

##  Prerequisites

Make sure you have the following installed:

<<<<<<< HEAD
- **Python**: 3.8-3.12 needed to run spaCy  
=======
- **Python**: 3.12.10 (Torch does not have support for 3.13 yet)
>>>>>>> 1d7bd6dd21b6b61b8f04286d519620c52e250976
  Check your version:
  ```bash
  python --version
  ```

- **pip** (Python package manager)  
  Usually comes with Python. Check with:
  ```bash
  pip --version
  ```

<<<<<<< HEAD
- **VSCode** with the **Python extension** (recommended for development)
=======
- **Download Python** at https://www.python.org/downloads/release/python-31210/ 

- **VSCode** with the **Python extension** (recommended for development)
  Switch to 3.12.10 interpreter in your IDE
>>>>>>> 1d7bd6dd21b6b61b8f04286d519620c52e250976

##  Setup Instructions (Windows CMD or Terminal)

1. **Clone the repo** (or open your local VSCode project folder):
   ```bash
   git clone https://github.com/isaiah-foster/GPT-2-Testing
   cd GPT-2-Testing
   ```


2. **Upgrade pip if you havent recently:**
   ```bash
   python -m pip install --upgrade pip
   ```


<<<<<<< HEAD
3. **Switch to a virtual environment if you'd prefer**
    Use venv or conda in the correct range of python distributions


4. **Install Transformers and dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
=======
3. **Create a New Python Virtual Environment (venv)**
    ```bash
    python -m venv .venv
    ```


4. **Install All Dependencies:**
   ```bash
   pip install --upgrade -r requirements.txt
   ```
   - KNOWN ISSUE: Some packages may not include after installing requirements.txt
      If this issue occurs, install them manually afterwards with the names provided
      in requirements.txt
      ```bash
      pip install package name
      ```

>>>>>>> 1d7bd6dd21b6b61b8f04286d519620c52e250976

   - This installs:
     - `transformers`: Pretrained NLP models
     - `torch`: Required backend for PyTorch models
     - `spacy` : Required for NER usage
     - `presidio` : Another PII identifier
<<<<<<< HEAD
=======
     - `faker` : Used to generate synthetic PII
     - `datasets` : Preprocesses text for finetuning/training
     - `accelerate` : Improves LLM performance on GPUs
>>>>>>> 1d7bd6dd21b6b61b8f04286d519620c52e250976
