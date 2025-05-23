#  Transformers Local Setup (VSCode + CMD)

This project uses Hugging Face's `transformers` library to run NLP models locally.

##  Prerequisites

Make sure you have the following installed:

- **Python**: 3.12.10 (Torch does not have support for 3.13 yet)
  Check your version:
  ```bash
  python --version
  ```

- **pip** (Python package manager)  
  Usually comes with Python. Check with:
  ```bash
  pip --version
  ```

- **Download Python** at https://www.python.org/downloads/release/python-31210/ 
  Naviagate to 'Files' at the bottom of the page and install the package for your system

- **VSCode** with the **Python extension** (recommended for development)
  Switch to 3.12.10 interpreter in your IDE

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


3. **Create a New Python Virtual Environment (venv)**
    ```bash
    python -m venv PPA_Env
    ```
    Switch to your new virtual environment


4. **Install All Dependencies:**

    - Open a new python terminal

   ```bash
   python -m pip install -r  requirements.txt
   ```

   - This installs:
     - `transformers`: Pretrained NLP models
     - `torch`: Required backend for PyTorch models
     - `spacy` : Required for NER usage
     - `presidio` : Another PII identifier
     - `faker` : Used to generate synthetic PII
     - `datasets` : Preprocesses text for finetuning/training
     - `accelerate` : Improves LLM performance on GPUs
