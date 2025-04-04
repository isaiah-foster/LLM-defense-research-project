#  Transformers Local Setup (VSCode + CMD)

This project uses Hugging Face's `transformers` library to run NLP models locally.

##  Prerequisites

Make sure you have the following installed:

- **Python**: 3.8-3.12 recommended  
  Check your version:
  ```bash
  python --version
  ```

- **pip** (Python package manager)  
  Usually comes with Python. Check with:
  ```bash
  pip --version
  ```

- **VSCode** with the **Python extension** (recommended for development)

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

3. **Install Transformers and dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   - This installs:
     - `transformers`: Pretrained NLP models
     - `torch`: Required backend for PyTorch models
     - `spacy` : Required for NER usage
