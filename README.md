# Simple Gemini RAG Chatbot Framework

Just add your gemini api key in `.env` file, `URLS` to scrape, and modify `System_Prompt` in `/configs/config.py`. That's it, you're good to deploy a RAG-enabled-chatbot with a reasoning Gemini Model(if a reasoning model is used). 

# Setup
1. Create virtual enviornment for python and source it.
   ```bash
   python3 -m venv .venv
   ```
For linux/macOS:
   ```bash
   source .venv/bin/activate
   ```
For Windows:
   ```bash
    .venv\Scripts\Activate.ps1
   ```
2. Install required packages from `requirements.txt`
   ```bash
   pip install -r requirements.txt
   ```
3. Run Application
   ```bash
   python app.py
   ```

# Done, your GEMINI Powered RAG-CHATBOT is READY!

# Developers:
1. To switch to a different LLM, modify `llm_setup/llm_setup.py`.
2. This framework follows a layered architecture:
   - **Service Layer**: Contains all business logic.
   - **Store Layer**: Handles adding/retrieving embeddings from Chroma DB (basic functionality).
