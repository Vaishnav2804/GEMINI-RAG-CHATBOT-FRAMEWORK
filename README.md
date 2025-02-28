# Simple Gemini RAG Chatbot Framework

Just add your gemini api key in `.env` file, `URLS` to scrape, and modify `System_Prompt` in `/configs/config.py`. That's it, you're good to deploy a RAG-enabled-chatbot with a reasoning Gemini Model(if a reasoning model is used). 

## Deploy on Hugging Face
Follow Hugging Face's documentation to deploy the framework under Chatbot.

# Done, your GEMINI Powered RAG-CHATBOT is READY!

# Developers:

1. To switch to a different LLM, modify `llm_setup/llm_setup.py`.
2. This framework follows a layered architecture:
   - **Service Layer**: Contains all business logic.
   - **Store Layer**: Handles adding/retrieving embeddings from Chroma DB (basic functionality).
