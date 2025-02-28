# Simple Gemini RAG Chatbot Framework

## Setup

1. Add your GEMINI_API key in `configs/.env`.
   
2. In `configs/config.py`:
   - Add `URLs` to scrape (ensure permission from sites).
   - Customize `SYSTEM_PROMPT` for your use case.
   - Customize your `GEMINI_MODEL`.

## Deploy on Hugging Face
Follow Hugging Face's documentation to deploy the framework under Chatbot.

# Done, your GEMINI Powered RAG-CHATBOT is READY!

# Developers:

1. To switch to a different LLM, modify `llm_setup/llm_setup.py`.
2. This framework follows a layered architecture:
   - **Service Layer**: Contains all business logic.
   - **Store Layer**: Handles adding/retrieving embeddings from Chroma DB (basic functionality).
