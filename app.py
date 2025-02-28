import logging
import gradio as gr
import configs.config as config
import services.scraper
import stores.chroma
from llm_setup.llm_setup import LLMService
from caching.lfu import LFUCache
import time 

logger = logging.getLogger()  # Create a logger object
logger.setLevel(logging.INFO)  # Set the logging level to INFO

config.set_envs()  # Set environment variables using the config module

store = stores.chroma.ChromaDB(config.EMBEDDINGS)
service = services.scraper.Service(store)

# Scrape data and get the store vector retriever
service.scrape_and_get_store_vector_retriever(config.URLS)

# Initialize the LLMService with logger, prompt, and store vector retriever
llm_svc = LLMService(logger = logger, system_prompt= config.SYSTEM_PROMPT, web_retriever = store.get_chroma_instance().as_retriever(),llm_model_name = config.LLM_MODEL_NAME)

def respond(user_input,session_hash):
    if user_input == "clear_chat_history_aisdb_override":
        llm_svc.store={}
        return "Memory Cache cleared"
    response = llm_svc.conversational_rag_chain().invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_hash}},
    )["answer"]

    return response

def echo(text, chat_history, request: gr.Request):
    if request:
        session_hash = request.session_hash
        resp = respond(text, session_hash)
        for i in range(len(resp)):
            time.sleep(0.02)
            yield resp[: i + 1]
    else:
        return "No request object received."


def on_reset_button_click():
    llm_svc.store=LFUCache(capacity=50)

if __name__ == '__main__':
    logging.info("Starting AIVIz Bot")

    with gr.Blocks() as demo:
        gr.ChatInterface(fn=echo, type="messages")
        reset_button = gr.Button("Reset Chat Memory Cache")             
        reset_button.click(on_reset_button_click)

    # Launch the interface
    demo.launch()