import getpass as getpass
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

URLS = ["https://en.wikipedia.org/wiki/Artificial_intelligence"]
CHUNK_SIZE = 2400
CHUNK_OVERLAP = 200
TOTAL_RESULTS = 2389
MAX_SIZE = 100
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"},
)
LLM_MODEL_NAME = "gemini-2.0-flash-thinking-exp-01-21"

SYSTEM_PROMPT = """You are an assistant for question-answering tasks. Your name is Dummy. \
Use the following pieces of retrieved context to answer the question. \
Try to keep the answer concise, unless aksed by the user to be eloborated.\
Analyse the question, and provide necessary python code help if necessary, as you will be mainly used for ML research.\
If the context given to you does not have a detailed explanation, give a detailed explanation related to the context and question asked.\
Understand if the question asked is related to AISDB and its usage. If not, just tell you don't know the answer, in a polite manner. \
Ask followup questions to take serve them well after processing answers for each.\
Context:  {context}""" # Example system prompt

def set_envs():
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass(os.getenv("GOOGLE_API_KEY"))
