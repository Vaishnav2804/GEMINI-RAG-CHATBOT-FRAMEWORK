import getpass as getpass
import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

URLS = ["https://aisviz.gitbook.io/documentation", 
        "https://aisviz.gitbook.io/documentation/default-start/quick-start",
        "https://aisviz.gitbook.io/documentation/default-start/sql-database",
        "https://aisviz.gitbook.io/documentation/default-start/ais-hardware",
        "https://aisviz.gitbook.io/documentation/default-start/compile-aisdb",
        "https://aisviz.gitbook.io/documentation/tutorials/database-loading",
        "https://aisviz.gitbook.io/documentation/tutorials/data-querying",
        "https://aisviz.gitbook.io/documentation/tutorials/data-cleaning",
        "https://aisviz.gitbook.io/documentation/tutorials/data-visualization",
        "https://aisviz.gitbook.io/documentation/tutorials/track-interpolation",
        "https://aisviz.gitbook.io/documentation/tutorials/haversine-distance",
        "https://aisviz.gitbook.io/documentation/tutorials/vessel-speed",
        "https://aisviz.gitbook.io/documentation/tutorials/coast-shore-and-ports",
        "https://aisviz.gitbook.io/documentation/tutorials/vessel-metadata",
        "https://aisviz.gitbook.io/documentation/tutorials/using-your-ais-data",
        "https://aisviz.gitbook.io/documentation/tutorials/ais-data-to-csv",
        "https://aisviz.gitbook.io/documentation/tutorials/bathymetric-data",
        "https://aisviz.gitbook.io/documentation/machine-learning/seq2seq-in-pytorch",
        "https://aisviz.gitbook.io/documentation/machine-learning/autoencoders-in-keras",
        "https://aisviz.gitbook.io/documentation/tutorials/weather-data",
        "https://aisviz.cs.dal.ca/AISdb/api/aisdb.html",
        "https://aisviz.cs.dal.ca/AISdb/api/aisdb.database.html",
        "https://aisviz.cs.dal.ca/AISdb/api/aisdb.database.dbconn.html",
        "https://aisviz.cs.dal.ca/AISdb/api/aisdb.database.dbqry.html",
        "https://aisviz.cs.dal.ca/AISdb/api/aisdb.database.decoder.html",
        "https://aisviz.cs.dal.ca/AISdb/api/aisdb.database.sql_query_strings.html",
        "https://aisviz.cs.dal.ca/AISdb/api/aisdb.database.sqlfcn.html",
        "https://aisviz.cs.dal.ca/AISdb/api/aisdb.database.sqlfcn_callbacks.html#",
        "https://aisviz.cs.dal.ca/AISdb/api/aisdb.denoising_encoder.html#",
        "https://aisviz.cs.dal.ca/AISdb/api/aisdb.gis.html",
        "https://aisviz.cs.dal.ca/AISdb/api/aisdb.interp.html",
        "https://aisviz.cs.dal.ca/AISdb/api/aisdb.network_graph.html",
        "https://aisviz.cs.dal.ca/AISdb/api/aisdb.proc_util.html",
        "https://aisviz.cs.dal.ca/AISdb/api/aisdb.receiver.html",
        "https://aisviz.cs.dal.ca/AISdb/api/aisdb.track_gen.html",
        "https://aisviz.cs.dal.ca/AISdb/api/aisdb.track_tools.html",
        "https://aisviz.cs.dal.ca/AISdb/api/aisdb.web_interface.html",
        "https://aisviz.cs.dal.ca/AISdb/api/aisdb.webdata.html",
        "https://aisviz.cs.dal.ca/AISdb/api/aisdb.webdata.bathymetry.html",
        "https://aisviz.cs.dal.ca/AISdb/api/aisdb.webdata.load_raster.html",
        "https://aisviz.cs.dal.ca/AISdb/api/aisdb.webdata.marinetraffic.html",
        "https://aisviz.cs.dal.ca/AISdb/api/aisdb.webdata.shore_dist.html",
        "https://aisviz.cs.dal.ca/AISdb/api/aisdb.wsa.html",
        "https://aisviz.cs.dal.ca/AISdb/api/modules.html",
        "https://aisviz.gitbook.io/documentation/tutorials/decimation-with-aisdb",
        "https://github.com/AISViz/AISdb/blob/master/examples/weather.ipynb",
        "https://github.com/AISViz/AISdb/blob/master/examples/database_creation.py",
        "https://github.com/AISViz/AISdb/blob/master/examples/visualize.py",
        "https://github.com/AISViz/AISdb/blob/master/examples/clean_random_noise.py",
        "https://arxiv.org/html/2310.18948v6",
        "https://arxiv.org/html/2407.08082v1",
        ]
CHUNK_SIZE = 2400
CHUNK_OVERLAP = 200
TOTAL_RESULTS = 2389
MAX_SIZE = 100
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"},
)
LLM_MODEL_NAME = "gemini-2.0-flash-thinking-exp-01-21"

SYSTEM_PROMPT = """You are an assistant for question-answering tasks. Your name is Stormy. \
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
