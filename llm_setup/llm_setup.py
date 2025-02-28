from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from processing.documents import format_documents
from caching.lfu import LFUCache

def _initialize_llm(model_name) -> ChatGoogleGenerativeAI:
    """
    Initializes the LLM instance.
    """
    llm = ChatGoogleGenerativeAI(model= model_name)
    return llm


class LLMService:
    def __init__(self, logger, system_prompt: str, web_retriever: VectorStoreRetriever,cache_capacity: int = 50, llm_model_name = "gemini-2.0-flash-thinking-exp-01-21"):
        self._conversational_rag_chain = None
        self._logger = logger
        self.system_prompt = system_prompt
        self._web_retriever = web_retriever
        self.llm = _initialize_llm(llm_model_name)

        self._initialize_conversational_rag_chain()

        ### Statefully manage chat history ###
        self.store = LFUCache(capacity=cache_capacity)

    def _initialize_conversational_rag_chain(self):
        """
        Initializes the conversational RAG chain.
        """
        ### Contextualize question ###
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )


        history_aware_retriever = create_history_aware_retriever(
        self.llm, self._web_retriever, contextualize_q_prompt)

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain  = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        self._conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        history = self.store.get(session_id)
        if history is None:
            history = ChatMessageHistory()
            self.store.put(session_id, history)
        return history

    def conversational_rag_chain(self):
        """
        Returns the initialized conversational RAG chain.

        Returns:
            The conversational RAG chain instance.
        """
        return self._conversational_rag_chain
    
    def get_llm(self) -> ChatGoogleGenerativeAI:
        """
        Returns the LLM instance.
        """

        if self.llm is None:
            raise Exception("llm is not initialized")

        return self.llm
