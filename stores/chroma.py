from langchain.schema import Document
from langchain_chroma import Chroma


class ChromaDB:
    def __init__(self, embeddings):
        self._persistent_directory = "embeddings"
        self.embeddings = embeddings

        self.chroma = Chroma(persist_directory=self._persistent_directory, embedding_function=self.embeddings)

    def get_chroma_instance(self) -> Chroma:
        return self.chroma

    def store_embeddings(self, documents: list[Document]):
        """
        Store embeddings for the documents using HuggingFace embeddings and Chroma vectorstore.
        """
        self.chroma.add_documents(documents=documents, embeddings=self.embeddings,
                                  persist_directory=self._persistent_directory)
