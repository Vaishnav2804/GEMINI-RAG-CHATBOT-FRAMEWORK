from langchain.schema import Document

from processing.documents import load_documents, format_documents, split_documents
from processing.texts import clean_text


class Service:
    def __init__(self, store):
        self.store = store

    def scrape_and_get_store_vector_retriever(self, urls: list[str]):
        """
        Scrapes website content from fetched schemes and creates a VectorStore retriever.
        """
        documents: list[Document] = []

        for url in urls:
            try:
                website_documents = load_documents(url)
                formatted_content = format_documents(website_documents)
                cleaned_content = clean_text(formatted_content)
                documents.append(Document(page_content=cleaned_content))
            except Exception as e:
                raise Exception(f"Error processing {url}: {e}")

        self.store.store_embeddings(split_documents(documents))
