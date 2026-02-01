import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

class ChromaOperator:
    def __init__(self, host: str = "192.168.0.162", port: int = 9000):
        self.chroma_client = chromadb.HttpClient(host=host, port=port)

    def list_collections(self):
        return self.chroma_client.list_collections()
    
    def get_collection(self, collection_name: str):
        self.collection = self.chroma_client.get_collection(name=collection_name)
        return self.collection
    
    def delete_collection(self, collection_name: str):
        return self.chroma_client.delete_collection(name=collection_name)

if __name__ == "__main__":
    # chromadb_handler = ChromaOperator()

    # print("Available collections:")
    # print(chromadb_handler.list_collections())

    # col_name_get = "research"
    # collection = chromadb_handler.get_collection(collection_name=col_name_get)
    # print(f"\n\n{collection}\n")

    # col_name_del = "test"
    # chromadb_handler.delete_collection(collection_name=col_name_del)
    # print(f"collection {col_name_del} deleted!")

    # query
    # res = collection.query(query_texts=["LangGraph"])
    # print(res)
    client = chromadb.HttpClient(host="192.168.0.162", port=9000, ssl=False)
    embeddings = OllamaEmbeddings(
        base_url="http://192.168.0.162:11434",
        model="embeddinggemma:latest"
    )
    vector_store = Chroma(
        collection_name="research",
        client=client,
        embedding_function=embeddings
    )
    retrieval_res = vector_store.similarity_search("LangGraph", k=2)
    if retrieval_res:
        for doc in retrieval_res:
            print(f"* {doc.page_content} [{doc.metadata}]")
    else:
        print("No retrieval data")
    # print(vector_store.get())
    # collection = client.get_or_create_collection(
    #     name="research",
    #     configuration={
    #         "embedding_function": 
    #     })
    # print(collection.count())
    # res = collection.query(query_texts=["LangGraph benefits"])
    # print(res)
