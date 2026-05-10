import chromadb
from chromadb.config import Settings
from pathlib import Path

class VectorStore:
    """
    A wrapper class for ChromaDB to manage and search semantic 'user facts'.
    
    This class handles the initialization of a persistent vector database, 
    allowing for the storage of text documents as embeddings and performing 
    similarity searches using modern ChromaDB client patterns.
    """

    def __init__(self, db_path: str | Path) -> None:
        """
        Initializes the VectorStore by setting up the persistent storage directory.

        Args:
            db_path (str | Path): The directory where the ChromaDB data will be stored.
        """
        # Ensure the storage directory exists
        db_path_obj = Path(db_path)
        db_path_obj.mkdir(exist_ok=True, parents=True)
        
        # FIX: Use PersistentClient instead of the deprecated settings pattern
        # This replaces the need for chroma_db_impl="duckdb+parquet"
        self.client = chromadb.PersistentClient(path=str(db_path_obj))
        
        # Initialize or retrieve a collection using cosine similarity for distance
        self.collection = self.client.get_or_create_collection(
            name="user_facts", 
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_fact(self, fact: str, fact_id: str | None = None) -> None:
        """
        Adds a new fact (text document) to the vector collection.

        Args:
            fact (str): The text content to be stored.
            fact_id (str | None): A unique identifier for the fact. If None, 
                                  an ID is generated based on the collection count.
        """
        # Auto-generate an ID if one isn't provided
        if fact_id is None:
            fact_id = f'fact_{self.collection.count()}'
        
        # Add the document to the collection
        self.collection.add(ids=[fact_id], documents=[fact])
        
    def search(self, query: str, n_results: int = 5) -> list:
        """
        Performs a semantic similarity search against the stored facts.

        Args:
            query (str): The search query text.
            n_results (int): The maximum number of similar documents to return.

        Returns:
            list: A list of the most relevant documents found.
        """
        results = self.collection.query(query_texts=[query], n_results=n_results)
        
        # Return only the list of document strings if results were found
        if results['documents'] and len(results['documents']) > 0:
            return results["documents"][0]
        return []
    
    def get_all_facts(self) -> list:
        """
        Retrieves every document currently stored in the collection.

        Returns:
            list: A list containing all document strings.
        """
        all_data = self.collection.get()
        return all_data["documents"] if all_data["documents"] else []
    
    def delete_fact(self, fact_id: str) -> None:
        """
        Removes a specific fact from the collection by its ID.

        Args:
            fact_id (str): The unique identifier of the fact to be deleted.
        """
        self.collection.delete(ids=[fact_id])
        
if __name__ == '__main__':
    print(chromadb.__version__)
    vs = VectorStore(r'G:\\Projects\\Python\\Legion\\memory\\chromadb')
    
    vs.add_fact("user likes lo-fi music")
    vs.add_fact("user works in marketing")
    vs.add_fact("user has a cat named whiskers")
    vs.add_fact("user prefers spotify over youtube music")
    
    # Search
    results = vs.search("music preference", n_results=3)
    print("Search results for 'music preference':")
    for fact in results:
        print(f"  - {fact}")
        
    # Get all
    all_facts = vs.get_all_facts()
    print(f"\nAll facts: {all_facts}")

    