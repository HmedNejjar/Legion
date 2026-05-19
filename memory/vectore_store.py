import json
import ollama
from datetime import datetime
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

    def __init__(self, db_path: str | Path, llm_model: str) -> None:
        """
        Initializes the VectorStore by setting up the persistent storage directory.

        Args:
            db_path (str | Path): The directory where the ChromaDB data will be stored.
            llm_model (str): LLM for fact extraction and consolidation.
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
         # Metadata collection: tracks fact sources, timestamps, confidence
        self.metadata_path = db_path_obj / "fact_metadata.json"
        self.metadata = self._load_metadata()
        self.llm_model = llm_model
        self.extraction_confidence_threshold = 0.7
       
    def _load_metadata(self) -> dict:
        """
        Loads the metadata dictionary from a JSON file, or initializes it if not present.

        Returns:
            dict: The metadata dictionary mapping fact IDs to their metadata.
        """
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    def _save_metadata(self) -> None:
        """
        Saves the current metadata dictionary to a JSON file.
        """
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_stats(self) -> dict:
        """
        Retrieves statistics about the current state of the vector store.

        Returns:
            dict: A dictionary containing statistics such as total facts and metadata entries.
        """
        total_facts = self.collection.count()
        
        # Calculate average confidence and access patterns
        confidences = [meta['confidence'] for meta in self.metadata.values()]
        accesses = [meta['access_count'] for meta in self.metadata.values()]
        
        return {
            'total_facts': total_facts,
            'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
            'most_accessed': max(accesses) if accesses else 0,
            'metadata_tracked': len(self.metadata),
            'sources': {m['source']: 1 for m in self.metadata.values()}  # Count by source
        }
        
    def extract_store_fact(self, user_input: str, exchange_type: str) -> bool:
        """
        Automatically extracts a user preference/fact from input and stores it.
        
        Args:
            user_input (str): What the user said.
            exchange_type (str): "chat", "action", or "hybrid" - helps filter what to learn.
        
        Returns:
            bool: True if a fact was extracted and stored.
        """
        
        if exchange_type not in ('chat', 'hybrid'):
            return False
        
        prompt = (f"Extract a SHORT user preference or personal fact from this input.\n"
                 f"Return ONLY a single sentence starting with 'user' or 'the user'.\n"
                 f"If no preference is stated, return 'NONE'.\n"
                 f"Examples:\n"
                 f"  - 'the user prefers Spotify'\n"
                 f"  - 'user works in marketing'\n"
                 f"  - 'the user likes lo-fi music'\n\n"
                 f"Input: {user_input}\n"
                 f"Fact: ")

        
        try:
            response = ollama.generate(model=self.llm_model, prompt=prompt, stream= False)['response'].strip()
            
            if response == 'NONE' or not response:
                return False
            
            # Check if fact exists in database
            existing = self.search(response, n_results=1)
            if existing: return False
            
            # Store the fact
            self.add_fact(response, fact_id=None)
            print(f"✓ Learned: {response}")
            return True
        
        except Exception as e:
            print(f"Note: Could not extract fact: {e}")
            return False
            
        
    def add_fact(self, fact: str, fact_id: str | None = None) -> None:
        """
        Adds a new fact (text document) to the vector collection.

        Args:
            fact (str): The text content to be stored.
            fact_id (str | None): A unique identifier for the fact. If None, 
                                  an ID is generated based on the collection count.
        """
        timestamp = datetime.now().timestamp()
        # Auto-generate an ID if one isn't provided
        if fact_id is None:
            fact_id = f'fact_{self.collection.count()}_{int(timestamp)}'
        
        # Add the document to the database
        self.collection.add(ids=[fact_id], documents=[fact])
        
        # Update metadata
        self.metadata[fact_id] = {
            "fact": fact,
            "timestamp": timestamp,
            "source": "Extracted"
        }
        
        self._save_metadata()
        
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
    vs = VectorStore(r'G:\\Projects\\Python\\Legion\\memory\\chromadb', 'qwen3:1.7b')
    
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

    