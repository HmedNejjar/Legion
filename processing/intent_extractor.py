import ollama
import json
from pathlib import Path

class IntentExtractor:
    """
    Uses an LLM (typically a local SLM) to categorize user intent.
    """
    def __init__(self, pathfile: str | Path, model: str = 'qwen3:1.7b') -> None:
        """Initializes the extractor and loads valid intent IDs from config."""
        self.model = model
        self.intents_file = Path(pathfile)
        self.intents = self.load_intents()
        
    def load_intents(self) -> list:
        """Loads intent definitions from the configuration file."""
        with open(self.intents_file, 'r') as f:
            data = json.load(f)
        return  data['intents']
    
    def llm_fallback(self, user_input: str) -> str | None:
        """
        Queries the LLM to pick a single intent ID from the allowed list.

        Returns:
            str: The validated intent ID.
            None: If the LLM produces an invalid or out-of-scope response.
        """
        intent_list = ", ".join([f"{intent['id']} ({intent['description']})" for intent in self.intents])
        prompt = (f"You are an intent classifier. Given the user input, pick ONE and ONLY ONE "
                  f"intent from this list: {intent_list}\n User Input: {user_input}\n"
                  f"Respond with ONLY the intent ID, nothing else.")
        
        try:
            response = ollama.generate(self.model, prompt, stream=False)['response'].strip().lower()
            valid_ids = [intent['id'] for intent in self.intents]
            
            for word in response.split():
                if word in valid_ids:
                    return word
            
            return None
        except Exception as e:
            print(f"Error in LLM fallback: {e}")
            return None
        
if __name__ == '__main__':
    INTENTS_FILE = 'G:\\Projects\\Python\\Legion\\config\\intents.json'
    extractor = IntentExtractor(INTENTS_FILE)
    test_input = "Open YouTube Music"
    intent = extractor.llm_fallback(test_input)
    print(f"Extracted intent: {intent}")