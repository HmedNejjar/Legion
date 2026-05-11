import ollama
import json

class IntentExtractor:
    """
    Uses an LLM (typically a local SLM) to categorize user intent.
    """
    def __init__(self, model: str = 'qwen3:1.7b') -> None:
        """Initializes the extractor and loads valid intent IDs from config."""
        self.model = model
        self.intents = self.load_intents()
        
    def load_intents(self) -> list:
        """Loads valid intent IDs from the global intents configuration file."""
        # Note: Hardcoded path should be updated for deployment
        with open(r'G:\Projects\Python\Legion\config\intents.json', 'r') as f:
            data = json.load(f)
        return [intent['id'] for intent in data['intents']]
    
    def llm_fallback(self, user_input: str) -> str | None:
        """
        Queries the LLM to pick a single intent ID from the allowed list.

        Returns:
            str: The validated intent ID.
            None: If the LLM produces an invalid or out-of-scope response.
        """
        intent_list = ", ".join(self.intents)
        prompt = (f"You are an intent classifier. Given the user input, pick ONE and ONLY ONE "
                  f"intent from this list: {intent_list}\n User Input: {user_input}\n"
                  f"Respond with ONLY the intent ID, nothing else.")
        
        try:
            response = ollama.generate(self.model, prompt, stream=False)
            intent_id = response['response'].strip().lower()
            
            return intent_id if intent_id in self.intents else None
        except Exception as e:
            print(f"Error in LLM fallback: {e}")
            return None
        
if __name__ == '__main__':
    extractor = IntentExtractor()
    test_input = "Open YouTube Music"
    intent = extractor.llm_fallback(test_input)
    print(f"Extracted intent: {intent}")