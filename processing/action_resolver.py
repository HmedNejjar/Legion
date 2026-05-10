import sys
sys.path.insert(1, r'G:\Projects\Python\Legion')
import json
from pathlib import Path
from memory.execution_history import ExecutionHistory

# Hardcoded paths for local configuration and history storage
HISTORY_PATH = r'G:\Projects\Python\Legion\memory\history.json'
INTENTS_PATH = r'G:\Projects\Python\Legion\config\intents.json'
TOOLS_PATH = r'G:\Projects\Python\Legion\config\tools.json'

class ActionResolver:
    """
    Determines the appropriate tool or set of tools to execute for a specific intent.
    
    It checks historical patterns to see if a specific action has become 
    dominant for an intent, otherwise it falls back to category-based matching.
    """

    def __init__(self, tools_path: str | Path) -> None:
        """
        Initializes the resolver by loading tool definitions and execution history.

        Args:
            tools_path (str | Path): Path to the JSON file defining available tools.
        """
        self.tools = self.load_tools(tools_path)
        self.history = ExecutionHistory(HISTORY_PATH)
        
    def load_tools(self, path: str | Path) -> dict:
        """
        Parses the tool configuration file.

        Args:
            path (str | Path): File path to tools.json.

        Returns:
            dict: A dictionary mapping tool IDs to their metadata.
        """
        with open(path, 'r') as f:
            data = json.load(f)
        # Convert list of tools into a searchable dictionary
        return {tool['id']: tool for tool in data['tools']}
    
    def get_tools_for_intent(self, intent_id: str):            
        """
        Finds the best tool for an intent based on history or category matching.

        Args:
            intent_id (str): The unique identifier for the detected intent.

        Returns:
            tuple: (dominant_tool_dict, None) if a preferred action is found.
            tuple: (None, matching_tools) if multiple options exist.
            tuple: (None, []) if no matching tools are found.
        """
        # First, check if the user has a preferred 'dominant' action for this intent
        dominant_intent_id = self.history.get_dominant_action(intent_id)
        
        if dominant_intent_id and dominant_intent_id in self.tools:
            return self.tools[dominant_intent_id], None
        
        # Fallback: Load intent definitions to find the category
        with open(INTENTS_PATH, 'r') as f:
            intents = json.load(f)
        
        category = None
        for intent in intents['intents']:
            if intent['id'] == intent_id:
                category = intent['category']
                break
        
        if not category:
            return None, []
            
        # Return all tools that match the intent's category
        matching_tools = [
            tool for tool in self.tools.values() 
            if tool['category'] == category
        ]
        
        return None, matching_tools
    
if __name__ == '__main__':
    resolver = ActionResolver(TOOLS_PATH)
    
    # Test with learned default
    tool, options = resolver.get_tools_for_intent("music_play")
    print(f"music_play: {tool}, options: {options}")
    
    # Record some choices to create learned defaults
    resolver.history.record("music_play", "play_music_youtube")
    resolver.history.record("music_play", "play_music_youtube")
    resolver.history.record("music_play", "play_music_youtube")
    resolver.history.record("music_play", "play_music_youtube")
    resolver.history.record("music_play", "play_music_youtube")
    
    # Now it should return learned default
    tool, options = resolver.get_tools_for_intent("music_play")
    print(f"After learning: {tool}, options: {options}")