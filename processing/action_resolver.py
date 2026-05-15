#type: ignore
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

    def __init__(self, tools_path: str | Path, execution_history: ExecutionHistory) -> None:
        """
        Initializes the resolver by loading tool definitions and execution history.

        Args:
            tools_path (str | Path): Path to the JSON file defining available tools.
            execution_history: ExecutionHistory instance (queries ContextWindow)
        """
        self.tools_path = Path(tools_path)
        self.tools = self.load_tools
        self.history = execution_history
        
    @property   
    def load_tools(self) -> dict:
        """
        Parses the tool configuration file.

        Returns:
            dict: A dictionary mapping tool IDs to their metadata.
        """
        with open(self.tools_path, 'r') as f:
            data = json.load(f)
        # Convert list of tools into a searchable dictionary
        return {tool['id']: tool for tool in data['tools']}
    
    def get_tools_for_intent(self, intent_id: str, threshold: int = 5) -> tuple:            
        """
        Finds the best tool for an intent based on history or category matching.

        Args:
            intent_id (str): The unique identifier for the detected intent.
            threshold (int): Min confirmations before action is considered "dominant"

        Returns:
            tuple: (dominant_tool_dict, options_list)
            
            Cases:
            - (tool_dict, None): Dominant action found, return single tool
            - (None, [tool_dict, ...]): No dominant, return options for user choice
            - (None, []): No matching tools found
        """
        # First, check if the user has a preferred 'dominant' tool['intent_id'] for this intent
        dominant_tool_intent_id = self.history.get_dominant_action(intent_id, threshold)
        
        if dominant_tool_intent_id and dominant_tool_intent_id in self.tools:
            return self.tools[dominant_tool_intent_id], None
            
        # Return all tools that match the intent's id
        matching_tools = [
            tool for tool in self.tools.values() 
            if tool.get('intent_id') == intent_id
        ]
        
        return None, matching_tools
    
if __name__ == '__main__':
    class MockExecutionHistory:
        """Mock for testing"""
        def get_dominant_action(self, intent_id, threshold=5):
            # Simulate: user has confirmed play_music_spotify 6 times
            if intent_id == 'music_play':
                return 'play_music_spotify'
            return None
    
    mock_history = MockExecutionHistory()
    resolver = ActionResolver(TOOLS_PATH, mock_history)
    
    print("=" * 60)
    print("ACTION RESOLVER TESTS")
    print("=" * 60)
    
    # Test 1: Intent with dominant action
    print("\nTest 1: music_play (should have learned default)")
    dominant, options = resolver.get_tools_for_intent('music_play')
    if dominant:
        print(f"✓ Dominant tool found: {dominant['id']}")
        print(f"  Narration: {dominant['narration']}")
    else:
        print(f"Options: {[t['id'] for t in options]}")
    
    # Test 2: Intent without history (should return all options)
    print("\nTest 2: app_launch (no learned default, show options)")
    dominant, options = resolver.get_tools_for_intent('app_launch')
    if dominant:
        print(f"Dominant: {dominant['id']}")
    elif options:
        print(f"✓ Options available: {[t['id'] for t in options]}")
    
    