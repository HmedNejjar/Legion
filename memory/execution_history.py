#type: ignore
from .context import ContextWindow

class ExecutionHistory:
    """
    Derives learned defaults by querying ContextWindow history.
    
    Reads from ContextWindow JSON and counts
    action_intent + action_tool occurrences to determine dominants.
    """

    def __init__(self, context_window: ContextWindow, threshold: int = 5) -> None:
        """
        Initializes the history tracker by loading existing data from a JSON file.

        Args:
            context_window (ContextWindow): The path to the JSON file where history is stored.
            threshold (int): Minimum number of times an action must have been performed 
                             to be considered "dominant" for an intent.
        """
        self.context = context_window
        self.threshold = threshold
        self.learned_defaults = self._load_learned_defaults
    
    @property  
    def _load_learned_defaults(self):
        """
        Precomputes dominant actions for all intents based on the current history.

        Returns:
            dict: A mapping of intent_id to its dominant action_tool (if any).
        """
        learned = {}
        all_intents = set()
        
        # Get all intents from history
        for entry in self.context.get_all():
            intent = entry.get('detected_intent')
            
            if intent:
                all_intents.add(intent)
                
        # Check if the intent is the dominant one
        for intent in all_intents:
            dominant = self.get_dominant_action(intent, self.threshold)
            if dominant:
                learned[intent] = dominant
                
        return learned
    
    def is_learned(self, intent_id: str) -> bool:
        """Check if an intent has a learned default"""
        return intent_id in self.learned_defaults
    
    def get_learned_action(self, intent_id: str) -> str | None:
        """Get the learned default action for an intent, if it exists"""
        return self.learned_defaults.get(intent_id)
    
    def get_dominant_action(self, intent_id: str, threshold: int = 5) -> str | None:
        """
        Retrieves the action most frequently associated with a given intent, 
        provided it meets a minimum usage threshold.

        Args:
            intent_id (str): The intent to query.
            threshold (int): Minimum number of times the action must have been 
                             performed to be considered "dominant".

        Returns:
            str | None: The action_id if a dominant action exists, otherwise None.
        """
        entries = self.context.get_by_intent(intent_id)
        
        if not entries: return None
        
        # Count occurrences of each action for the given intent
        action_counts = {}
        for entry in entries:
            action_tool = None
            
            if entry.get('type') == 'action':
                action_tool = entry.get('action_tool')
            
            elif entry.get('type') == 'hybrid':
                action_tool = entry.get('action', {}).get('action_tool')
                
            if not action_tool:
                return None
            
            action_counts[action_tool] = action_counts.get(action_tool, 0) + 1
        
        # Identify the most frequently used action
        dominant_action, count = max(action_counts.items(), key=lambda x: x[1])
        return dominant_action if count >= threshold else None
    
    def get_all_actions(self, intent_id: str) -> list | None:
        """
        Retrieves a list of all actions associated with a given intent.

        Args:
            intent_id (str): The intent to query.

        Returns:
            list | None: A list of action_ids if any exist, otherwise None.
        """
        entries = self.context.get_by_intent(intent_id)
        
        if not entries:
            return None
        
        # Count all actions
        action_counts = {}
        
        for entry in entries:
            action_tool = None
            
            if entry.get('type') == 'action':
                action_tool = entry.get('action_tool')
            elif entry.get('type') == 'hybrid':
                action_tool = entry.get('action', {}).get('action_tool')
            
            if action_tool:
                action_counts[action_tool] = action_counts.get(action_tool, 0) + 1
        
        if not action_counts:
            return None
        
        # Sort by count (descending)
        return sorted(action_counts.keys(), key=lambda x: action_counts[x], reverse=True)          
    
    def get_action_count(self, intent_id: str, action_tool: str) -> int:
        """
        Retrieves the count of how many times a specific action has been performed for a given intent.

        Args:
            intent_id (str): The intent to query.
            action_tool (str): The specific action/tool to check. 
        Returns:
            int: Number of times this action was used (0 if never)
        """
        entries = self.context.get_by_intent(intent_id)
        
        count = 0
        for entry in entries:
            action_tool_in_entry = None
            
            if entry.get('type') == 'action':
                action_tool_in_entry = entry.get('action_tool')
            elif entry.get('type') == 'hybrid':
                action_tool_in_entry = entry.get('action', {}).get('action_tool')
            
            if action_tool_in_entry == action_tool:
                count += 1
        
        return count
    def is_dominant(self, intent_id: str, action_tool: str, threshold: int = 5) -> bool:
        """
        Checks if a specific action is dominant for a given intent based on the threshold.

        Args:
            intent_id (str): The intent to query.
            action_tool (str): The specific action/tool to check.
            threshold (int): Minimum number of times the action must have been 
                             performed to be considered "dominant".

        Returns:
            bool: True if the action is dominant, False otherwise.
        """
        return action_tool == self.get_dominant_action(intent_id, threshold)
    
    def get_stats(self, intent_id: str) -> dict:
        """
        Returns detailed statistics about an intent's action history.
        
        Useful for debugging or understanding user behavior.
        
        Args:
            intent_id: The intent to analyze
            
        Returns:
            dict: {
                'intent': intent_id,
                'total_actions': int,
                'actions': {action_tool: count, ...},
                'dominant_action': str or None,
                'dominant_count': int or None
            }
        """
        entries = self.context.get_by_intent(intent_id)
        
        if not entries:
            return {
                'intent': intent_id,
                'total_actions': 0,
                'actions': {},
                'dominant_action': None,
                'dominant_count': None
            }
        
        # Count all actions
        action_counts = {}
        
        for entry in entries:
            action_tool = None
            
            if entry.get('type') == 'action':
                action_tool = entry.get('action_tool')
            elif entry.get('type') == 'hybrid':
                action_tool = entry.get('action', {}).get('action_tool')
            
            if action_tool:
                action_counts[action_tool] = action_counts.get(action_tool, 0) + 1
        
        total = sum(action_counts.values())
        dominant = self.get_dominant_action(intent_id, threshold=5)
        dominant_count = action_counts.get(dominant) if dominant else None
        
        return {
            'intent': intent_id,
            'total_actions': total,
            'actions': action_counts,
            'dominant_action': dominant,
            'dominant_count': dominant_count
        }

if __name__ == "__main__":
    class MockContextWindow:
        """Mock for testing"""
        def __init__(self):
            self.persistent_history = [
                {
                    "type": "action",
                    "detected_intent": "music_play",
                    "action_intent": "music_play",
                    "action_tool": "play_music_spotify"
                },
                {
                    "type": "action",
                    "detected_intent": "music_play",
                    "action_intent": "music_play",
                    "action_tool": "play_music_spotify"
                },
                {
                    "type": "action",
                    "detected_intent": "music_play",
                    "action_intent": "music_play",
                    "action_tool": "play_music_spotify"
                },
                {
                    "type": "action",
                    "detected_intent": "music_play",
                    "action_intent": "music_play",
                    "action_tool": "play_music_spotify"
                },
                {
                    "type": "action",
                    "detected_intent": "music_play",
                    "action_intent": "music_play",
                    "action_tool": "play_music_spotify"
                },
                {
                    "type": "action",
                    "detected_intent": "music_play",
                    "action_intent": "music_play",
                    "action_tool": "play_music_youtube"
                },
                {
                    "type": "hybrid",
                    "detected_intent": "hybrid",
                    "action": {
                        "action_intent": "music_play",
                        "action_tool": "play_music_spotify"
                    }
                }
            ]
        
        def get_by_intent(self, intent):
            return [e for e in self.persistent_history if e.get('detected_intent') == intent]
        
    # Test
    mock_context = MockContextWindow()
    history = ExecutionHistory(mock_context)
    
    print("=" * 60)
    print("EXECUTION HISTORY TESTS")
    print("=" * 60)
    
    dominant = history.get_dominant_action('music_play', threshold=5)
    print(f"\nDominant action for 'music_play': {dominant}")
    
    all_actions = history.get_all_actions('music_play')
    print(f"All actions for 'music_play': {all_actions}")
    
    spotify_count = history.get_action_count('music_play', 'play_music_spotify')
    print(f"Times Spotify was used: {spotify_count}")
    
    is_dom = history.is_dominant('music_play', 'play_music_spotify', threshold=5)
    print(f"Is Spotify dominant? {is_dom}")
    
    stats = history.get_stats('music_play')
    print(f"\nStats for 'music_play':")
    print(f"  Total actions: {stats['total_actions']}")
    print(f"  Breakdown: {stats['actions']}")
    print(f"  Dominant: {stats['dominant_action']} ({stats['dominant_count']} times)")