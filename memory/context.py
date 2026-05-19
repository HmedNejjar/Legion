import json
from datetime import datetime
from collections import deque
from pathlib import Path
class ContextWindow:
    """
    Manages a sliding window of conversation history for LLM interactions.
    
    This class uses a deque to ensure that only the most recent 'n' turns are 
    retained, preventing context window overflow and managing memory efficiency.
    """

    def __init__(self, history_filepath: str | Path, context_size: int = 20) -> None:
        """
        Initializes the context window with a fixed maximum size.

        Args:
             history_filepath (str | Path): Path to history.json file.
                                          Created if doesn't exist.
             context_size (int): The maximum number of recent exchanges to retain in memory.
        """        
        self.history_path = Path(history_filepath)
        self.persistent_history = self._load()
        
        self.context_size = context_size
    
    def _load(self) -> list:
        """
        Loads any existing conversation history from the .json file.

        Returns:
            list: A list of message dictionaries loaded from the file, or an empty list if none exist.
        """
        if self.history_path.exists():
            try:
                with open(self.history_path, 'r') as f:
                    return json.load(f)
                
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load history: {e}. Starting fresh.")
                return []
        return []
    
    def _save(self) -> None:
        """
        Persists the entire history to JSON file immediately.
        Called after every exchange.
        """
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.history_path, 'w') as f:
            json.dump(self.persistent_history, f, indent=3)
    

    def save_exchange(self, exchange_type: str, **kwargs) -> None:
        """
        Unified save method that routes to appropriate handler.

        exchange_type: 'action', 'chat', or 'hybrid'
        **kwargs: Depends on type future-proof
        """
        if exchange_type == "action":
            self._save_action(**kwargs)
        elif exchange_type == "chat":
            self._save_chat(**kwargs)
        elif exchange_type == "hybrid":
            self._save_hybrid(**kwargs)
            
        # ===== INTERNAL HANDLERS =====
            
    def _save_action(self, user_input: str, intent: str, action_intent: str, action_tool: str, action_result: str, assistant_narration: str) -> None:
        """
        Records an ACTION exchange to persistent history.
        
        Called internally via save_exchange('action', ...).
 
        Args:
            user_input (str): What the user said
            detected_intent (str): Classifier output (e.g., 'music_play', 'app_launch')
            action_intent (str): The action category (same as detected_intent for pure actions)
            action_tool (str): Specific tool executed (e.g., 'play_music_spotify')
            action_result (str): Outcome of the action (e.g., 'Spotify opened')
            assistant_narration (str): What Legion said about the action
        """
        if not self.history_path:
            return
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "detected_intent": intent,
            "action_intent": action_intent,
            "action_tool": action_tool,
            "action_result": action_result,
            "assistant": assistant_narration,
            "type": "action"
        }
        self.persistent_history.append(entry)
        self._save()
        
    def _save_chat(self, user_input: str, assistant_response: str) -> None:
        """
        Records a CHAT exchange to persistent history.
        
        Called internally via save_exchange('chat', ...).
 
        Args:
            user_input (str): What the user asked
            assistant_response (str): Legion's conversational response
        """
        if not self.history_path:
            return
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "detected_intent": "chat",
            "assistant": assistant_response,
            "type": "chat"
        }
        self.persistent_history.append(entry)
        self._save()
        
    def _save_hybrid(self, user_input: str, intent: str, action_intent: str, action_tool: str, action_result: str, action_narration: str, assistant_response: str) -> None:
        """
        Records a HYBRID exchange (action + chat) to persistent history.
        
        Called internally via save_exchange('hybrid', ...).
 
        Args:
            user_input (str): What the user asked
            detected_intent (str): Classifier output (e.g., 'music_play', 'app_launch')
            action_intent (str): The action category (same as detected_intent for pure actions)
            action_tool (str): Specific tool executed (e.g., 'play_music_spotify')
            action_result (str): Outcome of the action (e.g., 'Spotify opened')
            action_narration (str): What Legion said about the action
            assistant_response (str): Legion's conversational response to the user
        """
        if not self.history_path:
            return
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "detected_intent": intent,
            "type": "hybrid",
            "action": {
                "action_intent": action_intent,
                "action_tool": action_tool,
                "action_result": action_result,
                "narration": action_narration
            },
            "chat": {
                "response": assistant_response
            }
        }
        self.persistent_history.append(entry)
        self._save()
        
    # ===== QUERY METHODS =====
    def get_last_n(self, n: int) -> list:
        """
        Retrieves the last 'n' exchanges from history.

        Args:
            n (int): Number of recent exchanges to retrieve.

        Returns:
            list: A list of the last 'n' exchange dictionaries.
        """
        return self.persistent_history[-n:] if len(self.persistent_history) > 0 else []
    
    def get_all(self) -> list:
        """
        Retrieves the entire conversation history.

        Returns:
            list: A list of all exchange dictionaries in history.
        """
        return self.persistent_history
    
    def get_by_type(self, entry_type: str) -> list:
        """
        Retrieves all exchanges of a specific type (e.g., 'action', 'chat', 'hybrid').

        Args:
            entry_type (str): The type of exchanges to filter by.

        Returns:
            list: A list of exchange dictionaries matching the specified type.
        """
        return [entry for entry in self.persistent_history if entry.get('type') == entry_type]
    
    def get_by_intent(self, intent: str) -> list:
        """
        Retrieves all exchanges associated with a specific detected intent.

        Args:
            intent (str): The detected intent to filter by.

        Returns:
            list: A list of exchange dictionaries where 'detected_intent' matches the specified intent.
        """
        return [entry for entry in self.persistent_history if entry.get('detected_intent') == intent]
    
    # ===== FORMATTING FOR LLM =====
    
    def format_for_prompt(self) -> str:
        recent = self.get_last_n(self.context_size)
        lines = []
        
        for entry in recent:
            timestamp = entry.get('timestamp', 'unknown time')
            user_input = entry.get('user', '')
            detected_intent = entry.get('detected_intent', 'unknown intent')
            entry_type = entry.get('type', '')
            
            lines.append(f"[{timestamp}] USER ({detected_intent}): {user_input}")
            
            if entry_type in ('action','chat'):
                assistant = entry.get('assistant', '')
                lines.append(f"Assistant ({entry_type}): {assistant}")
            
            else:
                action, chat = entry.get('action', {}), entry.get('chat', {})
                action_narration, action_result = action.get('narration', ''), action.get('action_result', '')
                lines.append(f"ASSISTANT (action): {action_narration} → {action_result}")
                
                chat_response = chat.get('response', '')
                lines.append(f"ASSISTANT (chat): {chat_response}")
                
            lines.append("")
        return "\n".join(lines)
    
    def extract_conversation_threads(self, recent_exchanges: list, user_input: str) -> str:
        """
        Extracts a concise conversation thread relevant to the current user input.
        ensuring that only the most pertinent historical exchanges are included, thus optimizing context window usage.

        Args:
            recent_exchanges (str): The formatted recent exchanges from format_for_prompt().
            user_input (str): The current user input for which we want to extract relevant history.
        Returns:
            str: concise conversation thread containing last important exchanges
        """
        lines =  []
        
        # Check if we just did an action
        for exchange in reversed(recent_exchanges):
            if exchange.get('type') in ('action', 'hybrid'):
                tool = exchange.get('action_tool', '')
                outcome = exchange.get('action_result', '')
                lines.append(f'Recently executed: {tool} → {outcome}')
                break
            
        # Check if there was a recent chat response to continue from
        chat_exchanges = [exch for exch in recent_exchanges if exch.get('type') == 'chat']
        
        if chat_exchanges:
            last_chat = chat_exchanges[-1]
            last_response = last_chat.get('assistant', '')
            lines.append(f"Last response what about: {last_response}")
            
        # Detect pronouns in current input for follow-ups
        pronouns = ('it', 'that', 'this', 'those', 'them', 'they', 'these')
        if any(word in user_input.lower() for word in pronouns):
            lines.append("User input is likely referring to recent context.")
        
        return "\n".join(lines) if lines else ''
            
    # ==== UTILITY METHODS =====
    @property
    def count_exchanges(self) -> int:
        """
        Utility method to count total exchanges in history.
        """
        return len(self.persistent_history)
    
    def clear(self) -> None:
        """
        Clears the entire conversation history from memory and file.
        """
        self.persistent_history = []
        self._save()
        
        
if __name__ == "__main__":
    HISTORY_FILE = r'G:\Projects\Python\Legion\memory\history.json'
    
    ctx = ContextWindow(HISTORY_FILE)
    
    ctx.save_exchange('action',
        user_input="play music",
        intent="music_play",
        action_intent="music_play",
        action_tool="play_music_spotify",
        action_result="Spotify opened",
        assistant_narration="Opening Spotify."); print("✓ Saved ACTION exchange")
    
    ctx.save_exchange('chat',
        user_input="tell me a joke",
        assistant_response="Why did the AI go to school? To improve its training data!"); print("✓ Saved CHAT exchange")
    
    ctx.save_exchange('hybrid',
        user_input="play music and tell me what's playing",
        intent="hybrid",
        action_intent="music_play",
        action_tool="play_music_youtube",
        action_result="YouTube Music opened",
        action_narration="Opening YouTube Music.",
        assistant_response="I've opened YouTube Music for you! Currently playing Show by Ado"); print("✓ Saved HYBRID exchange")

    print(f'\ntotal entries: {ctx.count_exchanges}') # Expected: 5
    
    print(f"\nAll ACTION entries: {len(ctx.get_by_type('action'))}") #Expected: 2
    print(f"All CHAT entries: {len(ctx.get_by_type('chat'))}") #Expected: 1
    print(f"All HYBRID entries: {len(ctx.get_by_type('hybrid'))}") #Expected: 1
    
    print(f"All music_play intent entries: {len(ctx.get_by_intent('music_play'))}") #Expected: 2