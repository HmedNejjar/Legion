import json
from pathlib import Path

class ExecutionHistory:
    """
    Manages the persistent storage and retrieval of user intent-to-action mappings.
    
    This class tracks how often specific actions are paired with intents, allowing
    the system to learn user preferences over time and suggest "dominant" actions.
    """

    def __init__(self, filepath: str | Path) -> None:
        """
        Initializes the history tracker by loading existing data from a JSON file.

        Args:
            filepath (str | Path): The path to the JSON file where history is stored.
        """
        self.filepath = Path(filepath)
        self.data = self.load()
        
    def load(self) -> dict:
        """
        Loads the history data from the local filesystem.

        Returns:
            dict: A nested dictionary structure {intent_id: {action_id: count}}. 
                  Returns an empty dict if the file does not exist.
        """
        if Path.exists(self.filepath):
            with open(self.filepath, 'r') as f:
                return json.load(f)
        return {}
    
    def save(self) -> None:
        """
        Serializes the current in-memory history data back to the JSON file.
        """
        with open(self.filepath, 'w') as f:
            # Uses an indent of 3 for human-readable JSON
            json.dump(self.data, f, indent=3)
    
    def record(self, intent_id: str, action_id: str) -> None:
        """
        Increments the frequency count for a specific action associated with an intent.

        Args:
            intent_id (str): The unique identifier for the detected intent.
            action_id (str): The unique identifier for the executed action.
        """
        # Ensure the intent and action keys exist before incrementing
        if intent_id not in self.data:
            self.data[intent_id] = {}
        
        if action_id not in self.data[intent_id]:
            self.data[intent_id][action_id] = 0
        
        self.data[intent_id][action_id] += 1
        self.save()  # Persist changes immediately
    
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
        if intent_id not in self.data:
            return None
        
        actions = self.data[intent_id]
        
        # Find the key (action_id) with the highest value (count)
        max_action = max(actions, key=actions.get)
        max_count = actions[max_action]
        
        # Only return the action if it has been used enough times
        return max_action if max_count >= threshold else None
    
    def get_all_actions(self, intent_id: str) -> list | None:
        """
        Returns a list of all actions associated with an intent, sorted by frequency.

        Args:
            intent_id (str): The intent to query.

        Returns:
            list | None: Action IDs sorted from most frequent to least frequent if they exist, otherwise None.
        """
        if intent_id not in self.data:
            return None
        
        actions = self.data[intent_id]
        
        # Sort keys based on their stored counts in descending order
        return sorted(actions.keys(), key=lambda x: actions[x], reverse=True)
    
if __name__ == "__main__":
    eh = ExecutionHistory(r'G:\Projects\Python\Legion\memory\history.json')
    eh.record("app_launch", "launch_app")
    eh.record("app_launch", "launch_app")
    eh.record("app_launch", "close_app")
    eh.record("app_launch", "close_app")
    eh.record("app_launch", "launch_app")
    eh.record("web_interact", "fill_form")
    
    print(eh.get_dominant_action("app_launch", 3))  # Should return "launch_app"
    print(eh.get_all_actions("app_launch"))  # Should return ["launch_app", "close_app"]
    print(eh.get_dominant_action("web_interact"))  # Should return None (threshold not met)