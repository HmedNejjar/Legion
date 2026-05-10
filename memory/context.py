from collections import deque

class ContextWindow:
    """
    Manages a sliding window of conversation history for LLM interactions.
    
    This class uses a deque to ensure that only the most recent 'n' turns are 
    retained, preventing context window overflow and managing memory efficiency.
    """

    def __init__(self, max_size: int = 20) -> None:
        """
        Initializes the context window with a fixed maximum size.

        Args:
            max_size (int): The maximum number of conversation turns to store. 
                            Defaults to 20.
        """
        self.max_size = max_size
        # Deque automatically removes the oldest items when maxlen is reached
        self.history = deque(maxlen=max_size)
    
    def add(self, role: str, content: str) -> None:
        """
        Adds a new turn (user or assistant) to the conversation history.

        Args:
            role (str): The role of the speaker (e.g., 'user', 'assistant', 'system').
            content (str): The text content of the message.
        """
        self.history.append({"role": role, "content": content})
    
    def get_all(self) -> list[dict]:
        """
        Retrieves the entire stored conversation history.

        Returns:
            list[dict]: A list of message dictionaries.
        """
        return list(self.history)
    
    def get_last_n(self, n: int) -> list[dict]:
        """
        Retrieves the most recent 'n' messages from the history.

        Args:
            n (int): The number of recent messages to retrieve.

        Returns:
            list[dict]: A list containing the last n messages.
        """
        return list(self.history)[-n:]
    
    def clear(self) -> None:
        """
        Wipes all messages from the current conversation history.
        """
        self.history.clear()
        
    def format_for_prompt(self) -> str:
        """
        Formats the conversation history into a single string for LLM consumption.
        
        Example format:
        USER: hello
        ASSISTANT: hi there

        Returns:
            str: A newline-separated string of formatted dialogue turns.
        """
        lines = []
        for turn in self.history:
            # Uppercase the role for consistent prompt engineering style
            role, content = turn['role'].upper(), turn['content']
            lines.append(f'{role}: {content}')
        
        return '\n'.join(lines)
    
if __name__ == "__main__":
    ctx = ContextWindow(max_size=5)
    ctx.add("user", "play music")
    ctx.add("assistant", "playing spotify")
    ctx.add("user", "skip this")
    ctx.add("assistant", "skipped")
    
    print(ctx.get_all())
    print("\nFormatted:")
    print(ctx.format_for_prompt())