class TextInput:
    """
    A simple wrapper for standard terminal text input.
    """

    def __init__(self) -> None:
        """Pass through initialization."""
        pass
    
    def listen(self, timeout: int = 5) -> str | None:
        """
        Waits for user input from the command line.

        Args:
            timeout (int): Currently unused, kept for API consistency.

        Returns:
            str | None: The trimmed user input or None if empty.
        """
        try:
            # Standard blocking terminal input
            text = input('User: ').strip()
            return text if text else None
        except EOFError:
            return None