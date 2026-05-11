import sys
sys.path.insert(1, r'G:\Projects\Python\Legion')

from voice.tts import TextToSpeech
from input.voice import VoiceInput
from input.text import TextInput

class ConfirmationHandler:
    """
    Asks the user for verbal confirmation (yes/no) before proceeding with an action.
    """
    def __init__(self, model_path: str, voices_path: str, input_mode: str = 'text') -> None:
        self.tts = TextToSpeech(model_path, voices_path)
        self.voice = VoiceInput()
        self.text = TextInput()
        
        self.input_mode = input_mode

    def read(self, text: str) -> None:
        """Prints text to console and speaks it via TTS."""
        print(f"Legion: {text}")
        self.tts.speak(text)
    
    def ask_confirmation(self, narration: str, timeout: int = 5) -> bool:
        """Speaks the action narration and listens for a yes/no response."""
        self.tts.read(narration)
        
        response_text = self.voice.listen(timeout) if self.input_mode == 'voice' else self.text.listen(timeout)
        
        if response_text is None:
            self.tts.speak('I did not hear a response, Aborting Operation')
            return False
        response_text = response_text.lower().strip()
        
        if any(word in response_text for word in ['yes', 'yeah', 'ye', 'yep', 'ok', 'sure', 'go', 'do it', 'sure']): return True
        elif any(word in response_text for word in ['no', 'nah', 'nope', 'cancel', 'stop', 'abort', 'forget it']): return False
        else:
            self.tts.speak('I did not understand your response, kindly say yes or no')
            return self.ask_confirmation(narration, timeout)