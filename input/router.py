from .voice import VoiceInput
from .text import TextInput
from .camera import CameraImput

class InputRouter:
    """
    Manages multiple input methods and routes requests to the active mode.
    """
    def __init__(self, primary_mode: str = 'text') -> None:
        self.primary_mode = primary_mode
        self.voice = VoiceInput()
        self.text = TextInput()
        self.camera = CameraImput()
        
    def get_input(self) -> str | None:
        if self.primary_mode == 'voice': return self.voice.listen()
        return self.text.listen()
    
    def get_camera_input(self, user_input: str) -> str | None:
        return self.camera.capture_describe(user_input)