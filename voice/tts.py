from kokoro_onnx import Kokoro
import sounddevice as sd

class TextToSpeech:
    """
    A simple interface for converting text into spoken audio.
    
    Utilizes the Kokoro ONNX model for high-quality, local speech synthesis.
    """

    def __init__(self, model_path: str, voices_path: str, voice: str = 'bm_daniel') -> None:
        """
        Initializes the TTS engine with the specified voice profile.

        Args:
            voice (str): The identifier for the desired voice.
        """
        # Load the ONNX model and the corresponding voice binary
        self.tts = Kokoro(model_path, voices_path)
        self.voice = voice

    def speak(self, text: str) -> bool:
        """
        Synthesizes the provided text and plays it immediately.

        Args:
            text (str): The text string to convert to speech.

        Returns:
            bool: True if the audio played successfully.
        """
        # Handle empty string
        if not text or text.strip() == '':
            return True
        
        # Create the audio data from the text
        audio, sample_rate = self.tts.create(text, self.voice, 1.0)
        
        # Play the audio and block execution until it finishes
        sd.play(audio, samplerate=sample_rate)
        sd.wait()
        
        return True
    
    def read(self, text: str) -> None:
        """Prints text to console and speaks it via TTS."""
        print(f"Legion: {text}")
        self.speak(text)

if __name__ == '__main__':
    tts = TextToSpeech('kokoro-v1.0.onnx', 'voices-v1.0.bin')
    tts.speak("Hello, this is a test of the Kokoro module Text to Speech system. Seems like it's working perfectly fine, isn't it? Nice!")