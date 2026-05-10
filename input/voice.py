import whisper
import webrtcvad
import sounddevice as sd
import numpy as np

class VoiceInput:
    """
    A class to handle real-time voice recording and automated transcription.
    
    Uses WebRTC VAD to detect speech and stop recording automatically during silence,
    then processes the captured audio using OpenAI's Whisper model.
    """

    def __init__(self):
        """
        Initializes the VoiceInput system with Whisper and VAD settings.
        """
        # Load the Whisper model (base is a good balance of speed and accuracy)
        self.model = whisper.load_model("base")
        
        # Initialize WebRTC Voice Activity Detector
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(2)  # Mode 2: Moderate aggressiveness in filtering noise
        
        # Audio configuration constants
        self.sample_rate = 16000
        self.frame_duration = 20  # 20ms frames are required by webrtcvad
        self.frames_per_buffer = int(self.sample_rate * self.frame_duration / 1000)
    
    def listen(self, timeout_seconds: int = 10) -> str | None:
        """
        Records audio from the microphone until silence is detected or timeout occurs.

        Args:
            timeout_seconds (int): Maximum duration to wait for speech before stopping.

        Returns:
            str: The transcribed text if speech is found and processed.
            None: If no audio was recorded or an error occurred.
        """
        print("Listening...")
        
        # Calculate how many 20ms frames fit into the timeout period
        max_frames = int(timeout_seconds * 1000 / self.frame_duration)
        
        # State management for the recording loop
        audio_buffer = []
        silence_counter = 0
        max_silence_frames = 30  # Approx 600ms of silence triggers a stop
        
        def callback(indata, frames, time, status):
            """
            Internal callback processed for every audio block received by sounddevice.
            """
            nonlocal silence_counter
            if status:
                print(f"Audio status: {status}", flush=True)
            
            # WebRTC VAD requires 16-bit signed integer PCM data
            pcm_data = (indata * 32767).astype(np.int16).tobytes()
            
            # Determine if the current frame contains human speech
            is_speech = self.vad.is_speech(pcm_data, self.sample_rate)
            
            if is_speech:
                silence_counter = 0  # Reset silence timer on speech detection
                audio_buffer.append(indata.copy())
                print("", end="", flush=True)
            else:
                silence_counter += 1
                # Keep a small amount of silence in the buffer for natural phrasing
                if silence_counter < max_silence_frames:
                    audio_buffer.append(indata.copy())
                print(".", end="", flush=True)
        
        try:
            # Open the audio stream
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocksize=self.frames_per_buffer,
                callback=callback
            ):
                # Monitor the recording progress
                frame_count = 0
                while frame_count < max_frames and silence_counter < max_silence_frames:
                    sd.sleep(self.frame_duration)
                    frame_count += 1
        except Exception as e:
            print(f"Error recording audio: {e}")
            return None
        
        print("\nProcessing...")
        
        # Verify if we actually captured any audio
        if not audio_buffer:
            print("No audio recorded")
            return None
        
        # Flatten the list of arrays into a single 1D NumPy array for Whisper
        audio_data = np.concatenate(audio_buffer, axis=0).flatten()
        
        # Discard clips that are too short to be meaningful speech
        if len(audio_data) < self.sample_rate * 0.5:  # Under 0.5s
            print("No speech detected")
            return None
        
        # Run inference using Whisper
        # fp16=True is faster but requires a compatible GPU
        result = self.model.transcribe(audio_data, fp16=True)
        
        # Extract text and clean up whitespace
        text = str(result.get("text", "")).strip()
        
        return text
    
if __name__ == "__main__":
    voice_input = VoiceInput()
    text = voice_input.listen()
    if text:
        print(f"Transcribed text: {text}")