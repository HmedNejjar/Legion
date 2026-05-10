import cv2
import ollama
from PIL import Image
import io

class CameraImput:
    """
    Captures frames from the webcam and describes them using a vision model.
    """

    def __init__(self, model: str = 'moondream:latest', camera: int = 0) -> None:
        """
        Initializes the camera and sets the target vision model.
        """
        self.model = model
        self.cap = cv2.VideoCapture(camera)
        
    def capture_describe(self) -> str | None:
        """
        Captures a single frame, displays it, and asks Moondream to describe it.

        Returns:
            str | None: The generated description of the image.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Show local feedback window
        cv2.imshow("Camera Feedback", frame)

        # Convert OpenCV BGR to RGB for processing
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Prepare image as bytes for Ollama
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        image_base64 = img_bytes.read()
        
        try:
            full_response = ""
            # Stream the description from the local vision model
            for chunk in ollama.generate(
                self.model, 
                prompt="What do you see in this image? Describe it briefly.", 
                images=[image_base64], 
                stream=True
            ):
                text = chunk.get('response', '')
                print(text, end='', flush=True)
                full_response += text
            print()
            return full_response.strip()
        except Exception as e:
            print(f"Error querying vision model: {e}")
            return None
    
    def close(self) -> None:
        """Releases the camera hardware resources."""
        self.cap.release()
        
if __name__ == '__main__':
    camera = CameraImput()
    desc = camera.capture_describe()
    print(f"Camera sees: {desc}")
    camera.close()