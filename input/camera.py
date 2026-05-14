import cv2
import ollama
from PIL import Image
import base64
import io

class CameraInput:
    """
    Captures frames from the webcam and describes them using a vision model.
    Displays live video feed while analyzing.
    """

    def __init__(self, model: str = 'gemma4:e2b', camera: int = 0) -> None:
        """
        Initializes the camera and sets the target vision model.
        """
        self.model = model
        self.cap = cv2.VideoCapture(camera)
        self.window_name = "Legion Camera Feed"
        self.prompt = "You are an assistant for vision input, Analyze image and answer according to user request, be concise and brief "
        
    def capture_describe(self, user_input: str) -> str | None:
        """
        Displays live video feed and analyzes a frame with the vision model.
        User presses SPACE to capture the current frame for analysis.
 
        Args:
            user_input (str): Context/prompt for the vision model.
 
        Returns:
            str | None: The generated description of the image.
        """
        captured_frame = None
        
        if not self.cap.isOpened():
            return None
        
        print("\n📹 Camera stream active. Press SPACE to capture, ESC to cancel.")
        
        # Display live feed and wait for capture
        while captured_frame is None:
            ret, frame = self.cap.read()
            
            if not ret:
                return None
            
            # Display frame with instructions overlay
            display_frame = frame.copy()
            cv2.putText(display_frame, "Press SPACE to capture | ESC to cancel", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(self.window_name, display_frame)
            
            # Wait for key (1ms timeout for responsive display)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE key
                captured_frame = frame
                print("✓ Frame captured. Analyzing...")
            elif key == 27:  # ESC key
                print("🚫 Camera cancelled.")
                cv2.destroyWindow(self.window_name)
                return None
        
        # Close the live feed window
        cv2.destroyWindow(self.window_name)
        
        # Convert to RGB for processing
        image = Image.fromarray(cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB))
        
        # Prepare image as bytes for Ollama
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        image_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
        
        try:
            full_response = ""
            print("\n🤔 Model analyzing...")
            
            model_response = ollama.generate(
                model=self.model, 
                prompt=f"{self.prompt}. User:{user_input}", 
                images=[image_base64], 
                stream=False
            )
            
            full_response = model_response.response
            return full_response.strip()
        except Exception as e:
            print(f"Error querying vision model: {e}")
            return None
    
    def close(self) -> None:
        """Releases the camera hardware resources."""
        self.cap.release()
        cv2.destroyAllWindows()

        
if __name__ == '__main__':
    camera = CameraInput()
        
    desc = camera.capture_describe(user_input="what do you see here")
    print(f"Final result: {desc}")
    camera.close()