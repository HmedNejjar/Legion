from voice.confirmation import ConfirmationHandler

class TierHandler:
    """
    Handles tool execution based on safety tiers (1: Always ask, 2: Learned, 3: Silent).
    """
    def __init__(self, model_path: str, voices_path: str, input_mode: str = 'text') -> None:
        self.input_mode = input_mode
        self.confirmer = ConfirmationHandler(model_path, voices_path, input_mode)
        
    def handle(self, tool: dict, tier: int, learned_default: bool) -> bool:
        """Processes confirmation logic based on the tool's tier."""
        self.confirmer.input_mode = self.input_mode
        confirmation = f"{tool['narration']}, should i proceed sir" 
        if tier == 1: 
            return self.confirmer.ask_confirmation(confirmation)
        elif tier == 2:
            if learned_default:
                return self.confirmer.ask_confirmation(confirmation)
            return False
        elif tier == 3: 
            self.confirmer.tts.speak(confirmation); return True
        return False