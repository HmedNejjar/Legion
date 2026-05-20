import sys
sys.path.insert(1, r'G:\Projects\Python\Legion')

import yaml
import ollama
from input.router import InputRouter
from models.classifier import IntentClassifier
from processing.action_resolver import ActionResolver
from processing.tier_handler import TierHandler
from processing.intent_extractor import IntentExtractor
from memory.execution_history import ExecutionHistory
from memory.vector_store import VectorStore
from memory.context import ContextWindow
from voice.tts import TextToSpeech
from input.text import TextInput

# Load global configuration settings
with open(r'G:\Projects\Python\Legion\config\settings.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Hardcoded paths for system resources
PATHS = {
    'CLASSIFIER_PATH' : r'G:\\Projects\\Python\\Legion\\models\\classifier.pkl',
    'TRAINING_DATA_PATH' : r'G:\\Projects\\Python\\Legion\\memory\\classifier_training_data.json',
    'INTENTS_PATH' : r'G:\\Projects\\Python\\Legion\\config\\intents.json',
    'TOOLS_PATH' : r'G:\\Projects\\Python\\Legion\\config\\tools.json',
    'DB_PATH' : r'G:\\Projects\\Python\\Legion\\data\\chromadb',
    'HISTORY_PATH' : r'G:\\Projects\\Python\\Legion\\memory\\history.json',
    'HYBRID_PAIRS_PATH': r'G:\Projects\Python\Legion\config\hybrid_pairs.json',
    'TTS_MODEL_PATH' : r'G:\\Projects\\Python\\Legion\\voice\\kokoro-v1.0.onnx',
    'VOICES_PATH' : r'G:\\Projects\\Python\\Legion\\voice\\voices-v1.0.bin'
}

# System constants derived from config
def load_config(config_path: str) -> dict:
    """Safely loads config with fallback defaults."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"⚠ Config file not found at {config_path}, using defaults")
        return {}
    except yaml.YAMLError as e:
        print(f"⚠ Config parse error: {e}, using defaults")
        return {}

config = load_config(r'G:\Projects\Python\Legion\config\settings.yaml')

CONST = {
    'INPUT_MODE': config.get('session', {}).get('input_mode', 'text'),
    'MIC_ENABLED': config.get('session', {}).get('enable_mic', False),
    'CONTEXT_WINDOW_SIZE': config.get('session',{}).get('context_window_size', 20),
    'LLM_MODEL_NAME': config.get('classifier', {}).get('llm_model', 'qwen3:1.7b'),
    'CONFIDENCE_THRESHOLD': config.get('classifier', {}).get('confidence_threshold', 0.8),
    'CONFIRMATION_THRESHOLD': config.get('learning', {}).get('confirmations_for_default', 5)
}

class Legion:
    """
    The main controller class for the Legion AI assistant.
    
    Orchestrates:
    - Input collection (voice/text/camera)
    - Intent classification (action/chat/hybrid)
    - Action resolution with learned defaults
    - Safety tier handling (confirmation logic)
    - Persistent history (JSON-based)
    """

    def __init__(self, input_mode: str = CONST['INPUT_MODE']) -> None:
        """
        Initializes all internal components and models.

        Args:
            input_mode (str): Determines the primary input source ('text' or 'voice').
        """
        # Communication & Feedback
        self.input_mode = input_mode
        self.tts = TextToSpeech(PATHS['TTS_MODEL_PATH'], PATHS['VOICES_PATH'])
        self.router = InputRouter(primary_mode=input_mode)
        self.text = TextInput()
        
        # Memory & History
        self.context = ContextWindow(PATHS['HISTORY_PATH'], CONST['CONTEXT_WINDOW_SIZE'])
        self.history = ExecutionHistory(context_window=self.context)
        self.vector_store = VectorStore(PATHS['DB_PATH'], CONST['LLM_MODEL_NAME'])
        
        # Processing & Logic
        self.roles = ('assistant', 'user')
        self.exchanges = ('action', 'chat', 'hybrid')
        
        self.classifier = IntentClassifier(PATHS['CLASSIFIER_PATH'], PATHS['TRAINING_DATA_PATH'])
        self.resolver = ActionResolver(tools_path=PATHS['TOOLS_PATH'], execution_history=self.history)
        self.tier_handler = TierHandler(PATHS['TTS_MODEL_PATH'], PATHS['VOICES_PATH'], input_mode)
        self.extractor = IntentExtractor(pathfile=PATHS['INTENTS_PATH'], model=CONST['LLM_MODEL_NAME'])
        
        self.is_running = True

    def process_input(self, user_input: str) -> None:
        """
        Main processing pipeline: classify → resolve → execute.
 
        Logic Flow:
        1. Classify intent (classifier or LLM fallback)
        2. Route based on intent type:
           - ACTION: Resolve to tool, confirm, execute
           - CHAT: Generate conversational response
           - HYBRID: Execute tool + generate chat response
        3. Save exchange to persistent history
        """
        print(f'User: {user_input}')
        
        # ===== STEP 1: HYBRID DETECTION =====
        is_hybrid = self.detect_hybrid_intent(user_input)
        
        # ===== STEP 2: INTENT CLASSIFICATION =====
        intent_id, confidence = self.classifier.predict(user_input)
        
        # Fallback to LLM if classification is uncertain
        if confidence < CONST['CONFIDENCE_THRESHOLD']:
            print(f"Low confidence ({confidence:.2f}). Triggering LLM Fallback...")
            intent_id = self.extractor.llm_fallback(user_input)

            if intent_id:
                # Add to training data for classifier improvement
                print(f"LLM extracted intent: {intent_id}")
                self.classifier.add_to_training_sample(user_input, intent_id)
        if not intent_id:
            response = "I'm sorry, I couldn't understand that command."
            self.tts.read(response)
            self.context.save_exchange(exchange_type=self.exchanges[1], user_input = user_input, assistant_response = response)
            return

        # ===== STEP 3: ROUTE BY INTENT TYPE =====
        # Special handling for check_camera intent
        if intent_id == 'check_camera':
            self._handle_camera(user_input, intent_id)
        
        elif is_hybrid:
            self._handle_hybrid(user_input)
        
        # CHAT INTENT: No tool execution, just respond
        elif intent_id == self.exchanges[1]:
            chat_response = self.generate_chat_response(user_input, intent_id)
            self.tts.read(chat_response)
            self.context.save_exchange(exchange_type=self.exchanges[1], user_input= user_input, assistant_response= chat_response)
            self.vector_store.extract_store_fact(user_input, self.exchanges[1])
            return
        else:
            # ACTION INTENT
            self._handle_action(user_input, intent_id)
    
    def detect_hybrid_intent(self, user_input: str) -> bool:
        """
        Detects if input has BOTH action AND conversational elements.
        
        Args:
            user_input (str): user's query
            
        Returns:
            bool: True if hybrid intent is detected
        """
        
        # Action keywords
        action_keywords = ('turn on', 'play', 'open', 'launch', 'set', 'adjust', 'check', 'start', 'stop','pause', 'skip',
                           'delete', 'move', 'search', 'click', 'scroll', 'fill',
                           'queue', 'volume', 'upload', 'download', 'create', 'check')
        # Conversational keywords
        chat_keywords = ('how', 'what', 'why', 'when', 'who', 'where', 'can you', 'could you', 'would you','tell',
                                   'explain', 'describe', 'what', 'why', 'how',
                                   'suggest', 'recommend', 'summarize', 'analyze',
                                   'and then', 'also', 'and remind', 'what is', 'do you know', 'have you', 'is there', 'are there')
        
        words = user_input.lower().split()
        has_action = any(keyword in user_input.lower() for keyword in action_keywords)
        has_chat = any(keyword in user_input.lower() for keyword in chat_keywords)
        
        return has_action and has_chat
      
    def _handle_action(self, user_input: str, intent_id: str) -> None:   
        """
        Handles pure ACTION intents: resolve tool, confirm, execute.
        """     
        # Resolve intent to tools
        dominant_tool, matching_tools = self.resolver.get_tools_for_intent(intent_id, CONST['CONFIRMATION_THRESHOLD'])
        
        # Check if intent has been learned
        learned_default = self.history.is_learned(intent_id)
        
        #Case A: Dominant tool exists AND it's learned
        if dominant_tool and learned_default:
            tier = dominant_tool.get('tier', 1)
            narration = dominant_tool.get('narration', '')
            outcome = dominant_tool.get('outcome', '')
        
            print(f"Auto-executing (learned): {dominant_tool['id']}")
            self.tts.read(narration)
            
            # TODO: Execute tool here
            
            # Save to history
            self.context.save_exchange(self.exchanges[0], 
                                       user_input= user_input,
                                       intent= intent_id,
                                       action_intent= intent_id,
                                       action_tool= dominant_tool['id'],
                                       action_result= outcome,
                                       assistant_narration= narration)
            
        # Case B: Dominant tool exists but not learned (ask first)
        elif dominant_tool and not learned_default:
            tier = dominant_tool.get('tier', 1)
            confirmed = self.tier_handler.handle(dominant_tool, tier, learned_default)
            
            if confirmed:
                print(f"Executing: {dominant_tool['id']}")
                narration = dominant_tool.get('narration', 'Executing action.')
                self.tts.read(narration)
                
                # Save to history
                self.context.save_exchange(self.exchanges[0], 
                                           user_input= user_input,
                                           intent= intent_id,
                                           action_intent= intent_id,
                                           action_tool= dominant_tool['id'],
                                           action_result= dominant_tool.get('outcome', ''),
                                           assistant_narration= narration)
                # TODO: Implement physical tool execution logic
            
            else: self.tts.read("Aborting Operation, what would you like me to do.")
        
        # Case C: Multiple potential tools found; ask user to choose
        elif matching_tools:
            self.tts.read(f"I found multiple ways to handle that. Which would you prefer?")
            for i, tool in enumerate(matching_tools):
                print(f"  [{i+1}] {tool['id']}: {tool['narration']}")
            
            try:
                choice = int(input(f"Manually select an option (1 - {len(matching_tools)}): ")) - 1
                if not (0 <= choice < len(matching_tools)):
                    self.tts.read("Invalid choice number.")
                    return
                
                selected_tool = matching_tools[choice]
                learned_default = False
            except ValueError:
                self.tts.read("Please enter a valid number.")
                return
            
            tier = selected_tool.get('tier', 1)
            confirmed = self.tier_handler.handle(selected_tool, tier, learned_default)
                        
            if confirmed:
                self.tts.read(selected_tool['outcome'])
                print(f"Executing: {selected_tool['id']}")
                narration = selected_tool.get('narration', '')
                self.tts.read(narration)
                
                self.context.save_exchange(self.exchanges[0], 
                                           user_input= user_input,
                                           intent= intent_id,
                                           action_intent= intent_id,
                                           action_tool= selected_tool['id'],
                                           action_result= selected_tool.get('outcome', ''),
                                             assistant_narration= narration)
                
                #TODO: Implement physical tool execution logic
                
                print(f"  ✓ Recorded: {intent_id} → {selected_tool['id']}")
            
            else: self.tts.read("Aborting Operation, what would you like me to do."); return
                
        else: self.tts.read("No tools found for corresponding request")
        
    def _handle_hybrid(self, user_input: str) -> None:
        """
        Handles HYBRID intents: execute tool + generate chat response.
        """
        # Extract the action intent from the hybrid input using LLM
        action_intent = self.extractor.extract_action_intent_hybrid(user_input)
        if not action_intent:
            self.tts.read("Sorry, I couldn't determine the action part of your request.")
            return
        
        # Resolve to tool
        dominant_tool, options = self.resolver.get_tools_for_intent(action_intent, CONST['CONFIRMATION_THRESHOLD'])
        
        selected_tool = dominant_tool or (options[0] if options else None)
        if not selected_tool:
            self.tts.read("Sorry, I couldn't find a tool to handle that request.")
            return
        
        # Confirmation if tool belongs to tier 1
        tier = selected_tool.get('tier', 1)   
        if not self.tier_handler.handle(selected_tool, tier, dominant_tool is not None):
            self.tts.read("Aborting operation.")
            return     
        
        # Execute action
        action_narration = selected_tool.get('narration', '')
        action_result = selected_tool.get('outcome', '')
        self.tts.read(action_narration)
        
        #TODO: Implement physical tool execution logic
        
        # Generate chat response
        chat_response = self.generate_chat_response(user_input, intent_id=self.exchanges[2])
        self.tts.read(chat_response)
        
        # Save exchange
        self.context.save_exchange(self.exchanges[2],
                                   user_input= user_input,
                                   intent= self.exchanges[2],
                                   action_intent= action_intent,
                                   action_tool= selected_tool['id'],
                                   action_result= action_result,
                                   assistant_narration= f"{action_narration}\n{chat_response}")
        
        self.vector_store.extract_store_fact(user_input, self.exchanges[2])
     

    def _handle_camera(self, user_input: str, intent_id: str) -> None:
        """
        Handles Camera intent
        """
        
        camera_desc = self.router.get_camera_input(user_input)
            
        if camera_desc is not None:
            self.tts.read(camera_desc)
            self.context.save_exchange(exchange_type=self.exchanges[0], 
                                       user_input = user_input,
                                       intent = intent_id,
                                       action_intent = intent_id,
                                       action_tool = 'open_camera',
                                       action_result = "Camera feed analyzed and described.",
                                       assistant_narration = camera_desc)
        else:
            self.tts.read("Sorry, I couldn't analyze the camera feed.")
        try:
            self.router.camera.close()
        except Exception as e:
            print(f"Warning: Error closing camera: {e}")

    def generate_chat_response(self, user_input: str, intent_id: str) -> str:
        """
        Generates a conversational response using LLM.
        
        Builds context by:
            1. Recent exchange history (last N turns)
            2. Last action/tool (what are we discussing?)
            3. Semantic relevance (vector memory facts)
        Args:
            user_input: The user's question/statement
            intent_id: The detected intent ('chat' or from hybrid)
            
        Returns:
            str: LLM-generated response
        """
        # Get recent context for LLM
        context_str = self.context.format_for_prompt()
        
        # get relevent facts from database
        relevent = self.vector_store.search(user_input, n_results=5)
        facts = '\n'.join(relevent) if relevent else ""
        # Find what we're talking about (last action OR last chat topic)
        recent = self.context.get_last_n(5)
        conversation_thread = self.context.extract_conversation_threads(recent, user_input)
        
        prompt = (
            f"You are Legion, a helpful AI assistant. "
            f"What I know about you:\n{facts}\n\n"
            f"Recent conversation:\n{context_str}\n\n"
            f"Conversation thread (what we're discussing):\n{conversation_thread}\n\n"
            f"User: {user_input}\n"
            f"Respond naturally and helpfully, maintaining context about what we're discussing."
        )
        
        try:
            response = ollama.generate(CONST['LLM_MODEL_NAME'], prompt, stream=False)['response'].strip()
            return response
        except Exception as e:
            print(f"Error generating chat response: {e}")
            return "I'm sorry, I couldn't generate a response right now."
        
    def run(self) -> None:
        """Starts the infinite processing loop."""
        self.tts.read("Legion is ready. What would you like me to do sir?")
        
        while self.is_running:
            try:
                # Polling for input from the router
                user_input = self.router.get_input()
                if user_input:
                    self.process_input(user_input)
                else: 
                    print("No input detected...")
            
            except KeyboardInterrupt:
                self.tts.read("Shutting down. Goodbye!")
                self.is_running = False
            except Exception as e:
                print(f"Critical System Error: {e}")
                self.tts.read("An error occurred. Shutting Down.")
                self.is_running = False
        
        self.cleanup()

    def cleanup(self) -> None:
        """Performs teardown tasks like saving history."""
        print("Saving session...")
        print(f"Total history entries: {self.context.count_exchanges}")
        print("✓ Session saved")
        
        # Consolidate vector facts
        print("Consolidating learned facts...")
        self.vector_store.consolidate_facts()

        # Show stats
        stats = self.vector_store.get_stats()
        print(f"Vector DB: {stats['total_facts']} facts")
if __name__ == '__main__':
    session = Legion(input_mode="text")
    session.run()