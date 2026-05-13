from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import pickle
from pathlib import Path

class IntentClassifier:
    """
    A text classification engine used to map natural language inputs to specific intents.
    
    This class utilizes a TF-IDF vectorizer for feature extraction and a 
    Multinomial Naive Bayes classifier for prediction, providing a lightweight 
    and efficient way to categorize user commands.
    """  

    def __init__(self, model_path: str | Path, training_path: str | Path) -> None:
        """
        Initializes the classifier and attempts to load an existing model from disk.

        Args:
            model_path (str | Path): Path to the saved pickle file containing 
                                     the trained model and vectorizer.
        """
        self.model_path = Path(model_path)
        self.training_path = Path(self.training_path)
        self.training_data = self.load_data()
        self.retrain_threshold = 5
        
        # Initialize the TF-IDF vectorizer to convert text to numerical features
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
        
        # Multinomial Naive Bayes is well-suited for discrete text features
        self.classifier = MultinomialNB()
        self.trained = False
        
        # Load the model if a pre-trained file already exists
        if Path.exists(self.model_path):
            self.load()
                
    def load_data(self) -> dict:
        with open(self.training_path, 'r') as f:
            return json.load(f)
        
    def save_training_data(self) -> None:
        with open(self.training_path, 'w') as f:
            json.dump(self.training_data, f, indent=3)
        
    def train(self) -> None:
        """
        Trains the classifier on a provided dictionary of intents and examples.

        """
        text, labels = [], []
        
        # Unpack the training dictionary into parallel lists of text and labels
        for intent_id, examples in (self.training_data if self.training_data else {}).items():
            for example in examples:
                text.append(example)
                labels.append(intent_id)
        
        # Fit the vectorizer to the vocabulary and transform the text to a feature matrix
        X = self.vectorizer.fit_transform(text)
        
        # Train the Naive Bayes model on the feature matrix
        self.classifier.fit(X, labels)
        self.trained = True
        
        # Persist the newly trained model to disk
        self.save()
        
    def predict(self, text: str) -> tuple:
        """
        Predicts the intent of a given text string and returns the confidence score.

        Args:
            text (str): The raw input string from the user.

        Returns:
            tuple: (predicted_intent_id, confidence_score). 
                   Returns (None, 0.0) if the model is not yet trained.
        """
        if not self.trained:
            return None, 0.0
        
        # Transform the input text using the same vectorizer used during training
        X = self.vectorizer.transform([text])
        
        # Predict the most likely category
        intent = self.classifier.predict(X)[0]
        
        # Calculate the probability of the prediction to provide a confidence score
        confidence = self.classifier.predict_proba(X).max()
        
        return intent, confidence
    
    def save(self) -> None:
        """
        Serializes and saves the vectorizer and classifier to a binary file.
        """
        # Ensure the directory exists before saving
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.model_path, 'wb') as f:
            # Bundle both components into a single dictionary for consistency
            pickle.dump({
                'vectorizer': self.vectorizer,
                'classifier': self.classifier
            }, f)
    
    def load(self) -> None:
        """
        Loads the vectorizer and classifier from a binary file.
        """
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.classifier = data['classifier']
            self.trained = True
    
    def add_to_training_sample(self, text: str, intent_id: str) -> None:
        """Adds sample and retrains if threshold reached."""
        if intent_id not in self.training_data:
            self.training_data[intent_id] = []

        self.training_data[intent_id].append(text)
        self.save_training_data()

        if len(self.training_data[intent_id]) % self.retrain_threshold == 0:
            self.retrain()

    def retrain(self) -> None:
        """Retrains classifier."""
        self.train()
        print("✓ Classifier retrained")
        
if __name__ == '__main__':
    MODEL_PATH = Path(r'G:\Projects\Python\Legion\models\classifier.pkl')
    TRAINING_DATA_PATH = Path(r'G:\Projects\Python\Legion\config\training_data.json')
    
    # Train
    classifier = IntentClassifier(MODEL_PATH, TRAINING_DATA_PATH)
    classifier.train()
    print("✓ Classifier trained and saved")
    
    # Test
    print("\nTesting predictions:")
    test_inputs = [
        "play me some justin Bieber",
        "check my github page",
        "delete this file",
        "scroll down",
        "go to youtube",
        "skip to next",
        "move file to drive D:",
        "click the button"
    ]
    
    for text in test_inputs:
        intent, confidence = classifier.predict(text)
        print(f"  '{text}' -> {intent} ({confidence:.2f})")