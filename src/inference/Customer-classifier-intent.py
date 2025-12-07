import os
import re
import pickle
import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Spell checking with pyspellchecker
try:
    from spellchecker import SpellChecker
    SPELL_CHECK_AVAILABLE = True
except ImportError:
    SPELL_CHECK_AVAILABLE = False
    print("Warning: pyspellchecker not installed. Install with: pip install pyspellchecker")


class IntentClassifier:
    """Simple Intent Classifier for Customer Support with Spell Checking"""
    
    def __init__(
        self, 
        model_path: str,
        tokenizer_path: str,
        label_encoder_path: str,
        config_path: str,
        enable_spell_check: bool = True,
        confidence_threshold: float = 0.5
    ):
        """Initialize the Intent Classifier"""
        self.enable_spell_check = enable_spell_check and SPELL_CHECK_AVAILABLE
        self.confidence_threshold = confidence_threshold
        
        # Initialize spell checker
        self.spell_checker = None
        if self.enable_spell_check:
            try:
                self.spell_checker = SpellChecker()
                print("✓ PySpellChecker initialized")
            except Exception as e:
                print(f"Warning: Could not initialize spell checker - {e}")
                self.enable_spell_check = False
        
        # Load model and preprocessors
        print("Loading model...")
        self.model = load_model(model_path)
        
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        with open(config_path, 'rb') as f:
            self.config = pickle.load(f)
        
        self.max_sequence_length = self.config.get('max_sequence_length', 20)
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Spell checking: {'Enabled' if self.enable_spell_check else 'Disabled'}")
        print(f"✓ Number of intents: {len(self.label_encoder.classes_)}")
    
    def spell_check(self, text: str) -> str:
        """Correct spelling errors in text using pyspellchecker"""
        if not self.enable_spell_check or self.spell_checker is None:
            return text
        
        try:
            # Split text into words
            words = text.split()
            corrected_words = []
            
            for word in words:
                # Keep punctuation and special characters
                if not word.isalpha():
                    corrected_words.append(word)
                else:
                    # Get correction for the word
                    corrected = self.spell_checker.correction(word.lower())
                    if corrected:
                        corrected_words.append(corrected)
                    else:
                        corrected_words.append(word.lower())
            
            return ' '.join(corrected_words)
        except:
            return text
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-z\s]', '', text)
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def predict_intent(
        self, 
        text: str, 
        top_k: int = 3,
        apply_spell_check: bool = True
    ) -> Dict:
        """
        Predict intent from user input
        
        Returns:
            Dictionary with intent, confidence, and other info
        """
        if not text or not text.strip():
            return {
                'intent': 'unknown',
                'confidence': 0.0,
                'original_text': text,
                'corrected_text': text,
                'spell_corrected': False,
                'error': 'Empty input'
            }
        
        original_text = text
        
        # Apply spell checking
        if apply_spell_check:
            text = self.spell_check(text)
        
        corrected_text = text
        spell_corrected = (original_text != corrected_text)
        
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize and pad
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(
            sequence, 
            maxlen=self.max_sequence_length, 
            padding='post'
        )
        
        # Predict
        predictions = self.model.predict(padded_sequence, verbose=0)[0]
        
        # Get top-k predictions
        top_k_indices = np.argsort(predictions)[-top_k:][::-1]
        top_k_predictions = [
            {
                'intent': self.label_encoder.inverse_transform([idx])[0],
                'confidence': float(predictions[idx])
            }
            for idx in top_k_indices
        ]
        
        # Get top prediction
        top_intent = top_k_predictions[0]['intent']
        top_confidence = top_k_predictions[0]['confidence']
        
        return {
            'intent': top_intent,
            'confidence': top_confidence,
            'top_k_predictions': top_k_predictions,
            'is_confident': top_confidence >= self.confidence_threshold,
            'spell_corrected': spell_corrected,
            'original_text': original_text,
            'corrected_text': corrected_text
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict intents for multiple texts"""
        return [self.predict_intent(text) for text in texts]


def load_classifier(
    model_dir: str = '../models',
    enable_spell_check: bool = True,
    confidence_threshold: float = 0.5
) -> IntentClassifier:
    """Load the intent classifier"""
    model_path = os.path.join(model_dir, 'best_lstm_model.h5')
    tokenizer_path = os.path.join(model_dir, 'tokenizer.pkl')
    label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
    config_path = os.path.join(model_dir, 'model_config.pkl')
    
    return IntentClassifier(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        label_encoder_path=label_encoder_path,
        config_path=config_path,
        enable_spell_check=enable_spell_check,
        confidence_threshold=confidence_threshold
    )


def main():
    """Example usage"""
    print("="*60)
    print("Customer Support Intent Classification System")
    print("="*60)
    
    # Get model directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    model_dir = os.path.join(project_root, 'models')
    
    # Load classifier
    try:
        classifier = load_classifier(
            model_dir=model_dir,
            enable_spell_check=True,
            confidence_threshold=0.5
        )
    except Exception as e:
        print(f"Error loading classifier: {e}")
        print("\nMake sure you have:")
        print("1. Trained the model using ModelTraining.ipynb")
        print("2. Saved all required files in the models folder")
        return
    
    # Example queries with typos
    example_queries = [
        "I want to cancle my order",
        "How do I track my pacakge?",
        "Need help with refund",
        "What are your business hours?",
        "I forgot my password"
    ]
    
    print("\n" + "="*60)
    print("Example Predictions")
    print("="*60 + "\n")
    
    for query in example_queries:
        result = classifier.predict_intent(query, top_k=3)
        
        print(f"Original: {result['original_text']}")
        if result['spell_corrected']:
            print(f"Corrected: {result['corrected_text']}")
        print(f"Intent: {result['intent']}")
        print(f"Confidence: {result['confidence']:.2%}")
        
        if len(result['top_k_predictions']) > 1:
            print(f"Alternatives:")
            for pred in result['top_k_predictions'][1:]:
                print(f"  - {pred['intent']}: {pred['confidence']:.2%}")
        print("-" * 60)
    
    # Interactive mode
    print("\n" + "="*60)
    print("Interactive Mode - Type 'quit' to exit")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input("Enter query: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            result = classifier.predict_intent(user_input, top_k=3)
            
            print("\n" + "="*50)
            if result['spell_corrected']:
                print(f"Original: {result['original_text']}")
                print(f"Corrected: {result['corrected_text']}")
            
            print(f"Intent: {result['intent']}")
            print(f"Confidence: {result['confidence']:.2%}")
            
            if not result['is_confident']:
                print("\n⚠ Low confidence - Top alternatives:")
                for pred in result['top_k_predictions'][:3]:
                    print(f"  - {pred['intent']}: {pred['confidence']:.2%}")
            
            print("="*50 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
