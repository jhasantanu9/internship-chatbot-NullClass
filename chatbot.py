# chatbot.py
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
import re
from sklearn.preprocessing import LabelEncoder
import random
import logging
from typing import Dict, Optional, Any
import warnings

logger = logging.getLogger(__name__)

class ChatbotPredictor:
    def __init__(self, model_path: str, intents_file: str, confidence_threshold: float = 0.5):
        """Initialize the chatbot predictor.
        
        Args:
            model_path: Path to the trained model weights
            intents_file: Path to the JSON file containing intents
            confidence_threshold: Minimum confidence threshold for predictions
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.confidence_threshold = confidence_threshold
        self.label_encoder = LabelEncoder()
        
        # Load intents first so we know num_labels
        self.load_intents(intents_file)
        self.load_model(model_path)
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess input text by converting to lowercase and removing special characters."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return ' '.join(text.split())
    
    def load_intents(self, intents_file: str) -> None:
        """Load intents from JSON file and prepare label encoder."""
        try:
            with open(intents_file, 'r', encoding='utf-8') as f:
                self.intents = json.load(f)['intents']
        except Exception as e:
            logger.error(f"Failed to load intents file: {str(e)}")
            raise
            
        self.responses = {}
        labels = []
        for intent in self.intents:
            self.responses[intent['intent']] = intent['responses']
            labels.extend([intent['intent']] * len(intent['examples']))
        
        self.label_encoder.fit(labels)
        logger.info(f"Loaded {len(self.label_encoder.classes_)} intent classes")
    
    def load_model(self, model_path: str) -> None:
        """Load the BERT model and trained weights."""
        try:
            # Initialize model with correct number of labels
            self.model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=len(self.label_encoder.classes_),
                problem_type="single_label_classification"
            ).to(self.device)
            
            # Load the fine-tuned weights
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                
                # Initialize classifier weights if they don't exist in state_dict
                if 'classifier.weight' not in state_dict:
                    logger.info("Initializing new classifier weights")
                    state_dict['classifier.weight'] = torch.zeros(
                        (len(self.label_encoder.classes_), self.model.config.hidden_size)
                    )
                    state_dict['classifier.bias'] = torch.zeros(len(self.label_encoder.classes_))
                
                # Load the weights
                self.model.load_state_dict(state_dict, strict=False)
            
            self.model.eval()
            logger.info("Model loaded successfully with custom classifier")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def get_response(self, text: str) -> Dict[str, Any]:
        """Generate a response for the input text.
        
        Args:
            text: Input text from user
            
        Returns:
            Dictionary containing response, intent, and confidence score
        """
        try:
            # Preprocess and encode text
            processed_text = self.preprocess_text(text)
            encoding = self.tokenizer(
                processed_text,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Get prediction
            with torch.no_grad():
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                confidence, predicted_class = torch.max(probabilities, dim=1)
                
                confidence_score = confidence.item()
                
                # Check confidence threshold
                if confidence_score < self.confidence_threshold:
                    return {
                        "response": "I'm not quite sure I understand. Could you please rephrase that?",
                        "intent": None,
                        "confidence": confidence_score
                    }
                
                # Get response
                predicted_intent = self.label_encoder.inverse_transform([predicted_class.item()])[0]
                response = random.choice(self.responses[predicted_intent])
                
                return {
                    "response": response,
                    "intent": predicted_intent,
                    "confidence": confidence_score
                }
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "response": "I apologize, but I encountered an error. Please try again.",
                "intent": None,
                "confidence": 0.0
            }