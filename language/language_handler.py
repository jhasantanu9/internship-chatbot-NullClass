# language_handler.py
from typing import Dict, Optional, Any, List, Tuple
import langdetect
from googletrans import Translator
import json
import logging
from chatbot import ChatbotPredictor 

logger = logging.getLogger(__name__)

class LanguageHandler:
    def __init__(self):
        """Initialize language handling components."""
        self.translator = Translator()
        self.supported_languages = {
            'en': 'english',
            'es': 'spanish',
            'fr': 'french',
            'de': 'german'
        }
        
        # Rest of the LanguageHandler class implementation remains the same
        self.cultural_adaptations = {
            'es': {
                'greeting': {'formal': True, 'use_usted': True},
                'farewell': {'add_diminutive': True},
                'time_format': '24h'
            },
            'fr': {
                'greeting': {'formal': True, 'use_vous': True},
                'farewell': {'formal': True},
                'time_format': '24h'
            },
            'de': {
                'greeting': {'formal': True, 'use_sie': True},
                'farewell': {'formal': True},
                'time_format': '24h'
            }
        }

    # Rest of the class implementation remains unchanged
    def detect_language(self, text: str) -> str:
        """Detect the language of input text."""
        try:
            detected = langdetect.detect(text)
            return detected if detected in self.supported_languages else 'en'
        except:
            return 'en'
    
    def translate_to_english(self, text: str, source_lang: str) -> str:
        """Translate input text to English for processing."""
        if source_lang == 'en':
            return text
        try:
            translation = self.translator.translate(text, src=source_lang, dest='en')
            return translation.text
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return text
            
    def translate_response(self, text: str, target_lang: str) -> str:
        """Translate response to target language with cultural adaptations."""
        if target_lang == 'en':
            return text
            
        try:
            # Apply cultural adaptations before translation
            adapted_text = self.apply_cultural_adaptations(text, target_lang)
            translation = self.translator.translate(adapted_text, src='en', dest=target_lang)
            return translation.text
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return text
            
    def apply_cultural_adaptations(self, text: str, target_lang: str) -> str:
        """Apply cultural-specific adaptations to the response."""
        if target_lang not in self.cultural_adaptations:
            return text
            
        adaptations = self.cultural_adaptations[target_lang]
        adapted_text = text
        
        # Apply formal/informal adjustments
        if 'greeting' in adaptations:
            if adaptations['greeting'].get('formal'):
                # Add formal greetings based on language
                if target_lang == 'es':
                    adapted_text = adapted_text.replace('Hello', 'Buenos días')
                elif target_lang == 'fr':
                    adapted_text = adapted_text.replace('Hello', 'Bonjour')
                elif target_lang == 'de':
                    adapted_text = adapted_text.replace('Hello', 'Guten Tag')
                    
        return adapted_text

class MultilingualChatbotPredictor(ChatbotPredictor):
    def __init__(self, model_path: str, intents_file: str, confidence_threshold: float = 0.5):
        super().__init__(model_path, intents_file, confidence_threshold)
        self.language_handler = LanguageHandler()
        
    def get_response(self, text: str) -> Dict[str, Any]:
        """Generate a response for the input text in the appropriate language."""
        try:
            # Detect input language
            detected_lang = self.language_handler.detect_language(text)
            
            # Translate input to English for processing
            english_text = self.language_handler.translate_to_english(text, detected_lang)
            
            # Get response using parent class method
            result = super().get_response(english_text)
            
            # Translate response back to detected language
            if result["response"]:
                translated_response = self.language_handler.translate_response(
                    result["response"],
                    detected_lang
                )
                result["response"] = translated_response
                
            # Add language information to response
            result["detected_language"] = detected_lang
            result["language_name"] = self.language_handler.supported_languages[detected_lang]
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating multilingual response: {str(e)}")
            return {
                "response": "I apologize, but I encountered an error. Please try again.",
                "intent": None,
                "confidence": 0.0,
                "detected_language": "en",
                "language_name": "english"
            }