from flask import Flask, request, jsonify, render_template, session
from chatbot import ChatbotPredictor
from analytics.database_handler import ChatbotDB
from language.language_handler import MultilingualChatbotPredictor, LanguageHandler
import logging
import os
import uuid

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session handling

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize chatbot and database
try:
    chatbot = MultilingualChatbotPredictor(
        model_path='model/best_model.pth',
        intents_file='model/intents.json',
        confidence_threshold=0.4
    )
    db = ChatbotDB()
except Exception as e:
    logger.error(f"Failed to initialize: {str(e)}")
    raise

@app.route('/')
def home():
    # Create a new session ID if one doesn't exist
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
            
        message = request.json.get('message', '').strip()
        if not message:
            return jsonify({"error": "Empty message"}), 400
        
        result = chatbot.get_response(message)
        
        # Log the interaction without language information
        db.log_interaction(
            session_id=session.get('session_id', 'unknown'),
            user_message=message,
            bot_response=result["response"],
            intent=result.get("intent"),
            confidence=result.get("confidence", 0.0)
        )
        
        # Check if this response is an exit intent
        is_exit = False
        if result.get("intent") == "Exit":
            is_exit = True
        
        # Add language information to response without storing it
        response_data = {
            "response": result["response"],
            "intent": result["intent"],
            "confidence": result["confidence"],
            "is_exit": is_exit,
            "detected_language": result.get("detected_language", "en"),
            "language_name": result.get("language_name", "english")
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/submit_rating', methods=['POST'])
def submit_rating():
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
            
        rating = request.json.get('rating')
        if rating is None:
            return jsonify({"error": "Missing rating"}), 400
        
        # Log the rating
        db.log_rating(
            session_id=session.get('session_id', 'unknown'),
            rating=rating
        )
        
        return jsonify({"message": "Thank you for your feedback!"})

    except Exception as e:
        logger.error(f"Error processing rating submission: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=False)
    