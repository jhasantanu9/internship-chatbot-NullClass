# NullClass Customer Service Chatbot

A sophisticated customer service chatbot developed during my internship at [NullClass](https://nullclass.com). The chatbot uses BERT-based natural language understanding to handle customer queries efficiently in multiple languages and includes a Streamlit analytics dashboard for monitoring chatbot performance.

## 🌟 Features

- **Multilingual Support**: Seamlessly handles conversations in 4 languages:
            'en': 'english',
            'es': 'spanish',
            'fr': 'french',
            'de': 'german'
- **Auto Language Detection**: Automatically identifies the user's language and responds accordingly
- **Customer Service Focus**: Trained to handle common customer queries, support requests, and course-related questions
- **Analytics Dashboard**: Built with Streamlit for visualizing chatbot performance metrics
- **Easy Intent Customization**: Modular design allows for easy replacement of intent files and model retraining
- **Real-time Conversation Tracking**: Complete session management and interaction logging

## 📂 Project Structure

```
├── analytics/                       # Analytics components
│   ├── chatbot_analytics.db           # SQLite database for analytics
│   ├── dashboard.py                   # Streamlit analytics dashboard
│   └── database_handler.py            # Database operations handler
├── language/                        # Language processing
│   └── language_handler.py            # Multilingual support implementation
├── model/                           # Model and intent files
│   ├── best_model.pth                 # Trained BERT model (download required)
│   ├── intents.json                   # Intent configuration file
│   └──  MODEL_SETUP_INSTRUCTIONS.md   # Instructions for model download
├── model_artifacts/                 # Training related files
│   ├── bot.ipynb                      # Training notebook
│   ├── bot.py                         # Training script
│   └── intents.json                   # Training intents
├── static/                          # Frontend assets
│   ├── script.js                      # Chat interface interactions
│   ├── style.css                      # Chat interface styling
│   └── favicon.ico                    # Website favicon
├── templates/                       # HTML templates
│   └── index.html                     # Main chat interface
├── app.py                           # Main Flask application
├── chatbot.py                       # Core chatbot logic
└── requirements.txt                 # Project dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- CUDA-capable GPU (optional, for faster inference)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/jhasantanu9/internship-chatbot-NullClass.git
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the pre-trained model:
- Access the provided Google Drive link in `model/MODEL_SETUP_INSTRUCTIONS.md`
- Download and place the model file as `model/best_model.pth`

5. Start the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

### Launch Analytics Dashboard

```bash
cd analytics
streamlit run dashboard.py
```

## 💬 Language Support

The chatbot provides seamless support for multiple languages:

Key language features:
- Automatic language detection
- Language-specific response generation
- Maintains context across language switches
- Confidence scoring for language detection

## 📊 Analytics Dashboard

The Streamlit dashboard provides real-time insights into:
- Language distribution of queries
- User satisfaction per language
- Response accuracy metrics
- Most common queries by language
- Peak usage times
- Average response confidence scores

## 🔍 Core Components

### ChatbotPredictor
- BERT model initialization
- Intent classification
- Response generation

### Language Handler
- Language detection
- Multilingual response processing
- Translation coordination

### Analytics System
- SQLite database for data storage
- Real-time metrics tracking
- Interactive Streamlit dashboard

### Logging and Monitoring

The chatbot includes comprehensive logging features to ensure smooth operation. Logs are formatted as follows:

```
Copy2024-10-30 10:18:00,123 - chatbot - INFO - Message processed: Success
```

### Security Features

To ensure the security of your chatbot, consider the following features:

*   **Session management**: Secure key generation for session management.
*   **Input validation and sanitization**: Validate and sanitize user input to prevent malicious activities.
*   **Error handling and safe error messages**: Handle errors safely and display non-sensitive error messages.


## 💻 API Usage

### Chat Endpoint
```bash
POST /chat
Content-Type: application/json

{
    "message": "Need help with course access"
}
```

Response:
```json
{
    "response": "I'll help you with your course access. Could you please provide your registered email?",
    "intent": "course_access",
    "confidence": 0.95,
    "detected_language": "en",
    "language_name": "english",
    "is_exit": false
}
```

## ⚙️ Configuration

Adjust the chatbot settings in the initialization:

```python
chatbot = MultilingualChatbotPredictor(
    model_path='model/best_model.pth',
    intents_file='model/intents.json',
    confidence_threshold=0.4
)
```
**Customizing and Extending the Chatbot**
=====================================

### Intent File Structure

To customize the chatbot, you can modify the intent file (`model/intents.json`) to add new intents or update existing ones. The intent file should follow this structure:

```json
{
    "intents": [
        {
            "intent": "greeting",
            "examples": [
                "hello",
                "hi",
                "hey there"
            ],
            "responses": [
                "Hello! How can I help you today?",
                "Hi! Welcome to NullClass support!"
            ]
        }
    ]
}
```

Replace the `intent` field with the name of your custom intent and populate the `examples` and `responses` fields with relevant examples and responses.

### Training on New Intents

To train the model using a custom intents file, run the following command:

```bash
python bot.py --intents_file path/to/your/intents.json
```

**Contributing**
--------------

To contribute to the project, follow these steps:

1.  Fork the repository: `git fork https://github.com/username/chatbot.git`
2.  Create a feature branch: `git checkout -b feature/AmazingFeature`
3.  Commit changes: `git commit -m 'Add AmazingFeature'`
4.  Push to branch: `git push origin feature/AmazingFeature`
5.  Open a Pull Request

**License**
---------

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- NullClass.com for the internship opportunity
- BERT model architecture
- Flask and Streamlit frameworks
- PyTorch community

## 📧 Contact

For any queries regarding this project, please contact:
- Email : jhasantantanu9@gmail.com
- Linkedin : https://www.linkedin.com/in/santanu-jha-845510292
  
