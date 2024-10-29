import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
import json
import numpy as np
from sklearn.model_selection import train_test_split
import random
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class NullClassChatbot:
    def __init__(self, intents_file, confidence_threshold=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = None
        self.label_encoder = LabelEncoder()
        self.confidence_threshold = confidence_threshold
        self.intents = None
        self.responses = {}
        self.load_intents(intents_file)
        
    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and extra spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        return text
    
    def augment_data(self, texts, labels):
        augmented_texts = []
        augmented_labels = []
        
        for text, label in zip(texts, labels):
            # Add original text
            augmented_texts.append(text)
            augmented_labels.append(label)
            
            # Add slightly modified versions
            tokens = word_tokenize(text)
            if len(tokens) > 3:
                # Remove one random word
                removed_word = random.randint(0, len(tokens)-1)
                new_text = ' '.join(tokens[:removed_word] + tokens[removed_word+1:])
                augmented_texts.append(new_text)
                augmented_labels.append(label)
                
                # Shuffle word order slightly
                if len(tokens) > 4:
                    shuffled = tokens.copy()
                    idx = random.randint(0, len(tokens)-2)
                    shuffled[idx], shuffled[idx+1] = shuffled[idx+1], shuffled[idx]
                    augmented_texts.append(' '.join(shuffled))
                    augmented_labels.append(label)
        
        return augmented_texts, augmented_labels
    
    def load_intents(self, intents_file):
        with open(intents_file, 'r') as f:
            self.intents = json.load(f)['intents']
            
        # Create mapping of intents to responses
        for intent in self.intents:
            self.responses[intent['intent']] = intent['responses']
    
    def prepare_data(self):
        texts = []
        labels = []
        
        for intent in self.intents:
            for example in intent['examples']:
                texts.append(self.preprocess_text(example))
                labels.append(intent['intent'])
        
        # Convert labels to numerical format
        numerical_labels = self.label_encoder.fit_transform(labels)
        
        # Augment training data
        augmented_texts, augmented_labels = self.augment_data(texts, numerical_labels)
        
        return augmented_texts, augmented_labels
    
    def train(self, epochs=10, batch_size=16, learning_rate=2e-5):
        print(f"Using device: {self.device}")
        
        # Prepare data
        texts, labels = self.prepare_data()
        
        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Create datasets
        train_dataset = IntentDataset(X_train, y_train, self.tokenizer)
        val_dataset = IntentDataset(X_val, y_val, self.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=len(self.label_encoder.classes_)
        ).to(self.device)
        
        # Initialize optimizer with weight decay
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=2, verbose=True
        )
        
        best_val_accuracy = 0
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
            
            # Validation
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    val_loss += outputs.loss.item()
                    
                    predictions = torch.argmax(outputs.logits, dim=1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
            
            val_accuracy = 100 * correct / total
            avg_val_loss = val_loss / len(val_loader)
            
            print(f'Epoch {epoch + 1}/{epochs}')
            print(f'Average training loss: {total_loss / len(train_loader):.4f}')
            print(f'Average validation loss: {avg_val_loss:.4f}')
            print(f'Validation accuracy: {val_accuracy:.2f}%')
            print('--------------------')
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(self.model.state_dict(), 'best_model.pth')
    
    def predict(self, text):
        if self.model is None:
            raise ValueError("Model hasn't been trained yet!")
        
        # Preprocess input text
        processed_text = self.preprocess_text(text)
        
        self.model.eval()
        encoding = self.tokenizer(
            processed_text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)
            
            if confidence.item() < self.confidence_threshold:
                return None, confidence.item()
            
            predicted_intent = self.label_encoder.inverse_transform([predicted_class.item()])[0]
            return predicted_intent, confidence.item()
    
    def get_response(self, text):
        intent, confidence = self.predict(text)
        
        if intent is None:
            return "I'm not quite sure I understand. Could you please rephrase your question?"
        
        print(f"Detected intent: {intent} (confidence: {confidence:.2f})")  # Debug info
        responses = self.responses[intent]
        return random.choice(responses)

def main():
    # Initialize and train the chatbot
    chatbot = NullClassChatbot('intents.json', confidence_threshold=0.4)
    print("Training the chatbot...")
    chatbot.train(epochs=10, batch_size=8)  
    
    # Interactive loop
    print("\nNullClass Chatbot is ready! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
            
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye!")
            break
        
        try:
            response = chatbot.get_response(user_input)
            print(f"Chatbot: {response}")
        except Exception as e:
            print(f"Chatbot: I'm sorry, I encountered an error. Please try again.")
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()