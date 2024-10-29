import sqlite3
from datetime import datetime
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ChatbotDB:
    def __init__(self, db_path: str = 'analytics/chatbot_analytics.db'):
        """Initialize database connection and create tables if they don't exist."""
        self.db_path = db_path
        self._create_tables()
    
    def _create_tables(self):
        """Create necessary tables for storing chatbot analytics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create interactions table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    user_message TEXT NOT NULL,
                    bot_response TEXT NOT NULL,
                    intent TEXT,
                    confidence REAL,
                    session_id TEXT NOT NULL
                )
                ''')
                
                # Create ratings table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS ratings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    rating INTEGER NOT NULL,
                    timestamp DATETIME NOT NULL
                )
                ''')
                
                conn.commit()
                logger.info("Database tables created successfully")
        
        except Exception as e:
            logger.error(f"Error creating database tables: {str(e)}")
            raise
    
    def log_interaction(self, session_id: str, user_message: str, 
                       bot_response: str, intent: Optional[str], 
                       confidence: float):
        """Log a single interaction between user and chatbot."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO interactions 
                (timestamp, user_message, bot_response, intent, confidence, session_id)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    user_message,
                    bot_response,
                    intent,
                    confidence,
                    session_id
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging interaction: {str(e)}")
    
    def log_rating(self, session_id: str, rating: int):
        """Log a user satisfaction rating."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO ratings (session_id, rating, timestamp)
                VALUES (?, ?, ?)
                ''', (
                    session_id,
                    rating,
                    datetime.now().isoformat()
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging rating: {str(e)}")
    
    def get_analytics_data(self) -> Dict[str, Any]:
        """Retrieve analytics data for dashboard."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get total number of interactions
                cursor.execute('SELECT COUNT(*) FROM interactions')
                total_interactions = cursor.fetchone()[0]
                
                # Get average confidence score
                cursor.execute('SELECT AVG(confidence) FROM interactions WHERE confidence IS NOT NULL')
                avg_confidence = cursor.fetchone()[0] or 0
                
                # Get most common intents
                cursor.execute('''
                SELECT intent, COUNT(*) as count 
                FROM interactions 
                WHERE intent IS NOT NULL 
                GROUP BY intent 
                ORDER BY count DESC 
                LIMIT 10
                ''')
                common_intents = cursor.fetchall()
                
                # Get average rating
                cursor.execute('SELECT AVG(rating) FROM ratings')
                avg_rating = cursor.fetchone()[0] or 0
                
                # Get rating distribution
                cursor.execute('''
                SELECT rating, COUNT(*) as count 
                FROM ratings 
                GROUP BY rating 
                ORDER BY rating
                ''')
                rating_distribution = cursor.fetchall()
                
                # Get interactions over time (last 7 days)
                cursor.execute('''
                SELECT DATE(timestamp) as date, COUNT(*) as count 
                FROM interactions 
                GROUP BY DATE(timestamp) 
                ORDER BY date DESC 
                LIMIT 7
                ''')
                interactions_over_time = cursor.fetchall()
                
                return {
                    'total_interactions': total_interactions,
                    'avg_confidence': round(avg_confidence, 2),
                    'common_intents': common_intents,
                    'avg_rating': round(avg_rating, 2),
                    'rating_distribution': rating_distribution,
                    'interactions_over_time': interactions_over_time
                }
                
        except Exception as e:
            logger.error(f"Error retrieving analytics data: {str(e)}")
            return {}
