import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
from flask import Flask, render_template, request, jsonify
import sqlite3
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)
model = genai.GenerativeModel("gemini-pro",
    generation_config={
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 2048,
    },
    safety_settings=[
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
    ]
)
# Initialize database
# Update init_db with new tables
def init_db():
    conn = sqlite3.connect('chatbot_analytics.db')
    c = conn.cursor()
    
    # Existing tables...
    
    c.execute('''CREATE TABLE IF NOT EXISTS response_quality
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  query_id INTEGER,
                  response_length INTEGER,
                  timestamp DATETIME)''')
                 
    c.execute('''CREATE TABLE IF NOT EXISTS conversation_flows
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  session_id TEXT,
                  path TEXT,
                  duration INTEGER)''')
    
    conn.commit()
    conn.close()
init_db()

# Topic detection
def detect_topic(query):
    query = query.lower()
    topics = {
        'technology': {'ai', 'program', 'code', 'algorithm'},
        'weather': {'weather', 'temperature', 'forecast', 'rain'},
        'casual': {'hello', 'hi', 'how are you', 'hey'},
        'health': {'sleep', 'diet', 'exercise', 'medical'},
        'academic': {'explain', 'define', 'theory', 'science'}
    }
    
    for category, keywords in topics.items():
        if any(keyword in query for keyword in keywords):
            return category.capitalize()
    
    return 'General'
def log_query(text, topic):
    conn = sqlite3.connect('chatbot_analytics.db')
    c = conn.cursor()
    c.execute('INSERT INTO queries (timestamp, query_text, detected_topic) VALUES (?,?,?)',
              (datetime.now(), text, topic))
    query_id = c.lastrowid
    conn.commit()
    conn.close()
    return query_id

# Log feedback
def log_feedback(query_id, rating):
    conn = sqlite3.connect('chatbot_analytics.db')
    c = conn.cursor()
    c.execute('INSERT INTO feedback (query_id, rating, timestamp) VALUES (?,?,?)',
              (query_id, rating, datetime.now()))
    conn.commit()
    conn.close()

# AI Chat route
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message'].strip()
    topic = detect_topic(user_message)

    try:
        # Enhanced prompt with conversation handling
        prompt = f"""Act as a friendly AI assistant. Respond naturally to all types of queries.
        
        **Response Rules:**
        1. For greetings: Respond warmly and ask how you can help
        2. For weather queries: Ask for location first
        3. For casual conversation: Keep responses friendly but brief
        4. For complex topics: Provide detailed explanations
        5. If unsure: Ask clarifying questions

        **Examples:**
        User: Hello
        Bot: Hi there! How can I assist you today?

        User: How's the weather?
        Bot: I'd be happy to check! Could you share your location?

        User: Explain quantum physics
        Bot: Quantum physics studies subatomic particles... [detailed explanation]

        **Current Query:** {user_message}
        **Detected Topic:** {topic}
        """

        response = model.generate_content(prompt)
        
        # Handle different response formats
        if response.candidates:
            ai_response = response.candidates[0].content.parts[0].text
            # Log response quality
            conn = sqlite3.connect('chatbot_analytics.db')
            c = conn.cursor()
            c.execute('INSERT INTO response_quality (query_id, response_length, timestamp) VALUES (?,?,?)',
                     (query_id, len(ai_response), datetime.now()))
            conn.commit()
            conn.close()
        else:
            raise ValueError("Empty response from model")

    except Exception as e:
        print(f"Generation Error: {str(e)}")
        # Context-aware fallback
        if topic == 'Casual':
            ai_response = "Hello! How can I help you today?"
        elif topic == 'Weather':
            ai_response = "I need your location to check the weather. Please share a city or ZIP code."
        else:
            ai_response = "Could you please rephrase that? I want to make sure I understand correctly."

    query_id = log_query(user_message, topic)
    return jsonify({'response': ai_response, 'query_id': query_id})
# Dashboard route
@app.route('/dashboard')
def dashboard():
    conn = sqlite3.connect('chatbot_analytics.db')
    c = conn.cursor()

    # Add response quality metrics
    c.execute('''SELECT 
        SUM(CASE WHEN response_length < 50 THEN 1 ELSE 0 END),
        SUM(CASE WHEN response_length BETWEEN 50 AND 150 THEN 1 ELSE 0 END),
        SUM(CASE WHEN response_length > 150 THEN 1 ELSE 0 END)
        FROM response_quality''')
    quality_stats = list(c.fetchone())

    return render_template('dashboard.html',
                         quality_stats=quality_stats,
                         # ... existing params ...
                         )
# Home route
@app.route('/')
def home():
    return render_template('chatbot.html')

if __name__ == '__main__':
    app.run(debug=True)
