import logging
import uuid
import sqlite3
import os
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv
import sqlite3
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set up Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("models/gemini-1.5-pro-latest",
    generation_config={
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 2048,
    },
    safety_settings=[]
)

app = Flask(__name__)

# Initialize database

def init_db():
    conn = sqlite3.connect('chatbot_analytics.db')
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS queries
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME,
                  query_text TEXT,
                  detected_topic TEXT,
                  actual_topic TEXT)''')

    c.execute('''CREATE TABLE IF NOT EXISTS feedback
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  query_id INTEGER,
                  rating INTEGER,
                  timestamp DATETIME)''')

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

# Logging

def log_query(text, detected_topic, actual_topic=None):
    conn = sqlite3.connect('chatbot_analytics.db')
    c = conn.cursor()
    c.execute('INSERT INTO queries (timestamp, query_text, detected_topic, actual_topic) VALUES (?,?,?,?)',
              (datetime.now(), text, detected_topic, actual_topic))
    query_id = c.lastrowid
    conn.commit()
    conn.close()
    return query_id

def log_feedback(query_id, rating):
    conn = sqlite3.connect('chatbot_analytics.db')
    c = conn.cursor()
    c.execute('INSERT INTO feedback (query_id, rating, timestamp) VALUES (?,?,?)',
              (query_id, rating, datetime.now()))
    conn.commit()
    conn.close()

# Chat route
@app.route('/chat', methods=['POST'])
def chat():
    session_id = request.json.get('session_id', str(uuid.uuid4()))
    user_message = request.json['message'].strip()
    topic = detect_topic(user_message)

    try:
        prompt = f"""Act as a friendly AI assistant. Respond naturally to all types of queries.
        
        Current Query: {user_message}
        Detected Topic: {topic}
        """

        response = model.generate_content(prompt)
        ai_response = response.text.strip()

        query_id = log_query(user_message, topic)

        # Log conversation flow
        conn = sqlite3.connect('chatbot_analytics.db')
        c = conn.cursor()
        c.execute('INSERT INTO conversation_flows (session_id, path, duration) VALUES (?,?,?)',
                  (session_id, topic, 60))
        c.execute('INSERT INTO response_quality (query_id, response_length, timestamp) VALUES (?,?,?)',
                  (query_id, len(ai_response), datetime.now()))
        conn.commit()
        conn.close()

    except Exception as e:
        logging.error(f"Generation Error: {str(e)}")
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

    # Basic Metrics
    c.execute('SELECT COUNT(*) FROM queries')
    total_queries = c.fetchone()[0]
    
    c.execute('SELECT AVG(rating) FROM feedback')
    avg_rating = c.fetchone()[0] or 0

    # Topic Distribution
    c.execute('SELECT detected_topic, COUNT(*) FROM queries GROUP BY detected_topic')
    topic_data = c.fetchall()
    topics_labels = [row[0] for row in topic_data]
    topics_values = [row[1] for row in topic_data]

    # Response Quality
    c.execute('''
        SELECT 
            SUM(CASE WHEN response_length < 50 THEN 1 ELSE 0 END),
            SUM(CASE WHEN response_length BETWEEN 50 AND 150 THEN 1 ELSE 0 END),
            SUM(CASE WHEN response_length > 150 THEN 1 ELSE 0 END)
        FROM response_quality
    ''')
    quality_stats = list(c.fetchone())

    # Conversation Flows
    c.execute('''
        SELECT path, COUNT(*) 
        FROM conversation_flows 
        GROUP BY path 
        ORDER BY COUNT(*) DESC 
        LIMIT 5
    ''')
    flow_data = c.fetchall()
    flow_labels = [row[0] for row in flow_data]
    flow_values = [row[1] for row in flow_data]

    # Time Series Data (fixed string)
    c.execute('''
        SELECT DATE(timestamp), COUNT(*) 
        FROM queries 
        WHERE timestamp >= DATE('now', '-6 days') 
        GROUP BY DATE(timestamp)
    ''')
    time_data = c.fetchall()
    time_labels = [row[0] for row in time_data]
    time_values = [row[1] for row in time_data]

    conn.close()

    return render_template('dashboard.html',
                           total_queries=total_queries,
                           avg_rating=round(avg_rating, 1) if avg_rating else 0,
                           topics_labels=topics_labels,
                           topics_values=topics_values,
                           quality_stats=quality_stats,
                           flow_labels=flow_labels,
                           flow_values=flow_values,
                           time_labels=time_labels,
                           time_values=time_values)
@app.route('/metrics')
def metrics():
    import numpy as np

    conn = sqlite3.connect('chatbot_analytics.db')
    df = pd.read_sql_query(
        "SELECT detected_topic AS predicted, actual_topic AS expected FROM queries WHERE actual_topic IS NOT NULL", conn)
    conn.close()

    if df.empty:
        return render_template("metrics.html", report={}, confusion=[], labels=[], warning="Insufficient data.")

    y_true = df['expected']
    y_pred = df['predicted']

    report = classification_report(y_true, y_pred, output_dict=True)
    matrix = confusion_matrix(y_true, y_pred)
    labels = sorted(list(set(y_true) | set(y_pred)))

    return render_template(
        "metrics.html",
        report=report,
        confusion=matrix.tolist(),
        labels=labels,
        warning=None
    )


@app.route('/feedback/topic', methods=['POST'])
def submit_actual_topic():
    data = request.json
    query_id = data['query_id']
    actual_topic = data['actual_topic']

    conn = sqlite3.connect('chatbot_analytics.db')
    c = conn.cursor()
    c.execute('UPDATE queries SET actual_topic=? WHERE id=?', (actual_topic, query_id))
    conn.commit()
    conn.close()

    return jsonify({'status': 'success'})

@app.route('/')
def home():
    return render_template('chatbot.html')

if __name__ == '__main__':
    app.run(debug=True)
