import sqlite3

conn = sqlite3.connect('chatbot_analytics.db')
c = conn.cursor()

# Add column if it doesn't exist
c.execute("ALTER TABLE queries ADD COLUMN actual_topic TEXT")

conn.commit()
conn.close()
