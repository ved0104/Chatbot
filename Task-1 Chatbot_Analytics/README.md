# AI Chatbot with Analytics Dashboard

## Description

A smart AI chatbot powered by Google's Gemini API with integrated analytics capabilities. This solution provides:

- Natural language conversations
- Real-time usage analytics
- Performance metrics tracking
- User feedback integration
- Topic detection system

## Key Features

🗨️ **AI Chat Interface**
- Gemini-powered responses
- Session management
- Context-aware interactions

📊 **Analytics Dashboard**
- Total query counts
- Topic distribution trends
- User satisfaction ratings
- Response quality metrics
- Conversation flow analysis

🤖 **Smart Tracking**
- Automatic topic detection
- Response length monitoring
- Conversation path mapping
- Accuracy metrics (precision/recall)

## Installation

1. **Prerequisites**
   - Python 3.9+
   - Google Gemini API key

2. **Setup**
   ```bash
   git clone https://github.com/yourusername/chatbot-analytics.git
   cd chatbot-analytics
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configuration**
   Create `.env` file:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

4. **Initialize Database**
   ```bash
   flask init-db
   ```

5. **Run Application**
   ```bash
   flask run
   ```

## Usage

1. **Access Chat Interface**
   - Visit `http://localhost:5000`
   - Start chatting with the AI assistant

2. **Provide Feedback**
   - Rate responses using the star system
   - Correct misclassified topics

3. **View Analytics**
   - Dashboard: `/dashboard`
   - Model Metrics: `/metrics`

## Project Structure

```
.
├── app.py                 # Main application
├── requirements.txt       # Dependencies
├── chatbot_analytics.db   # Database
├── templates/
│   ├── chatbot.html       # Chat interface
│   ├── dashboard.html     # Analytics dashboard
│   └── metrics.html       # Model metrics
└── static/
    └── style.css          # Styling
```

## Database Schema

**Queries Table**
| Column         | Type        | Description                     |
|----------------|-------------|---------------------------------|
| id             | INTEGER     | Unique query identifier         |
| timestamp      | DATETIME    | Interaction time                |
| query_text     | TEXT        | User's original message         |
| detected_topic | TEXT        | Auto-detected conversation topic|
| actual_topic   | TEXT        | User-corrected topic            |

**Additional Tables**
- `feedback`: User ratings
- `response_quality`: Response characteristics
- `conversation_flows`: Dialogue patterns

## Configuration

Edit `.env` file:
```env
GEMINI_API_KEY=your_actual_key_here
DEBUG=True  # Set to False in production
```

## Dependencies

- Flask (Web framework)
- Google Generative AI (AI engine)
- SQLite (Database)
- pandas (Data analysis)
- scikit-learn (Metrics calculation)
- python-dotenv (Environment management)

```bash
pip install -r requirements.txt
```

## License

MIT License - See [LICENSE](LICENSE) for details

## Disclaimer

This application collects anonymous usage data for improving service quality. User messages are processed by third-party AI services. Do not submit sensitive information.

## Acknowledgements

- Google Gemini API team
- Flask community
- Chart.js for visualization components

---

**Happy Chatting!** 🤖📊
