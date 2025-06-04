# Life Coach Telegram Bot

A conversational life coach bot that uses Gemini AI for responses and stores conversations in MongoDB for journaling purposes.

## Features

- AI-powered life coaching via Telegram
- Conversation history stored in MongoDB
- Session management with timeout handling
- Context-aware responses using conversation history
- Simple state machine for session tracking

## Requirements

- Python 3.11+
- Telegram bot token
- Google Gemini API key
- MongoDB Atlas connection string

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -e .
```

3. Create a `.env` file with the following variables:
```env
TELEGRAM_BOT_TOKEN=your_bot_token
GEMINI_API_KEY=your_gemini_key
MONGO_USERNAME=your_mongo_username
MONGO_PASSWORD=your_mongo_password
```

## Usage

1. Start the bot:
```bash
python main.py
```

2. Interact with the bot on Telegram:
- `/start` - Begin a new session
- `/bye` - End the current session
- Any other message - Chat with the life coach

## Configuration

The bot can be configured via environment variables:

| Variable | Description |
|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Your Telegram bot token |
| `GEMINI_API_KEY` | Google Gemini API key |
| `MONGO_USERNAME` | MongoDB Atlas username |
| `MONGO_PASSWORD` | MongoDB Atlas password |

## Dependencies

- google-generativeai
- pymongo
- python-dotenv
- python-telegram-bot

*This README will be updated as the project evolves.*
