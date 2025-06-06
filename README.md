# Life Coach Telegram Bot

A conversational life coach bot that uses Gemini AI for responses and stores conversations in MongoDB for journaling purposes, with advanced memory and context capabilities.

## Features

- **AI-powered life coaching** via Telegram using Gemini 1.5 Flash
- **Episodic Memory System** that:
  - Stores conversation insights with context tags and summaries
  - Uses vector embeddings for semantic search
  - Implements Retrieval Augmented Generation (RAG)
- **Advanced State Management**:
  - Finite State Machine (FSM) for session tracking
  - Background tasks for memory processing
  - Automatic timeout handling (30 minutes inactive)
- **Conversation Processing**:
  - Batch processing of completed conversations
  - Automatic memory generation for each session
  - Context-aware responses using past conversations
- **Technical Infrastructure**:
  - Asynchronous processing for better performance
  - Robust error handling and logging
  - Efficient MongoDB queries and indexing

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

## Work in Progress

This project is under active development with ongoing improvements to:
- Memory retrieval and context handling
- Conversation analysis quality
- Performance optimizations
- User experience enhancements

New features are being tested and refined regularly.
