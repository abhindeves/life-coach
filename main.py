import datetime
import os
import uuid
import asyncio
from enum import Enum, auto
from urllib.parse import quote_plus
from collections import defaultdict
from dotenv import load_dotenv
from pymongo import MongoClient
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
)
import google.generativeai as genai

# --- Load environment variables ---
load_dotenv()

# --- MongoDB Configuration ---
username = quote_plus(os.getenv("MONGO_USERNAME"))
password = quote_plus(os.getenv("MONGO_PASSWORD"))
uri = f"mongodb+srv://{username}:{password}@cluster0.xjcfm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DB_NAME = 'life_coach_db'
COLLECTION_NAME = 'journal_entries'

# --- Gemini API Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# --- MongoDB Connection ---
try:
    client = MongoClient(uri)
    db = client[DB_NAME]
    journal_collection = db[COLLECTION_NAME]
    print("Successfully connected to MongoDB!")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    exit()

# --- FSM State Enum ---
class EpisodeState(Enum):
    IDLE = auto()
    ACTIVE = auto()
    AWAITING_END = auto()
    ENDED = auto()

# --- Session Tracker ---
user_states = defaultdict(dict)
IDLE_TIMEOUT = datetime.timedelta(minutes=30)

# --- Gemini Response Generator ---
def generate_reply_from_gemini(user_input: str) -> str:
    try:
        model = genai.GenerativeModel('gemini-1.5-flash', system_instruction="Act as a life coach. Provide thoughtful and empathetic responses to user queries.")
        response = model.generate_content(user_input)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating reply from Gemini: {e}")
        return "Sorry, I'm having trouble coming up with a response right now."

# --- FSM Transition Handler ---
def transition_state(user_id, username, event):
    now = datetime.datetime.now(datetime.timezone.utc)
    state_info = user_states.get(user_id, {"state": EpisodeState.IDLE, "episode_id": None, "last_activity": now})
    state = state_info["state"]

    if event == "message":
        if state in [EpisodeState.IDLE, EpisodeState.ENDED]:
            new_episode_id = str(uuid.uuid4())
            state_info.update({"state": EpisodeState.ACTIVE, "episode_id": new_episode_id, "last_activity": now})
            journal_collection.insert_one({
                "user_id": user_id,
                "username": username,
                "episode_id": new_episode_id,
                "started_at": now,
                "messages": []
            })
        elif state == EpisodeState.ACTIVE:
            state_info["last_activity"] = now
        elif state == EpisodeState.AWAITING_END:
            state_info.update({"state": EpisodeState.ACTIVE, "last_activity": now})

    elif event == "bye_command":
        if state in [EpisodeState.ACTIVE, EpisodeState.AWAITING_END]:
            state_info["state"] = EpisodeState.ENDED
            journal_collection.update_one({"episode_id": state_info["episode_id"]}, {"$set": {"ended_at": now}})

    elif event == "timeout":
        if state == EpisodeState.ACTIVE:
            state_info["state"] = EpisodeState.AWAITING_END
        elif state == EpisodeState.AWAITING_END:
            state_info["state"] = EpisodeState.ENDED
            journal_collection.update_one({"episode_id": state_info["episode_id"]}, {"$set": {"ended_at": now}})

    user_states[user_id] = state_info

# --- Background Task to Monitor Timeouts ---
async def monitor_idle_users():
    print("Starting user idle monitoring...")
    while True:
        now = datetime.datetime.now(datetime.timezone.utc)
        for user_id, info in list(user_states.items()):
            if info["state"] in [EpisodeState.ACTIVE, EpisodeState.AWAITING_END]:
                if now - info["last_activity"] > IDLE_TIMEOUT:
                    print(f"User {user_id} timed out, transitioning state...")
                    transition_state(user_id, None, "timeout")
        await asyncio.sleep(60)  # Check every minute

# --- Telegram Bot Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await update.message.reply_text(
        f"Hello {user.first_name}! I am your personal life coach bot.\nYou can chat with me, and I'll help you journal your thoughts.\nEach conversation will be saved as a journal episode."
    )

async def bye(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    transition_state(user.id, user.username, "bye_command")
    await update.message.reply_text("Take care! Your journal entry has been saved. See you next time!")

async def save_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    message_text = update.message.text
    now = datetime.datetime.now(datetime.timezone.utc)

    # FSM Transition
    transition_state(user_id, user.username, "message")
    episode_id = user_states[user_id]["episode_id"]

    # Save user message
    journal_collection.update_one(
        {"episode_id": episode_id},
        {"$push": {"messages": {"sender": "user", "text": message_text, "timestamp": now}}}
    )

    # Get Gemini reply
    gemini_reply = generate_reply_from_gemini(message_text)
    await update.message.reply_text(gemini_reply)

    # Save bot reply
    journal_collection.update_one(
        {"episode_id": episode_id},
        {"$push": {"messages": {"sender": "bot", "text": gemini_reply, "timestamp": datetime.datetime.now(datetime.timezone.utc)}}}
    )

# --- Application Setup ---
async def main():
    # Build the application
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Register Handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("bye", bye))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, save_message))
    
    # Start the monitoring task in the background
    monitor_task = asyncio.create_task(monitor_idle_users())
    
    try:
        # Initialize and start the bot
        await app.initialize()
        await app.start()
        
        print("Bot is starting...")
        
        # Start polling for updates
        await app.updater.start_polling()
        
        # Keep the application running
        await asyncio.Event().wait()
        
    except KeyboardInterrupt:
        print("Bot stopped by user")
    finally:
        # Clean shutdown
        monitor_task.cancel()
        await app.stop()
        await app.shutdown()

if __name__ == '__main__':
    asyncio.run(main())