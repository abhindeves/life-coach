import datetime
from dotenv import load_dotenv
import os
from urllib.parse import quote_plus
from pymongo import MongoClient
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

load_dotenv()

# --- MongoDB Configuration ---
username = quote_plus(os.getenv("MONGO_USERNAME"))
password = quote_plus(os.getenv("MONGO_PASSWORD"))
uri = f"mongodb+srv://{username}:{password}@cluster0.xjcfm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DB_NAME = 'life_coach_db'
COLLECTION_NAME = 'journal_entries'
# ... rest of your code ...

# --- MongoDB Connection ---
try:
    client = MongoClient(uri)
    db = client[DB_NAME]
    journal_collection = db[COLLECTION_NAME]
    print("✅ Successfully connected to MongoDB!")
except Exception as e:
    print(f"❌ Error connecting to MongoDB: {e}")
    exit()
    
# --- Telegram Bot Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Sends a welcome message when the /start command is issued."""
    user = update.effective_user
    welcome_message = (
        f"Hello {user.first_name}! I am your personal life coach bot. "
        "You can chat with me, and I'll help you journal your thoughts. "
        "Everything you write here will be saved as a journal entry."
    )
    await update.message.reply_text(welcome_message)
    print(f"👤 User {user.id} ({user.username}) started the bot.")

async def save_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Saves the user's message to MongoDB and sends an acknowledgment."""
    user = update.effective_user
    message_text = update.message.text
    timestamp = datetime.datetime.now(datetime.timezone.utc)

    # Prepare document to save
    entry = {
        "user_id": user.id,
        "username": user.username,
        "telegram_message_id": update.message.message_id,
        "text": message_text,
        "timestamp": timestamp,
        "source": "telegram_bot"
    }

    try:
        journal_collection.insert_one(entry)
        print(f"📝 Saved message from {user.username}: {message_text}")
        await update.message.reply_text("Noted. Your thoughts have been journaled.")
    except Exception as e:
        print(f"❌ Error saving message to MongoDB: {e}")
        await update.message.reply_text("Sorry, I couldn't save your thoughts right now. Please try again.")


"""Start the bot."""

# Create the Application and pass it your bot's token.
app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

# Register handlers
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, save_message))

# Run the bot
if __name__ == '__main__':
    app.run_polling()