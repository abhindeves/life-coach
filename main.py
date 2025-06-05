import datetime
import os
import uuid
import asyncio
import logging
import json
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Load environment variables ---
load_dotenv()

# --- MongoDB Configuration ---
username = quote_plus(os.getenv("MONGO_USERNAME"))
password = quote_plus(os.getenv("MONGO_PASSWORD"))
uri = f"mongodb+srv://{username}:{password}@cluster0.xjcfm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DB_NAME = 'life_coach_db'
COLLECTION_NAME = 'journal_entries'
MEMORY_COLLECTION_NAME = 'episodic_memories'

# --- Gemini API Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# --- MongoDB Connection ---
try:
    client = MongoClient(uri)
    db = client[DB_NAME]
    journal_collection = db[COLLECTION_NAME]
    memory_collection = db[MEMORY_COLLECTION_NAME]
    logger.info("Successfully connected to MongoDB!")
except Exception as e:
    logger.error(f"Error connecting to MongoDB: {e}")
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

# --- Episodic Memory Prompt Template ---
life_coach_prompt_template = """
You are analyzing life coaching conversations to create memory reflections that help guide future sessions. Your job is to extract helpful insights that reflect the user's core issues, goals, and what coaching strategies worked or didn't.
Review the conversation and create a memory reflection with these rules:
1. Only extract context_tags that are *explicitly mentioned or clearly implied* in the conversation — do not invent or generalize them.
2. Be extremely concise — each field should be one clear, actionable sentence.
3. Focus on practical, repeatable coaching insights that would help in similar future sessions.
4. Use terms or phrases directly from the conversation for context_tags when possible.
Output only a valid JSON object in this format:
{{
    "context_tags": [              // 2–4 keywords or phrases lifted directly from the user's concerns or goals
        string,
        ...
    ],
    "conversation_summary": string, // One sentence summarizing the user's main concern or progress
    "what_worked": string,         // The most effective thing said or done in the conversation
    "what_to_avoid": string        // The least effective or unhelpful moment in the conversation
}}
Guidelines:
- Pull actual keywords or short phrases from the conversation for context_tags, e.g., ["procrastination", "gym_routine", "lack_of_motivation"]
- Don't use vague tags like ["life_coaching", "feelings", "helpful_talk"]
- Use direct user quotes or paraphrased terms from the session
Here is the conversation:
{conversation}
"""

# --- Episodic Memory Functions ---
def format_conversation_for_memory(messages: list) -> str:
    """Format conversation messages into a readable string for memory processing"""
    formatted_conversation = ""
    for msg in messages:
        sender = "User" if msg["sender"] == "user" else "Coach"
        timestamp = msg["timestamp"].strftime("%H:%M")
        formatted_conversation += f"[{timestamp}] {sender}: {msg['text']}\n"
    return formatted_conversation

def create_episodic_memory(episode_data: dict) -> dict:
    """Create episodic memory from a completed conversation episode"""
    try:
        # Format conversation for analysis
        conversation_text = format_conversation_for_memory(episode_data.get("messages", []))
        
        if not conversation_text.strip():
            logger.warning(f"Empty conversation for episode {episode_data.get('episode_id')}")
            return None
        
        # Generate memory using Gemini
        prompt = life_coach_prompt_template.format(conversation=conversation_text)
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        # Parse JSON response
        try:
            memory_data = json.loads(response.text.strip())
        except json.JSONDecodeError:
            # Try to extract JSON from response if it's wrapped in other text
            response_text = response.text.strip()
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                memory_data = json.loads(response_text[start_idx:end_idx])
            else:
                raise
        
        # Validate required fields
        required_fields = ["context_tags", "conversation_summary", "what_worked", "what_to_avoid"]
        if not all(field in memory_data for field in required_fields):
            logger.error(f"Missing required fields in memory data: {memory_data}")
            return None
        
        # Create complete memory document
        memory_document = {
            "episode_id": episode_data["episode_id"],
            "user_id": episode_data["user_id"],
            "username": episode_data.get("username"),
            "created_at": datetime.datetime.now(datetime.timezone.utc),
            "episode_started_at": episode_data.get("started_at"),
            "episode_ended_at": episode_data.get("ended_at"),
            "message_count": len(episode_data.get("messages", [])),
            "context_tags": memory_data["context_tags"],
            "conversation_summary": memory_data["conversation_summary"],
            "what_worked": memory_data["what_worked"],
            "what_to_avoid": memory_data["what_to_avoid"],
            "processed": True
        }
        
        return memory_document
        
    except Exception as e:
        logger.error(f"Error creating episodic memory for episode {episode_data.get('episode_id')}: {e}")
        return None

def save_episodic_memory(memory_document: dict) -> bool:
    """Save episodic memory to MongoDB"""
    try:
        # Check if memory already exists for this episode
        existing_memory = memory_collection.find_one({"episode_id": memory_document["episode_id"]})
        if existing_memory:
            logger.info(f"Memory already exists for episode {memory_document['episode_id']}")
            return True
        
        # Insert new memory
        result = memory_collection.insert_one(memory_document)
        logger.info(f"Saved episodic memory for episode {memory_document['episode_id']}: {result.inserted_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving episodic memory: {e}")
        return False

def process_completed_episodes():
    """Process all completed episodes that don't have memories yet"""
    try:
        # Find completed episodes without memories
        completed_episodes = journal_collection.find({
            "ended_at": {"$exists": True},
            "messages": {"$exists": True, "$not": {"$size": 0}}
        })
        
        processed_count = 0
        for episode in completed_episodes:
            # Check if memory already exists
            existing_memory = memory_collection.find_one({"episode_id": episode["episode_id"]})
            if existing_memory:
                continue
            
            # Create and save memory
            memory_document = create_episodic_memory(episode)
            if memory_document and save_episodic_memory(memory_document):
                processed_count += 1
                logger.info(f"Processed memory for episode {episode['episode_id']}")
        
        if processed_count > 0:
            logger.info(f"Processed {processed_count} new episodic memories")
            
    except Exception as e:
        logger.error(f"Error processing completed episodes: {e}")

# --- Gemini Response Generator with Working Memory ---
def generate_reply_from_gemini(user_input: str, conversation_history: list = None) -> str:
    try:
        # Build conversation context
        context_prompt = "Act as a life coach. Provide thoughtful and empathetic responses to user queries."
        
        if conversation_history and len(conversation_history) > 0:
            context_prompt += "\n\nHere's the conversation history from this session:\n"
            for msg in conversation_history:
                sender = "User" if msg["sender"] == "user" else "Assistant"
                timestamp = msg["timestamp"].strftime("%H:%M")
                context_prompt += f"[{timestamp}] {sender}: {msg['text']}\n"
            context_prompt += f"\nNow respond to the user's latest message: {user_input}"
        else:
            context_prompt += f"\n\nUser message: {user_input}"
        
        model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=context_prompt)
        response = model.generate_content(user_input)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error generating reply from Gemini: {e}")
        return "Sorry, I'm having trouble coming up with a response right now."

# --- Helper function to get conversation history ---
def get_episode_history(episode_id: str) -> list:
    """Retrieve conversation history for the current episode"""
    try:
        episode = journal_collection.find_one({"episode_id": episode_id})
        if episode and "messages" in episode:
            return episode["messages"]
        return []
    except Exception as e:
        logger.error(f"Error retrieving episode history: {e}")
        return []

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

# --- Background Task to Monitor Timeouts and Process Memories ---
async def monitor_idle_users():
    logger.info("Starting user idle monitoring...")
    while True:
        now = datetime.datetime.now(datetime.timezone.utc)
        for user_id, info in list(user_states.items()):
            if info["state"] in [EpisodeState.ACTIVE, EpisodeState.AWAITING_END]:
                if now - info["last_activity"] > IDLE_TIMEOUT:
                    logger.info(f"User {user_id} timed out, transitioning state...")
                    transition_state(user_id, None, "timeout")
        await asyncio.sleep(60)  # Check every minute

async def process_episodic_memories():
    """Background task to process completed episodes into episodic memories"""
    logger.info("Starting episodic memory processing...")
    while True:
        try:
            process_completed_episodes()
        except Exception as e:
            logger.error(f"Error in episodic memory processing: {e}")
        
        # Process memories every 5 minutes
        await asyncio.sleep(300)

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

    # Get conversation history for working memory
    conversation_history = get_episode_history(episode_id)
    
    # Get Gemini reply with working memory context
    gemini_reply = generate_reply_from_gemini(message_text, conversation_history)
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
    
    # Start background tasks
    monitor_task = asyncio.create_task(monitor_idle_users())
    memory_task = asyncio.create_task(process_episodic_memories())
    
    try:
        # Initialize and start the bot
        await app.initialize()
        await app.start()
        
        logger.info("Bot is starting...")
        
        # Start polling for updates
        await app.updater.start_polling()
        
        # Keep the application running
        await asyncio.Event().wait()
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    finally:
        # Clean shutdown
        monitor_task.cancel()
        memory_task.cancel()
        await app.stop()
        await app.shutdown()

if __name__ == '__main__':
    asyncio.run(main())