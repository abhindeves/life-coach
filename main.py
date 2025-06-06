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
# Import glm for embedding functionalities
import google.generativeai.types as glm

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

# --- Define Gemini Embedding Model ---
# Using the best available general-purpose embedding model for now
EMBEDDING_MODEL = "models/embedding-001" 

# --- MongoDB Connection ---
try:
    client = MongoClient(uri)
    db = client[DB_NAME]
    journal_collection = db[COLLECTION_NAME]
    memory_collection = db[MEMORY_COLLECTION_NAME]
    # Ensure a vector index is available for future similarity searches if you switch to Atlas Vector Search
    # For a basic setup, you might not need to define it here, but it's good practice.
    # If you use Atlas Vector Search, you'd define the index in MongoDB Atlas UI.
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

# --- Memory Processing Configuration ---
MEMORY_BATCH_SIZE = 10  # Process max 10 episodes per cycle
MEMORY_PROCESSING_INTERVAL = 60  # 1 minutes

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
    "context_tags": [             // 2–4 keywords or phrases lifted directly from the user's concerns or goals
        string,
        ...
    ],
    "conversation_summary": string, // One sentence summarizing the user's main concern or progress
    "what_worked": string,        // The most effective thing said or done in the conversation
    "what_to_avoid": string       // The least effective or unhelpful moment in the conversation
}}
Guidelines:
- Pull actual keywords or short phrases from the conversation for context_tags, e.g., ["procrastination", "gym_routine", "lack_of_motivation"]
- Don't use vague tags like ["life_coaching", "feelings", "helpful_talk"]
- Use direct user quotes or paraphrased terms from the session
Here is the conversation:
{conversation}
"""

# --- Embedding Function ---
async def embed_text_with_gemini(text: str, task_type = "semantic_similarity") -> list:
    """Generates embeddings for a given text using Gemini embedding model."""
    try:
        # The embed_content function is designed for efficient embedding
        response = await genai.embed_content_async(
            model=EMBEDDING_MODEL,
            content=text,
            task_type=task_type
        )
        # The embedding is a list of floats
        return response['embedding']
    except Exception as e:
        logger.error(f"Error generating embedding for text: '{text[:50]}...' Error: {e}")
        return []

# --- Episodic Memory Functions ---
def format_conversation_for_memory(messages: list) -> str:
    """Format conversation messages into a readable string for memory processing"""
    formatted_conversation = ""
    for msg in messages:
        sender = "User" if msg["sender"] == "user" else "Coach"
        timestamp = msg["timestamp"].strftime("%H:%M")
        formatted_conversation += f"[{timestamp}] {sender}: {msg['text']}\n"
    return formatted_conversation

async def create_episodic_memory(episode_data: dict) -> dict:
    """
    Create episodic memory from a completed conversation episode,
    including generating embeddings for the summary and tags.
    """
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
        
        # --- Generate Embeddings for Memory Data ---
        # Combine relevant fields for a comprehensive embedding of the memory
        combined_memory_text = (
            f"Summary: {memory_data['conversation_summary']}. "
            f"Context Tags: {', '.join(memory_data['context_tags'])}. "
            f"What worked: {memory_data['what_worked']}. "
            f"What to avoid: {memory_data['what_to_avoid']}."
        )
        
        memory_embedding = await embed_text_with_gemini(
            combined_memory_text,
            task_type= "semantic_similarity"# This memory will be retrieved as a document
        )

        if not memory_embedding:
            logger.error(f"Failed to generate embedding for episode {episode_data.get('episode_id')}")
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
            "embedding": memory_embedding, # Store the embedding here!
            "processed": True
        }
        
        return memory_document
        
    except Exception as e:
        logger.error(f"Error creating episodic memory for episode {episode_data.get('episode_id')}: {e}")
        return None

def save_episodic_memory(memory_document: dict) -> bool:
    """Save episodic memory to MongoDB (with duplicate checking)"""
    try:
        # Check if memory already exists for this episode
        existing_memory = memory_collection.find_one(
            {"episode_id": memory_document["episode_id"]},
            {"_id": 1}  # Only get the ID to check existence
        )
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

async def process_completed_episodes():
    """Process completed episodes that don't have memories yet (with efficient batching)"""
    try:
        # Find completed episodes without memories using efficient query
        # Only get episodes that are ended, have messages, and haven't been processed yet
        query = {
            "ended_at": {"$exists": True},
            "messages": {"$exists": True, "$not": {"$size": 0}},
            "memory_generated": {"$ne": True}  # Only episodes not yet processed
        }
        
        # Project only necessary fields to reduce memory usage
        projection = {
            "episode_id": 1,
            "user_id": 1,
            "username": 1,
            "started_at": 1,
            "ended_at": 1,
            "messages": 1
        }
        
        # Use pagination for efficient processing
        completed_episodes = journal_collection.find(
            query, 
            projection
        ).sort("ended_at", 1).limit(MEMORY_BATCH_SIZE)  # Process oldest first
        
        processed_count = 0
        failed_count = 0
        
        # Collect episodes to process them asynchronously
        episodes_to_process = list(completed_episodes)
        
        # Use asyncio.gather to run memory creation for multiple episodes concurrently
        memory_creation_tasks = [create_episodic_memory(episode) for episode in episodes_to_process]
        memory_documents = await asyncio.gather(*memory_creation_tasks, return_exceptions=True)

        for i, memory_document in enumerate(memory_documents):
            episode = episodes_to_process[i] # Get the original episode data
            
            if isinstance(memory_document, Exception):
                logger.error(f"Error processing episode {episode.get('episode_id')}: {memory_document}")
                # Mark as failed
                journal_collection.update_one(
                    {"episode_id": episode["episode_id"]},
                    {"$set": {"memory_generation_failed": True, "memory_failed_at": datetime.datetime.now(datetime.timezone.utc)}}
                )
                failed_count += 1
            elif memory_document and save_episodic_memory(memory_document):
                # Mark episode as processed
                journal_collection.update_one(
                    {"episode_id": episode["episode_id"]},
                    {"$set": {"memory_generated": True, "memory_processed_at": datetime.datetime.now(datetime.timezone.utc)}}
                )
                processed_count += 1
                logger.info(f"Processed memory for episode {episode['episode_id']}")
            else:
                # Mark as failed but don't retry immediately
                journal_collection.update_one(
                    {"episode_id": episode["episode_id"]},
                    {"$set": {"memory_generation_failed": True, "memory_failed_at": datetime.datetime.now(datetime.timezone.utc)}}
                )
                failed_count += 1
                logger.warning(f"Failed to process memory for episode {episode['episode_id']}")
                        
        if processed_count > 0 or failed_count > 0:
            logger.info(f"Memory processing cycle: {processed_count} processed, {failed_count} failed")
            
    except Exception as e:
        logger.error(f"Error in process_completed_episodes: {e}")

# --- Gemini Response Generator with Working Memory ---
async def generate_reply_from_gemini(user_input: str, conversation_history: list = None, user_id: int = None, top_k: int = 5) -> str:
    try:
        # --- Future Enhancement: Retrieval Augmented Generation (RAG) ---
        # 1. Embed the user's current query
        user_query_embedding = await embed_text_with_gemini(user_input, task_type="semantic_similarity")
        
        # 2. Use this embedding to query your episodic_memories collection for similar memories
        #    This would typically involve a vector search (e.g., using MongoDB Atlas Vector Search) 
        pipeline = [  
            {  
                "$vectorSearch": {  
                    "index": "vector_index", 
                    "path": "embedding",
                    "queryVector": user_query_embedding,   
                    "numCandidates": 50,   
                    "limit": 2  
                }  
            },  
            {  
                "$project":  {
                    "episode_id": 1,
                    "user_id": 1,
                    "username": 1,
                    "context_tags": 1,
                    "conversation_summary": 1,
                    "what_worked": 1,
                    "what_to_avoid": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }   
                
            }  
        ]
        score_threshold = 0.8  # Adjust based on your needs
        try:
            results = list(memory_collection.aggregate(pipeline))
            filtered_results = [res for res in results if res.get("score", 0) >= score_threshold]
            results = filtered_results if filtered_results else []
        except Exception as e:
            logger.error(f"Error executing aggregation pipeline: {e}")
            return "Sorry, I couldn't retrieve relevant information at the moment."

        # 3. Format retrieved memories to include in the prompt
        memory_context = ""
        if results:
            memory_context = "\n\nRelevant past coaching insights (for context, do not directly quote):\n"
            for mem in results:
                if mem.get("conversation_summary"):  # Only add non-empty summaries
                    memory_context += f"- Summary: {mem.get('conversation_summary', '')}. Worked: {mem.get('what_worked', '')}. Avoid: {mem.get('what_to_avoid', '')}. Tags: {', '.join(mem.get('context_tags', []))}\n"

        # Build conversation context
        context_prompt = "Act as a life coach. Provide thoughtful and empathetic responses to user queries.\n\nAlways Try to reference to the user's past experiences. and insights from previous conversations. If there is no Previous conversations give then start fresh.\n\n"
        
        if memory_context:
           context_prompt += memory_context

        if conversation_history and len(conversation_history) > 0:
            context_prompt += "\n\nHere's the conversation history from this session:\n"
            for msg in conversation_history:
                sender = "User" if msg["sender"] == "user" else "Assistant"
                timestamp = msg["timestamp"].strftime("%H:%M")
                context_prompt += f"[{timestamp}] {sender}: {msg['text']}\n"
            context_prompt += f"\nNow respond to the user's latest message: {user_input}"
        else:
            context_prompt += f"\n\nUser message: {user_input}"
            
        print(context_prompt)
        
        model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=context_prompt)
        response = await model.generate_content_async(user_input) # Use async version
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
                "messages": [],
                "memory_generated": False  # Flag for memory processing
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

async def process_episodic_memories_task(): # Renamed to avoid confusion with the internal function
    """Background task to process completed episodes into episodic memories (with batching)"""
    logger.info("Starting episodic memory processing...")
    while True:
        try:
            await process_completed_episodes() # Await this now that it's async
        except Exception as e:
            logger.error(f"Error in episodic memory processing: {e}")
        
        # Process memories every 5 minutes
        await asyncio.sleep(MEMORY_PROCESSING_INTERVAL)

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
    # Pass user_id for potential future RAG implementation specific to the user
    gemini_reply = await generate_reply_from_gemini(message_text, conversation_history, user_id) 
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
    # Use the renamed async function for the background task
    memory_task = asyncio.create_task(process_episodic_memories_task()) 
    
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