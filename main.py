import os
from dotenv import load_dotenv
from urllib.parse import quote_plus

load_dotenv()

# --- MongoDB Configuration ---
username = quote_plus(os.getenv("MONGO_USERNAME"))
password = quote_plus(os.getenv("MONGO_PASSWORD"))
uri = f"mongodb+srv://{username}:{password}@cluster0.xjcfm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# ... rest of your code ...


def main():
    print("Hello from coach!")
    print("Lets start making LLM Integration!!")

if __name__ == "__main__":
    main()
