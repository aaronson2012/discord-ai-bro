# Discord AI Bro

A Discord bot that uses Google's Gemini AI to chat with users in a sassy, witty persona.

## Features

- Responds to direct mentions, replies, and messages containing "aibro" or "ai bro"
- Maintains short-term conversation memory for context-aware responses
- Stores long-term user profiles in Supabase for personalized interactions
- Extracts user preferences and personal details from conversations
- Handles rate limiting for all API calls
- Robust error handling and logging

## Setup

1. Clone this repository
2. Copy `.env.example` to `.env` and fill in your API keys
3. Install dependencies: `pip install -r requirements.txt`
4. Run the bot: `python bot.py`

## Configuration

The bot can be configured using environment variables in the `.env` file:

```
DISCORD_BOT_TOKEN=your_discord_bot_token
GOOGLE_API_KEY=your_google_api_key
GEMINI_MODEL_NAME="gemini-1.5-flash-latest"
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key

# Timeout configurations (in seconds)
API_TIMEOUT_SECONDS=30
GEMINI_TIMEOUT_SECONDS=45
SUPABASE_TIMEOUT_SECONDS=20
```

## Recent Improvements

- Added file-based logging with rotation
- Implemented task manager to prevent memory leaks
- Improved rate limiting with better concurrency handling
- Enhanced JSON parsing for more robust data extraction
- Added connection pooling for database operations
- Implemented graceful shutdown handling
- Added health monitoring
- Added Discord API rate limiting
- Made timeout values configurable

## License

MIT